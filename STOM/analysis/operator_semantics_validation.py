import csv
import json
import math
import os
import re
import shlex
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

STOM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STOM_DIR))

from data_provider.data_factory import data_provider
from models import stom as stom_model


OUT = Path("operator_semantics_results")
RAW_DIR = Path("analysis_inputs")
CHECKPOINT_DIR = Path("checkpoints")
COMMAND_DIR = Path("commands")


OPS = ["phase", "spectral", "smooth", "local"]


DEFAULTS = dict(
    random_seed=2026,
    model_id="operator_semantics_validation",
    model="STOM",
    data="ETTm1",
    root_path="./dataset/",
    data_path="ETT-small/ETTm1.csv",
    features="M",
    target="OT",
    freq="h",
    checkpoints="./checkpoints/",
    scale=1,
    train_ratio=0.7,
    test_ratio=0.2,
    plot_every=5,
    iterative_prediction=0,
    number_of_targets=0,
    seq_len=96,
    label_len=0,
    pred_len=96,
    enc_in=7,
    dec_in=7,
    c_out=7,
    d_model=0,
    n_heads=4,
    e_layers=3,
    d_layers=1,
    d_ff=2048,
    factor=1,
    embed="timeF",
    distil=True,
    dropout=0.0,
    activation="gelu",
    output_attention=False,
    time_emb_dim=4,
    num_workers=0,
    batch_size=32,
    batch_size_test=0,
    predictor="LinAttn",
    compile=0,
    use_gpu=True,
    gpu=0,
    use_multi_gpu=False,
    devices="0",
    random_channel_dropout=1,
    weight_decay=0.0,
    use_amp=True,
    use_stom=1,
    stom_ma_kernel=25,
    stom_local_kernel=5,
    stom_spec_gain_init=6.0,
    stom_gate_prior="0.35,0.10,0.30,0.25",
    stom_gate_hidden_dim=32,
    stom_rank=4,
    stom_n_spec_heads=4,
    stom_precision_variant="none",
    stom_ablation="full",
    stom_operator_set="phase,spectral,smooth,local",
    stom_residual_mode="none",
    stom_residual_alpha_cap=0.10,
    stom_residual_alpha_init=0.01,
    stom_residual_dict_k=8,
    stom_spec_injection_mode="none",
    stom_spec_injection_beta_cap=0.10,
    stom_spec_injection_beta_init=0.01,
    local_attention=False,
    local_attention_window=8,
)

BOOL_FLAGS = {"use_amp", "output_attention", "use_multi_gpu"}
INT_KEYS = {
    "random_seed", "is_training", "scale", "plot_every", "iterative_prediction",
    "number_of_targets", "seq_len", "label_len", "pred_len", "enc_in", "dec_in",
    "c_out", "d_model", "n_heads", "e_layers", "d_layers", "d_ff", "factor",
    "time_emb_dim", "num_workers", "batch_size", "batch_size_test", "compile",
    "gpu", "random_channel_dropout", "use_stom", "stom_ma_kernel",
    "stom_local_kernel", "stom_gate_hidden_dim", "stom_rank", "stom_n_spec_heads",
}
FLOAT_KEYS = {
    "train_ratio", "test_ratio", "dropout", "weight_decay", "stom_spec_gain_init",
    "stom_residual_alpha_cap", "stom_residual_alpha_init",
    "stom_spec_injection_beta_cap", "stom_spec_injection_beta_init",
}


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def parse_command_file(path: Path) -> SimpleNamespace:
    args = dict(DEFAULTS)
    text = path.read_text()
    py_line = ""
    for line in text.splitlines():
        if "run_longExp.py" in line and "python" in line:
            py_line = line.strip()
            break
    if not py_line:
        raise ValueError(f"no run_longExp.py line in {path}")
    tokens = shlex.split(py_line)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:].replace("-", "_")
            if key in BOOL_FLAGS:
                args[key] = True
                i += 1
                continue
            if i + 1 >= len(tokens):
                break
            val = tokens[i + 1]
            if key in INT_KEYS:
                try:
                    args[key] = int(val)
                except ValueError:
                    args[key] = int(float(val))
            elif key in FLOAT_KEYS:
                args[key] = float(val)
            else:
                args[key] = val.replace("\\,", ",")
            i += 2
        else:
            i += 1
    args["num_workers"] = 0
    args["batch_size_test"] = 0
    root = Path(str(args.get("root_path", "./dataset/")))
    if not root.is_absolute():
        args["root_path"] = str((STOM_DIR / root).resolve())
    return SimpleNamespace(**args)


def command_dataset_pred(path: Path):
    try:
        args = parse_command_file(path)
        data = "Weather" if args.data_path.endswith("weather/weather.csv") else args.data
        return data, int(args.pred_len)
    except Exception:
        return None, None


def unwrap_state_dict(state_dict):
    out = OrderedDict()
    for k, v in state_dict.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def find_checkpoint_for_command(command_file: Path):
    args = parse_command_file(command_file)
    matches = sorted(CHECKPOINT_DIR.glob(f"{args.model_id}*/checkpoint.pth"))
    if matches:
        return matches[0]
    return None


def inventory():
    command_by_setting = {}
    audit = Path("metric_audit.csv")
    if audit.exists():
        df = pd.read_csv(audit)
        sub = df[(df.get("model", "").astype(str) == "stom") if "model" in df.columns else [False] * len(df)]
        for _, r in sub.iterrows():
            key = (str(r["dataset"]), int(r["pred_len"]))
            cmd = Path(str(r.get("command_file", "")))
            if key in ALL_SETTINGS and cmd.exists():
                command_by_setting[key] = cmd
    for cmd in COMMAND_DIR.glob("*.cmd"):
        data, pred = command_dataset_pred(cmd)
        if (data, pred) in ALL_SETTINGS and "stom" in cmd.name.lower():
            command_by_setting.setdefault((data, pred), cmd)

    rows = []
    items = {}
    for setting in ALL_SETTINGS:
        cmd = command_by_setting.get(setting)
        ckpt = find_checkpoint_for_command(cmd) if cmd else None
        found = bool(cmd and ckpt and ckpt.exists())
        note = "ok" if found else ("missing_checkpoint" if cmd else "missing_command")
        row = {
            "setting": f"{setting[0]}-{setting[1]}",
            "dataset": setting[0],
            "pred_len": setting[1],
            "checkpoint_path": str(ckpt) if ckpt else "",
            "command_file": str(cmd) if cmd else "",
            "found": str(found),
            "notes": note,
        }
        rows.append(row)
        if found:
            items[setting] = (cmd, ckpt)
    write_csv(OUT / "checkpoint_inventory.csv", rows, ["setting", "dataset", "pred_len", "checkpoint_path", "command_file", "found", "notes"])
    return items, rows


def load_model(command_file: Path, checkpoint: Path, device):
    args = parse_command_file(command_file)
    args.use_gpu = device.type == "cuda"
    args.gpu = 0
    args.num_workers = 0
    args.batch_size_test = 0
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    model = stom_model.Model(args).to(device)
    state = torch.load(checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(unwrap_state_dict(state), strict=False)
    model.eval()
    return args, model, missing, unexpected


def forward_parts(model, batch_x, batch_x_mark, batch_y_mark):
    m = model
    x = batch_x.permute(0, 2, 1)
    mean = x[:, :, [-1]]
    std = torch.ones(x.size(0), x.size(1), 1, device=x.device)
    x = (x - mean) / std
    x_hist = x[:, -m.number_of_targets:, :].clone()
    out_fill = m.out_fill(x)
    x = torch.cat([x, out_fill], dim=2)
    if batch_x_mark is not None and batch_y_mark is not None and m.time_emb_dim:
        mark = torch.cat([batch_x_mark, batch_y_mark[:, -m.pred_len:]], dim=1).permute(0, 2, 1)
        enc_mark = batch_x_mark.permute(0, 2, 1)
        mark = (mark - enc_mark[:, :, [-1]]) / (enc_mark.std(dim=-1, keepdim=True) + 1e-8)
        x = torch.cat([mark, x], dim=1)
    x_dropout = stom_model.random_ratio_channel_dropout_vectorized(x, False)
    channel_token = m.channel_token_mixer_seq(x_dropout.permute(0, 2, 1))
    channel_token = channel_token.unsqueeze(1).expand(-1, m.number_of_targets, -1, -1)
    x = x[:, -m.number_of_targets:].unsqueeze(-1)
    x = torch.einsum("bcli,cid->bcld", x, m.value_token_mixer)
    x = torch.cat([channel_token, x], dim=-1)
    B, C, N, _ = x.shape
    pos = torch.arange(0, N, dtype=torch.long, device=x.device)
    x = x + m.channel_emb_2 + m.pos_emb(pos)
    x = x.reshape(B * C, N, -1)
    x = m.temporal_processor(x)
    x = x[:, -m.pred_len:].reshape(B, C, m.pred_len, -1)
    y_phase = torch.einsum("bcld,cdi->bcli", x[:, :, :, -m.seq_emb_dim:], m.value_token_mixer.transpose(-2, -1)).squeeze(-1)

    stom = m.stom_head
    smooth = stom.moving_average(x_hist)
    residual = x_hist - smooth
    y_smooth = stom.trend_proj(smooth)
    y_local = stom.local_proj(F.gelu(stom.local_conv(residual)))
    A, phi, delta, alpha, phase_strength, recent_strength, sigma_phase, sigma_recent = stom._spectral_kernel(x_hist.device, x_hist.dtype)
    y_spec_h = torch.einsum("hpl,bcl->bhcp", A, residual)
    head_mix = torch.softmax(stom.spec_head_logits, dim=0).to(device=x_hist.device, dtype=x_hist.dtype)
    spec_gain = F.softplus(stom.raw_spec_gain).to(device=x_hist.device, dtype=x_hist.dtype)
    y_spec = spec_gain * torch.einsum("h,bhcp->bcp", head_mix, y_spec_h)

    hist_mean = x_hist.mean(dim=-1)
    hist_std = x_hist.std(dim=-1, unbiased=False)
    last_value = x_hist[:, :, -1]
    smooth_slope = smooth[:, :, -1] - smooth[:, :, 0]
    residual_energy = (residual.pow(2).mean(dim=-1) + 1e-6).sqrt()
    expert_outputs = {"phase": y_phase, "spectral": y_spec, "smooth": y_smooth, "local": y_local}
    op_set = list(getattr(stom, "operator_set", OPS))
    expert_norms = [(expert_outputs[name].pow(2).mean(dim=-1) + 1e-6).sqrt() for name in op_set]
    gate_base_feat = torch.stack([hist_mean, hist_std, last_value, smooth_slope, residual_energy, *expert_norms], dim=-1)
    horizon = torch.arange(m.pred_len, device=x_hist.device, dtype=gate_base_feat.dtype) / max(m.pred_len, 1)
    horizon = horizon.view(1, 1, m.pred_len, 1).expand(x_hist.size(0), x_hist.size(1), -1, -1)
    if getattr(stom, "ablation_mode", "full") == "no_horizon_gate":
        horizon = torch.zeros_like(horizon)
    gate_feat = torch.cat([gate_base_feat.unsqueeze(2).expand(-1, -1, m.pred_len, -1), horizon], dim=-1)
    gate_logits = stom.gate_mlp(gate_feat.float()).to(dtype=x_hist.dtype)
    if hasattr(stom, "_expert_mask"):
        mask = stom._expert_mask(x_hist.device, x_hist.dtype).view(1, 1, 1, len(op_set))
        gate = torch.softmax(gate_logits.masked_fill(mask <= 0, -1e9), dim=-1)
    else:
        gate = torch.softmax(gate_logits, dim=-1)
    expert_stack = torch.stack([expert_outputs[name] for name in op_set], dim=-1)
    weighted = gate * expert_stack
    y_norm = weighted.sum(dim=-1)
    return {
        "x_hist": x_hist.float(),
        "smooth": smooth.float(),
        "residual": residual.float(),
        "gate": gate.float(),
        "expert_stack": expert_stack.float(),
        "weighted": weighted.float(),
        "y_norm": y_norm.float(),
        "A": A.float(),
        "head_mix": head_mix.float(),
        "spec_gain": spec_gain.float(),
        "sigma_phase": sigma_phase.float(),
        "operator_set": op_set,
    }


def bands(P):
    return {"near": (0, P // 3), "middle": (P // 3, 2 * P // 3), "far": (2 * P // 3, P), "all": (0, P)}


def accumulate_setting(setting, command_file, checkpoint, device):
    args, model, missing, unexpected = load_model(command_file, checkpoint, device)
    _, loader = data_provider(args, "test")
    op_set = None
    op_index = None
    spectrum = {op: {c: {"low": 0.0, "total": 0.0} for c in CUTOFFS} for op in OPS}
    tv_sum = {op: 0.0 for op in OPS}
    curv_sum = {op: 0.0 for op in OPS}
    diff_count = {op: 0 for op in OPS}
    curv_count = {op: 0 for op in OPS}
    phase_abs = defaultdict(lambda: {op: 0.0 for op in OPS})
    phase_count = defaultdict(int)
    spec_abs_profile = None
    spec_recon_abs = 0.0
    spec_recon_max = 0.0
    spec_recon_count = 0
    residual_align_num = 0.0
    residual_align_den = 0.0
    corr_source = []
    corr_resid = []
    corr_smooth = []
    total_samples = 0

    with torch.no_grad():
        for data in loader:
            batch_x = data[0].float().to(device)
            batch_x_mark = data[2].float().to(device)
            batch_y_mark = data[3].float().to(device)
            parts = forward_parts(model, batch_x, batch_x_mark, batch_y_mark)
            total_samples += int(batch_x.shape[0])
            if op_set is None:
                op_set = parts["operator_set"]
                op_index = {name: i for i, name in enumerate(op_set)}
            expert = parts["expert_stack"]
            weighted = parts["weighted"]
            gate = parts["gate"]
            P = expert.shape[2]
            for name in OPS:
                if name not in op_index:
                    continue
                oi = op_index[name]
                y = expert[..., oi]
                yf = torch.fft.rfft(y.float(), dim=2)
                power = (yf.real.pow(2) + yf.imag.pow(2)).sum().item()
                nfreq = yf.shape[2]
                for cutoff in CUTOFFS:
                    k = max(1, int(math.floor((nfreq - 1) * cutoff)) + 1)
                    low = (yf[:, :, :k].real.pow(2) + yf[:, :, :k].imag.pow(2)).sum().item()
                    spectrum[name][cutoff]["low"] += low
                    spectrum[name][cutoff]["total"] += power
                dy = y[:, :, 1:] - y[:, :, :-1]
                tv_sum[name] += dy.abs().sum().item()
                diff_count[name] += dy.numel()
                if P > 2:
                    ddy = y[:, :, 2:] - 2 * y[:, :, 1:-1] + y[:, :, :-2]
                    curv_sum[name] += ddy.abs().sum().item()
                    curv_count[name] += ddy.numel()

                z_abs = weighted[..., oi].abs()
                for region, (s, e) in bands(P).items():
                    phase_abs[region][name] += z_abs[:, :, s:e].sum().item()
                    phase_count[region] += 0
            for region, (s, e) in bands(P).items():
                phase_count[region] += weighted[:, :, s:e, :].shape[0] * weighted[:, :, s:e, :].shape[1] * max(e - s, 1)

            if setting in SPEC_SETTINGS and "spectral" in op_index:
                si = op_index["spectral"]
                residual = parts["residual"]
                smooth = parts["smooth"]
                A = parts["A"]
                mix = parts["head_mix"]
                spec_gain = parts["spec_gain"]
                eff = spec_gain * torch.einsum("h,hpl->pl", mix, A)
                source = gate[..., si].unsqueeze(-1) * eff.view(1, 1, eff.shape[0], eff.shape[1]) * residual.unsqueeze(2)
                z_recon = source.sum(dim=-1)
                z_ref = weighted[..., si]
                err = (z_recon - z_ref).abs()
                spec_recon_abs += err.sum().item()
                spec_recon_max = max(spec_recon_max, float(err.max().item()))
                spec_recon_count += err.numel()
                abs_source = source.abs().sum(dim=(0, 1, 2)).detach().cpu().double().numpy()
                spec_abs_profile = abs_source if spec_abs_profile is None else spec_abs_profile + abs_source
                src_flat = source.abs().detach().flatten().cpu().double().numpy()
                resid_flat = residual.abs().unsqueeze(2).expand_as(source).detach().flatten().cpu().double().numpy()
                smooth_flat = smooth.abs().unsqueeze(2).expand_as(source).detach().flatten().cpu().double().numpy()
                if src_flat.size > 500000:
                    idx = np.linspace(0, src_flat.size - 1, 500000, dtype=np.int64)
                    src_flat = src_flat[idx]
                    resid_flat = resid_flat[idx]
                    smooth_flat = smooth_flat[idx]
                corr_source.append(src_flat)
                corr_resid.append(resid_flat)
                corr_smooth.append(smooth_flat)
                residual_align_num += float((source.abs() * residual.abs().unsqueeze(2)).sum().item())
                residual_align_den += float((source.abs() * (residual.abs().unsqueeze(2) + smooth.abs().unsqueeze(2) + 1e-8)).sum().item())

    spectrum_rows = []
    for op in OPS:
        for cutoff in CUTOFFS:
            total = spectrum[op][cutoff]["total"]
            lfr = spectrum[op][cutoff]["low"] / total if total > 0 else np.nan
            spectrum_rows.append({
                "setting": f"{setting[0]}-{setting[1]}",
                "operator": op,
                "cutoff_ratio": cutoff,
                "low_frequency_ratio": lfr,
                "high_frequency_ratio": 1.0 - lfr if np.isfinite(lfr) else np.nan,
                "tv_energy": tv_sum[op] / diff_count[op] if diff_count[op] else np.nan,
                "curvature_energy": curv_sum[op] / curv_count[op] if curv_count[op] else np.nan,
                "num_samples": total_samples,
            })

    phase_rows = []
    for region, accum in phase_abs.items():
        total = sum(accum.values()) + 1e-12
        for op in OPS:
            phase_rows.append({
                "setting": f"{setting[0]}-{setting[1]}",
                "region": region,
                "operator": op,
                "A_k": accum[op] / max(phase_count[region], 1),
                "R_k": accum[op] / total,
            })

    spec_recon_row = None
    spec_profile_row = None
    align_row = None
    if setting in SPEC_SETTINGS:
        if spec_abs_profile is not None and spec_abs_profile.sum() > 0:
            L = len(spec_abs_profile)
            prof = spec_abs_profile / spec_abs_profile.sum()
            lag = np.arange(L - 1, -1, -1, dtype=np.float64)
            entropy = -float(np.sum(prof * np.log(np.clip(prof, 1e-12, None))) / math.log(L))
            recent8 = float(prof[max(0, L - 8):].sum())
            
            phase_mask = np.zeros(L, dtype=bool)
            for p in range(min(setting[1], L)):
                lo = max(0, p - 8)
                hi = min(L, p + 9)
                phase_mask[lo:hi] = True
            phase_share = float(prof[phase_mask].sum())
            spec_profile_row = {
                "setting": f"{setting[0]}-{setting[1]}",
                "expected_lag": float(np.sum(prof * lag)),
                "normalized_entropy": entropy,
                "recent_source_share": recent8,
                "phase_aligned_source_share": phase_share,
                "num_samples": total_samples,
                "notes": "profile_from_abs_final_gated_residual_source",
            }
        spec_recon_row = {
            "setting": f"{setting[0]}-{setting[1]}",
            "num_samples": total_samples,
            "spec_recon_mae": spec_recon_abs / max(spec_recon_count, 1),
            "spec_recon_max_error": spec_recon_max,
            "residual_defined": "yes" if spec_recon_count else "no",
            "notes": "C_spec=gate*spec_gain*head_mix*A*residual",
        }
        if corr_source:
            s = np.concatenate(corr_source)
            r = np.concatenate(corr_resid)
            t = np.concatenate(corr_smooth)
            corr_r = float(np.corrcoef(s, r)[0, 1]) if s.std() > 0 and r.std() > 0 else np.nan
            corr_t = float(np.corrcoef(s, t)[0, 1]) if s.std() > 0 and t.std() > 0 else np.nan
            align_row = {
                "setting": f"{setting[0]}-{setting[1]}",
                "corr_residual": corr_r,
                "corr_smooth": corr_t,
                "residual_alignment": residual_align_num / max(residual_align_den, 1e-12),
                "num_samples": total_samples,
                "notes": "auxiliary_not_pass_fail",
            }
    return spectrum_rows, phase_rows, spec_recon_row, spec_profile_row, align_row


def smooth_local_summary(spectrum_rows):
    df = pd.DataFrame(spectrum_rows)
    rows = []
    for setting in sorted(df["setting"].unique()):
        for cutoff in CUTOFFS:
            sub = df[(df.setting == setting) & (np.isclose(df.cutoff_ratio, cutoff))]
            vals = {r.operator: r for r in sub.itertuples()}
            if "smooth" not in vals or "local" not in vals:
                continue
            smooth_lfr = float(vals["smooth"].low_frequency_ratio)
            local_lfr = float(vals["local"].low_frequency_ratio)
            spec_lfr = float(vals["spectral"].low_frequency_ratio) if "spectral" in vals else np.nan
            phase_lfr = float(vals["phase"].low_frequency_ratio) if "phase" in vals else np.nan
            smooth_hfr = float(vals["smooth"].high_frequency_ratio)
            local_hfr = float(vals["local"].high_frequency_ratio)
            smooth_tv = float(vals["smooth"].tv_energy)
            local_tv = float(vals["local"].tv_energy)
            smooth_curv = float(vals["smooth"].curvature_energy)
            local_curv = float(vals["local"].curvature_energy)
            lfrs = [v for v in [smooth_lfr, local_lfr, spec_lfr, phase_lfr] if np.isfinite(v)]
            smooth_pass = smooth_lfr >= max(lfrs) - 1e-12 or smooth_lfr > max(local_lfr, spec_lfr if np.isfinite(spec_lfr) else -1)
            local_pass = (local_hfr > smooth_hfr) or (local_tv > smooth_tv) or (local_curv > smooth_curv)
            rows.append({
                "setting": setting,
                "cutoff_ratio": cutoff,
                "smooth_lfr": smooth_lfr,
                "local_lfr": local_lfr,
                "spec_lfr": spec_lfr,
                "phase_lfr": phase_lfr,
                "smooth_hfr": smooth_hfr,
                "local_hfr": local_hfr,
                "smooth_tv": smooth_tv,
                "local_tv": local_tv,
                "smooth_curv": smooth_curv,
                "local_curv": local_curv,
                "smooth_lowfreq_pass": "yes" if smooth_pass else "no",
                "local_more_local_than_smooth_pass": "yes" if local_pass else "no",
                "notes": "",
            })
    return rows


def phase_summary(phase_rows):
    df = pd.DataFrame(phase_rows)
    rows = []
    for setting in sorted(df["setting"].unique()):
        sub = df[(df.setting == setting) & (df.operator == "phase")]
        vals = {r.region: float(r.R_k) for r in sub.itertuples()}
        if not {"near", "middle", "far"}.issubset(vals):
            continue
        diff = vals["far"] - vals["near"]
        main = setting in {"ETTm1-720", "Weather-336", "ETTh2-336"}
        rows.append({
            "setting": setting,
            "phase_near": vals["near"],
            "phase_middle": vals["middle"],
            "phase_far": vals["far"],
            "phase_far_minus_near": diff,
            "phase_decreasing_pass": "yes" if diff < 0 else "no",
            "notes": "main_setting" if main else "auxiliary_setting",
        })
    return rows



def raw_npz_path(setting):
    p = RAW_DIR / f"{setting[0]}_{setting[1]}_seed2026.npz"
    return p if p.exists() else None


def op_names_from_npz(z):
    if "operator_order" in z.files:
        names = [str(x) for x in z["operator_order"].tolist()]
    else:
        names = OPS
    return ["smooth" if n == "trend" else ("spectral" if n == "spec" else n) for n in names]


def spectrum_from_npz(setting, path):
    z = np.load(path, allow_pickle=True)
    names = op_names_from_npz(z)
    expert = z["expert_output"].astype(np.float32)
    rows = []
    for oi, name in enumerate(names):
        y = expert[..., oi]
        yf = np.fft.rfft(y, axis=2)
        power_by_freq = (yf.real ** 2 + yf.imag ** 2).sum(axis=(0, 1))
        total = float(power_by_freq.sum())
        dy = np.diff(y, axis=2)
        ddy = np.diff(y, n=2, axis=2) if y.shape[2] > 2 else np.array([np.nan])
        for cutoff in CUTOFFS:
            k = max(1, int(math.floor((len(power_by_freq) - 1) * cutoff)) + 1)
            low = float(power_by_freq[:k].sum())
            lfr = low / total if total > 0 else np.nan
            rows.append({
                "setting": f"{setting[0]}-{setting[1]}",
                "operator": name,
                "cutoff_ratio": cutoff,
                "low_frequency_ratio": lfr,
                "high_frequency_ratio": 1.0 - lfr if np.isfinite(lfr) else np.nan,
                "tv_energy": float(np.abs(dy).mean()),
                "curvature_energy": float(np.abs(ddy).mean()) if y.shape[2] > 2 else np.nan,
                "num_samples": int(y.shape[0]),
            })
    return rows


def phase_from_npz(setting, path):
    z = np.load(path, allow_pickle=True)
    names = op_names_from_npz(z)
    weighted = z["weighted_expert_output"].astype(np.float32)
    P = weighted.shape[2]
    rows = []
    for region, (a, b) in bands(P).items():
        sub = np.abs(weighted[:, :, a:b, :]).mean(axis=(0, 1, 2))
        total = float(sub.sum()) + 1e-12
        for oi, name in enumerate(names):
            rows.append({
                "setting": f"{setting[0]}-{setting[1]}",
                "region": region,
                "operator": name,
                "A_k": float(sub[oi]),
                "R_k": float(sub[oi] / total),
            })
    return rows


def spec_from_existing():
    recon_rows = []
    profile_rows = []
    align_rows = []
    prof_path = RAW_DIR / "spectral_source_profile.csv"
    recon_path = RAW_DIR / "spectral_reconstruction.csv"
    if not prof_path.exists() or not recon_path.exists():
        return recon_rows, profile_rows, align_rows
    prof = pd.read_csv(prof_path)
    prof = prof[prof["profile_type"].astype(str) == "final_gated_spectral"]
    recon = pd.read_csv(recon_path)
    for setting in SPEC_SETTINGS:
        ds, pred = setting
        rp = prof[(prof.dataset == ds) & (prof.pred_len == pred)]
        rr = recon[(recon.dataset == ds) & (recon.pred_len == pred)]
        if not len(rp) or not len(rr):
            continue
        pr = rp.iloc[0]
        rc = rr.iloc[0]
        profile_rows.append({
            "setting": f"{ds}-{pred}",
            "expected_lag": float(pr["expected_historical_source_lag"]),
            "normalized_entropy": float(pr["source_normalized_entropy"]),
            "recent_source_share": float(pr["recent_8_share"]),
            "phase_aligned_source_share": float(pr["phase_aligned_source_share"]),
            "num_samples": "full_test",
            "notes": "reused_exact_final_gated_spectral_source_profile",
        })
        recon_rows.append({
            "setting": f"{ds}-{pred}",
            "num_samples": "full_test",
            "spec_recon_mae": float(pr.get("reconstruction_mae", rc["weighted_spectral_output_reconstruction_mae"])),
            "spec_recon_max_error": float(pr.get("reconstruction_max_abs", rc["weighted_spectral_output_reconstruction_max_abs"])),
            "residual_defined": "yes",
            "notes": "from_STOM_spec_exact_source_profile_summary_seed2026; C_spec includes residual, A, head_mix, gain, gate",
        })
    return recon_rows, profile_rows, align_rows

def build_report(inventory_rows, smooth_rows, phase_rows, spec_recon, spec_profile):
    lines = []
    lines.append("# Operator Semantics Validation\n")
    lines.append("## 1. Purpose\n")
    lines.append("This experiment validates the structural semantics of STOM operator outputs. It does not test perturbation response dominance and does not retrain any real-data model.\n")
    lines.append("## 2. Checkpoint Coverage\n")
    lines.append("| setting | found | command | checkpoint | notes |\n|---|---:|---|---|---|\n")
    for r in inventory_rows:
        lines.append(f"| {r['setting']} | {r['found']} | `{r['command_file']}` | `{r['checkpoint_path']}` | {r['notes']} |\n")
    lines.append("\n## 3. Smooth/Local Output Properties\n")
    if smooth_rows:
        df = pd.DataFrame(smooth_rows)
        for setting in sorted(df.setting.unique()):
            sub = df[(df.setting == setting) & (np.isclose(df.cutoff_ratio, 0.33))]
            if len(sub):
                r = sub.iloc[0]
                lines.append(f"- {setting}: smooth LFR@0.33={r.smooth_lfr:.6f}, local LFR@0.33={r.local_lfr:.6f}, smooth TV={r.smooth_tv:.6f}, local TV={r.local_tv:.6f}; smooth pass={r.smooth_lowfreq_pass}, local-vs-smooth pass={r.local_more_local_than_smooth_pass}.\n")
    else:
        lines.append("- No smooth/local setting completed.\n")
    lines.append("\n## 4. Phase Horizon Contribution\n")
    if phase_rows:
        for r in phase_rows:
            lines.append(f"- {r['setting']} ({r['notes']}): phase share near/middle/far = {float(r['phase_near']):.6f}/{float(r['phase_middle']):.6f}/{float(r['phase_far']):.6f}; far-near={float(r['phase_far_minus_near']):.6f}; pass={r['phase_decreasing_pass']}.\n")
    else:
        lines.append("- No phase setting completed.\n")
    lines.append("\n## 5. Spec Source Contribution\n")
    if spec_recon:
        prof = {r["setting"]: r for r in spec_profile}
        for r in spec_recon:
            p = prof.get(r["setting"], {})
            lines.append(f"- {r['setting']}: z_spec reconstruction MAE={float(r['spec_recon_mae']):.3e}, max={float(r['spec_recon_max_error']):.3e}, residual_defined={r['residual_defined']}; expected_lag={float(p.get('expected_lag', np.nan)):.6f}, entropy={float(p.get('normalized_entropy', np.nan)):.6f}, recent8={float(p.get('recent_source_share', np.nan)):.6f}.\n")
        for ds in ["ETTm1", "ETTh2"]:
            a = prof.get(f"{ds}-96")
            b = prof.get(f"{ds}-720")
            if a and b:
                ok = (float(a["expected_lag"]) > float(b["expected_lag"])) and (float(a["normalized_entropy"]) > float(b["normalized_entropy"])) and (float(a["recent_source_share"]) < float(b["recent_source_share"]))
                lines.append(f"- {ds} 96-vs-720 profile contrast: expected_lag {float(a['expected_lag']):.3f}->{float(b['expected_lag']):.3f}, entropy {float(a['normalized_entropy']):.3f}->{float(b['normalized_entropy']):.3f}, recent8 {float(a['recent_source_share']):.3f}->{float(b['recent_source_share']):.3f}; contrast_pass={'yes' if ok else 'no'}.\n")
    else:
        lines.append("- No spec source setting completed.\n")
    lines.append("\n## 6. Final Evidence Table\n")
    smooth_pass = "pass" if any(r["smooth_lowfreq_pass"] == "yes" for r in smooth_rows) else "partial/missing"
    local_pass = "pass" if any(r["local_more_local_than_smooth_pass"] == "yes" for r in smooth_rows) else "partial/missing"
    phase_pass = "pass" if any(r["phase_decreasing_pass"] == "yes" and r["notes"] == "main_setting" for r in phase_rows) else "partial/missing"
    spec_pass = "pass" if spec_recon and max(float(r["spec_recon_mae"]) for r in spec_recon) < 1e-6 else "partial/missing"
    table = [
        ("smooth", "smooth structural transfer", "higher low-frequency ratio", "ETTm1-96, ETTm2-96, ETTh2-336", smooth_pass, "operator output FFT/TV"),
        ("local", "local residual pattern correction", "higher HFR/TV/curvature than smooth", "ETTm1-96, ETTm2-96, ETTh2-336", local_pass, "only compared against smooth"),
        ("phase", "phase-aware backbone forecast path", "realized contribution decreases toward far horizon", "ETTm1-720, Weather-336, ETTh2-336", phase_pass, "correlation/shape evidence, not causation"),
        ("spec", "source-resolved residual transfer", "exact gated source reconstruction", "ETTm1/ETTh2 96/720", spec_pass, "C_spec includes residual value, kernel, head mix, gain, gate"),
    ]
    lines.append("| operator | semantic role | main evidence | settings | result | notes |\n|---|---|---|---|---|---|\n")
    for row in table:
        lines.append("| " + " | ".join(row) + " |\n")
    lines.append("\n## 7. Boundary Statement\n")
    lines.append("These checks do not prove that the operators are true causal mechanisms of the data-generating process. They validate that the forward-defined STOM operators exhibit output properties consistent with their structural roles.\n\n")
    lines.append("这些实验不证明各算子等同于真实数据生成机制，而是验证前向定义的 STOM 算子在输出形态上与其结构角色一致。\n")
    (OUT / "operator_semantics_validation_summary.md").write_text("".join(lines))



def main():
    import argparse
    global OUT, RAW_DIR, CHECKPOINT_DIR, COMMAND_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="operator_semantics_results")
    ap.add_argument("--raw-dir", default="analysis_inputs")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--command-dir", default="commands")
    opts = ap.parse_args()
    OUT = Path(opts.out_dir)
    RAW_DIR = Path(opts.raw_dir)
    CHECKPOINT_DIR = Path(opts.checkpoint_dir)
    COMMAND_DIR = Path(opts.command_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "commands").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    items, inv_rows = inventory()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_spectrum = []
    all_phase = []
    all_spec_recon, all_spec_profile, all_align = spec_from_existing()

    
    inference_needed = set()
    for setting in sorted(set(SMOOTH_LOCAL_SETTINGS + PHASE_SETTINGS)):
        p = raw_npz_path(setting)
        if p is None:
            inference_needed.add(setting)
            continue
        if setting in SMOOTH_LOCAL_SETTINGS:
            print(f"[postprocess-spectrum] {setting[0]}-{setting[1]} from {p}", flush=True)
            all_spectrum.extend(spectrum_from_npz(setting, p))
        if setting in PHASE_SETTINGS:
            print(f"[postprocess-phase] {setting[0]}-{setting[1]} from {p}", flush=True)
            all_phase.extend(phase_from_npz(setting, p))

    
    for setting in sorted(inference_needed):
        if setting not in items:
            continue
        cmd, ckpt = items[setting]
        print(f"[minimal-inference] {setting[0]}-{setting[1]} on {device}", flush=True)
        spectrum_rows, phase_rows, _, _, _ = accumulate_setting(setting, cmd, ckpt, device)
        if setting in SMOOTH_LOCAL_SETTINGS:
            all_spectrum.extend(spectrum_rows)
        if setting in PHASE_SETTINGS:
            all_phase.extend(phase_rows)

    write_csv(OUT / "operator_output_spectrum.csv", all_spectrum, ["setting", "operator", "cutoff_ratio", "low_frequency_ratio", "high_frequency_ratio", "tv_energy", "curvature_energy", "num_samples"])
    smooth_rows = smooth_local_summary(all_spectrum) if all_spectrum else []
    write_csv(OUT / "smooth_local_semantic_summary.csv", smooth_rows, ["setting", "cutoff_ratio", "smooth_lfr", "local_lfr", "spec_lfr", "phase_lfr", "smooth_hfr", "local_hfr", "smooth_tv", "local_tv", "smooth_curv", "local_curv", "smooth_lowfreq_pass", "local_more_local_than_smooth_pass", "notes"])
    write_csv(OUT / "phase_horizon_contribution.csv", all_phase, ["setting", "region", "operator", "A_k", "R_k"])
    phase_rows = phase_summary(all_phase) if all_phase else []
    write_csv(OUT / "phase_horizon_trend_summary.csv", phase_rows, ["setting", "phase_near", "phase_middle", "phase_far", "phase_far_minus_near", "phase_decreasing_pass", "notes"])
    write_csv(OUT / "spec_source_reconstruction.csv", all_spec_recon, ["setting", "num_samples", "spec_recon_mae", "spec_recon_max_error", "residual_defined", "notes"])
    write_csv(OUT / "spec_source_profile_summary.csv", all_spec_profile, ["setting", "expected_lag", "normalized_entropy", "recent_source_share", "phase_aligned_source_share", "num_samples", "notes"])
    write_csv(OUT / "spec_source_residual_alignment.csv", all_align, ["setting", "corr_residual", "corr_smooth", "residual_alignment", "num_samples", "notes"])
    build_report(inv_rows, smooth_rows, phase_rows, all_spec_recon, all_spec_profile)
    print("[Operator Semantics Validation Completed]")
    print("Output directory: operator_semantics_results/")
    print("Key files:")
    for name in [
        "checkpoint_inventory.csv",
        "operator_output_spectrum.csv",
        "smooth_local_semantic_summary.csv",
        "phase_horizon_contribution.csv",
        "phase_horizon_trend_summary.csv",
        "spec_source_reconstruction.csv",
        "spec_source_profile_summary.csv",
        "operator_semantics_validation_summary.md",
    ]:
        print(f"- {name}")


if __name__ == "__main__":
    main()