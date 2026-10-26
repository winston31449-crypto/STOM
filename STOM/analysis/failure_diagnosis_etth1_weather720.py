import csv
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

STOM_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = Path("failure_diagnosis_results")
CACHE_DIR = OUT_DIR / "cache"
FIG_DIR = OUT_DIR / "figures"
for _p in (OUT_DIR, CACHE_DIR, FIG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(STOM_DIR))
sys.path.insert(0, str(STOM_DIR / "scripts_local"))

from data_provider.data_factory import data_provider
from dump_stom_interpretability_raw import parse_command_file, unwrap_state_dict
from models import stom as stom_model

OPS = ["phase", "spectral", "smooth", "local"]
SEGMENTS = {"near": (0, 240), "middle": (240, 480), "far": (480, 720)}


ARTIFACTS = {}




def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def setting_from_args(args):
    return "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
        args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len,
        args.pred_len, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
        args.factor, args.embed, args.distil, args.des, 0
    )


def prepare_args(command_file):
    args = parse_command_file(Path(command_file))
    if args.root_path and not os.path.isabs(args.root_path):
        args.root_path = str((STOM_DIR / args.root_path).resolve())
    args.num_workers = 0
    args.batch_size_test = 0
    args.compile = 0
    args.use_gpu = torch.cuda.is_available()
    if not getattr(args, "number_of_targets", 0):
        args.number_of_targets = args.enc_in
    return args


def checkpoint_from_cmd(command_file):
    args = parse_command_file(Path(command_file))
    setting = setting_from_args(args)
    return STOM_DIR / "checkpoints" / setting / "checkpoint.pth"


def truth_from_cmd(command_file):
    args = prepare_args(command_file)
    _, loader = data_provider(args, "test")
    ys = []
    for data in loader:
        batch_y = data[1].float().numpy()
        f_dim = -args.number_of_targets
        if args.features == "MS":
            f_dim = -1
        ys.append(batch_y[:, -args.pred_len:, f_dim:])
    return np.concatenate(ys, axis=0).reshape(-1, args.pred_len, args.number_of_targets)


def load_model(command_file, device):
    args = prepare_args(command_file)
    args.use_gpu = device.type == "cuda"
    args.gpu = 0
    model = stom_model.Model(args).to(device)
    ckpt = checkpoint_from_cmd(command_file)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(unwrap_state_dict(state), strict=False)
    model.eval()
    return args, model, ckpt


def forward_components(model, batch_x, batch_x_mark, batch_y_mark):
    m = model
    x0 = batch_x.permute(0, 2, 1)
    mean_all = x0[:, :, [-1]]
    std_all = torch.ones(x0.size(0), x0.size(1), 1, device=x0.device)
    x_norm = (x0 - mean_all) / std_all
    x_hist_norm_targets = x_norm[:, -m.number_of_targets:, :].clone()
    target_mean = mean_all[:, -m.number_of_targets:, :]

    x = torch.cat([x_norm, m.out_fill(x_norm)], dim=2)
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
    smooth = stom.moving_average(x_hist_norm_targets)
    residual = x_hist_norm_targets - smooth
    y_smooth = stom.trend_proj(smooth)
    y_local = stom.local_proj(F.gelu(stom.local_conv(residual)))
    a, *_ = stom._spectral_kernel(x_hist_norm_targets.device, x_hist_norm_targets.dtype)
    y_spec_h = torch.einsum("hpl,bcl->bhcp", a, residual)
    spec_head_mix = torch.softmax(stom.spec_head_logits, dim=0).to(device=x_hist_norm_targets.device, dtype=x_hist_norm_targets.dtype)
    spec_gain = F.softplus(stom.raw_spec_gain).to(device=x_hist_norm_targets.device, dtype=x_hist_norm_targets.dtype)
    y_spec = spec_gain * torch.einsum("h,bhcp->bcp", spec_head_mix, y_spec_h)

    hist_mean = x_hist_norm_targets.mean(dim=-1)
    hist_std = x_hist_norm_targets.std(dim=-1, unbiased=False)
    last_value = x_hist_norm_targets[:, :, -1]
    smooth_slope = smooth[:, :, -1] - smooth[:, :, 0]
    residual_energy = (residual.pow(2).mean(dim=-1) + 1e-6).sqrt()
    y_phase_norm = (y_phase.pow(2).mean(dim=-1) + 1e-6).sqrt()
    y_spec_norm = (y_spec.pow(2).mean(dim=-1) + 1e-6).sqrt()
    y_smooth_norm = (y_smooth.pow(2).mean(dim=-1) + 1e-6).sqrt()
    y_local_norm = (y_local.pow(2).mean(dim=-1) + 1e-6).sqrt()
    gate_base_feat = torch.stack([hist_mean, hist_std, last_value, smooth_slope, residual_energy, y_phase_norm, y_spec_norm, y_smooth_norm, y_local_norm], dim=-1)
    horizon = torch.arange(m.pred_len, device=x_hist_norm_targets.device, dtype=gate_base_feat.dtype) / max(m.pred_len, 1)
    horizon = horizon.view(1, 1, m.pred_len, 1).expand(x_hist_norm_targets.size(0), x_hist_norm_targets.size(1), -1, -1)
    if getattr(stom, "ablation_mode", "full") == "no_horizon_gate":
        horizon = torch.zeros_like(horizon)
    gate_feat = torch.cat([gate_base_feat.unsqueeze(2).expand(-1, -1, m.pred_len, -1), horizon], dim=-1)
    gate_logits = stom.gate_mlp(gate_feat.float()).to(dtype=x_hist_norm_targets.dtype)
    expert_mask = stom._expert_mask(x_hist_norm_targets.device, x_hist_norm_targets.dtype).view(1, 1, 1, 4)
    gate = torch.softmax(gate_logits.masked_fill(expert_mask <= 0, -1e9), dim=-1)
    expert = torch.stack([y_phase, y_spec, y_smooth, y_local], dim=-1)
    weighted = expert * gate
    y_norm = weighted.sum(dim=-1)
    y_final = (y_norm + target_mean).permute(0, 2, 1)
    return {"gate": gate, "expert": expert, "weighted": weighted, "y_norm": y_norm, "target_mean": target_mean, "y_final": y_final}


def generate_stom_pred(setting_key, device):
    art = ARTIFACTS[setting_key]
    pred_path = Path(art["stom_pred"])
    if pred_path.exists():
        return
    args, model, _ = load_model(art["stom_cmd"], device)
    _, loader = data_provider(args, "test")
    preds = []
    with torch.no_grad():
        for data in loader:
            batch_x = data[0].float().to(device)
            batch_x_mark = data[2].float().to(device)
            batch_y_mark = data[3].float().to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.use_amp and device.type == "cuda")):
                comp = forward_components(model, batch_x, batch_x_mark, batch_y_mark)
            preds.append(comp["y_final"].detach().float().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    np.save(pred_path, pred)


def compute_metrics(pred, true):
    d = pred - true
    return float(np.mean(d * d)), float(np.mean(np.abs(d)))


def rolling(x, k=31):
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def save_position_plot(path, rows, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.array([r["horizon"] for r in rows])
    delta = np.array([r["delta_mse"] for r in rows])
    lin = np.array([r["linattn_mse"] for r in rows])
    stom = np.array([r["stom_mse"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(h, lin, label="LinAttn", lw=1.2)
    axes[0].plot(h, stom, label="STOM", lw=1.2)
    axes[0].set_ylabel("MSE")
    axes[0].legend(frameon=False)
    axes[1].plot(h, delta, color="#9b2226", alpha=0.45, lw=0.8)
    axes[1].plot(h, rolling(delta), color="#9b2226", lw=1.8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Delta MSE")
    axes[2].plot(h, np.cumsum(stom) / h, label="STOM", lw=1.2)
    axes[2].plot(h, np.cumsum(lin) / h, label="LinAttn", lw=1.2)
    axes[2].set_ylabel("Cumulative MSE")
    axes[2].set_xlabel("Forecast position")
    axes[2].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_channel_plot(path, rows, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows_s = sorted(rows, key=lambda r: r["delta_mse"], reverse=True)
    labels = [str(r["channel"]) for r in rows_s]
    vals = [r["delta_mse"] for r in rows_s]
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(rows)), 4))
    ax.bar(labels, vals, color=["#9b2226" if v > 0 else "#2a9d8f" for v in vals])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("STOM - LinAttn MSE")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def localization(setting_key, true, lin, stom):
    art = ARTIFACTS[setting_key]
    dataset = art["dataset"]
    P = art["pred_len"]
    se_l = (lin - true) ** 2
    se_s = (stom - true) ** 2
    ae_l = np.abs(lin - true)
    ae_s = np.abs(stom - true)

    pos_rows = []
    for p in range(P):
        lm = float(se_l[:, p, :].mean())
        sm = float(se_s[:, p, :].mean())
        la = float(ae_l[:, p, :].mean())
        sa = float(ae_s[:, p, :].mean())
        pos_rows.append({
            "dataset": dataset, "pred_len": P, "horizon": p + 1,
            "linattn_mse": lm, "stom_mse": sm, "delta_mse": sm - lm,
            "relative_delta_mse": (sm - lm) / max(lm, 1e-12),
            "linattn_mae": la, "stom_mae": sa, "delta_mae": sa - la,
            "rolling_delta_mse_31": float(rolling(np.array([float(se_s[:, q, :].mean() - se_l[:, q, :].mean()) for q in range(P)]))[p]),
            "cumulative_linattn_mse": float(se_l[:, : p + 1, :].mean()),
            "cumulative_stom_mse": float(se_s[:, : p + 1, :].mean()),
        })
    write_csv(OUT_DIR / f"positionwise_error_{dataset}_720.csv", pos_rows, list(pos_rows[0].keys()))
    save_position_plot(FIG_DIR / f"positionwise_error_{dataset}_720.png", pos_rows, f"{dataset}-720 position-wise failure")

    channel_rows = []
    total_delta_sum = float((se_s - se_l).mean(axis=(0, 1)).sum())
    for c in range(true.shape[-1]):
        row = {
            "dataset": dataset, "pred_len": P, "channel": c,
            "linattn_mse": float(se_l[:, :, c].mean()),
            "stom_mse": float(se_s[:, :, c].mean()),
            "delta_mse": float(se_s[:, :, c].mean() - se_l[:, :, c].mean()),
            "relative_delta_mse": float((se_s[:, :, c].mean() - se_l[:, :, c].mean()) / max(se_l[:, :, c].mean(), 1e-12)),
            "linattn_mae": float(ae_l[:, :, c].mean()),
            "stom_mae": float(ae_s[:, :, c].mean()),
            "contribution_to_total_delta": float((se_s[:, :, c].mean() - se_l[:, :, c].mean()) / total_delta_sum) if abs(total_delta_sum) > 1e-12 else 0.0,
        }
        for seg, (a, b) in SEGMENTS.items():
            row[f"{seg}_linattn_mse"] = float(se_l[:, a:b, c].mean())
            row[f"{seg}_stom_mse"] = float(se_s[:, a:b, c].mean())
            row[f"{seg}_delta_mse"] = float(se_s[:, a:b, c].mean() - se_l[:, a:b, c].mean())
        channel_rows.append(row)
    write_csv(OUT_DIR / f"channelwise_error_{dataset}_720.csv", channel_rows, list(channel_rows[0].keys()))
    save_channel_plot(FIG_DIR / f"channelwise_error_{dataset}_720.png", channel_rows, f"{dataset}-720 channel-wise degradation")

    sample_rows = []
    for i in range(true.shape[0]):
        per_c = (se_s[i] - se_l[i]).mean(axis=0)
        row = {
            "dataset": dataset, "pred_len": P, "sample_index": i,
            "linattn_mse": float(se_l[i].mean()), "stom_mse": float(se_s[i].mean()),
            "delta_mse": float(se_s[i].mean() - se_l[i].mean()),
            "linattn_mae": float(ae_l[i].mean()), "stom_mae": float(ae_s[i].mean()),
            "worst_channel": int(np.argmax(per_c)),
            "worst_channel_delta_mse": float(np.max(per_c)),
            "max_abs_forecast_error_stom": float(np.max(np.abs(stom[i] - true[i]))),
            "max_abs_forecast_error_linattn": float(np.max(np.abs(lin[i] - true[i]))),
        }
        for seg, (a, b) in SEGMENTS.items():
            row[f"{seg}_delta_mse"] = float(se_s[i, a:b, :].mean() - se_l[i, a:b, :].mean())
        sample_rows.append(row)
    write_csv(OUT_DIR / f"samplewise_delta_{dataset}_720.csv", sample_rows, list(sample_rows[0].keys()))

    seg_rows = []
    for seg, (a, b) in SEGMENTS.items():
        seg_rows.append({
            "dataset": dataset, "pred_len": P, "segment": seg, "start_horizon": a + 1, "end_horizon": b,
            "linattn_mse": float(se_l[:, a:b, :].mean()),
            "stom_mse": float(se_s[:, a:b, :].mean()),
            "delta_mse": float(se_s[:, a:b, :].mean() - se_l[:, a:b, :].mean()),
            "linattn_mae": float(ae_l[:, a:b, :].mean()),
            "stom_mae": float(ae_s[:, a:b, :].mean()),
            "delta_mae": float(ae_s[:, a:b, :].mean() - ae_l[:, a:b, :].mean()),
        })
    write_csv(OUT_DIR / f"segment_error_{dataset}_720.csv", seg_rows, list(seg_rows[0].keys()))
    return pos_rows, channel_rows, sample_rows, seg_rows


class StreamStats:
    def __init__(self, pred_len, channels):
        self.P = pred_len
        self.C = channels
        self.n_sc = 0
        self.gate_sum = np.zeros((pred_len, 4), dtype=np.float64)
        self.expert_abs_sum = np.zeros((pred_len, 4), dtype=np.float64)
        self.weight_abs_sum = np.zeros((pred_len, 4), dtype=np.float64)
        self.weight_signed_sum = np.zeros((pred_len, 4), dtype=np.float64)
        self.interventions = OrderedDict()

    def update_components(self, comp):
        gate = comp["gate"].detach().float()
        expert = comp["expert"].detach().float()
        weighted = comp["weighted"].detach().float()
        B, C, P, _ = gate.shape
        self.n_sc += B * C
        self.gate_sum += gate.sum(dim=(0, 1)).cpu().numpy()
        self.expert_abs_sum += expert.abs().sum(dim=(0, 1)).cpu().numpy()
        self.weight_abs_sum += weighted.abs().sum(dim=(0, 1)).cpu().numpy()
        self.weight_signed_sum += weighted.sum(dim=(0, 1)).cpu().numpy()

    def add_intervention(self, name, pred, true):
        se = (pred - true) ** 2
        ae = np.abs(pred - true)
        if name not in self.interventions:
            self.interventions[name] = {"se_sum": 0.0, "ae_sum": 0.0, "n": 0, "pos_se": np.zeros(self.P), "pos_n": 0, "chan_se": np.zeros(self.C), "chan_n": 0}
        d = self.interventions[name]
        d["se_sum"] += float(se.sum())
        d["ae_sum"] += float(ae.sum())
        d["n"] += se.size
        d["pos_se"] += se.mean(axis=2).sum(axis=0)
        d["pos_n"] += se.shape[0]
        d["chan_se"] += se.mean(axis=1).sum(axis=0)
        d["chan_n"] += se.shape[0]

    def alignment_rows(self, dataset):
        gate = self.gate_sum / max(self.n_sc, 1)
        expert_abs = self.expert_abs_sum / max(self.n_sc, 1)
        weighted_abs = self.weight_abs_sum / max(self.n_sc, 1)
        signed = self.weight_signed_sum / max(self.n_sc, 1)
        share = weighted_abs / np.maximum(weighted_abs.sum(axis=1, keepdims=True), 1e-12)
        rows = []
        for p in range(self.P):
            for k, op in enumerate(OPS):
                rows.append({
                    "dataset": dataset, "pred_len": self.P, "horizon": p + 1, "operator": op,
                    "mean_gate": float(gate[p, k]),
                    "mean_abs_operator_output": float(expert_abs[p, k]),
                    "mean_abs_realized_contribution": float(weighted_abs[p, k]),
                    "mean_signed_realized_contribution": float(signed[p, k]),
                    "contribution_share": float(share[p, k]),
                })
        return rows

    def intervention_rows(self, dataset):
        rows = []
        for name, d in self.interventions.items():
            pos_mse = d["pos_se"] / max(d["pos_n"], 1)
            chan_mse = d["chan_se"] / max(d["chan_n"], 1)
            row = {
                "dataset": dataset, "pred_len": self.P, "intervention": name,
                "mse": d["se_sum"] / max(d["n"], 1), "mae": d["ae_sum"] / max(d["n"], 1),
            }
            for seg, (a, b) in SEGMENTS.items():
                row[f"{seg}_mse"] = float(pos_mse[a:b].mean())
            row["worst_channel"] = int(np.argmax(chan_mse))
            row["worst_channel_mse"] = float(np.max(chan_mse))
            rows.append(row)
        return rows


def alt_norm(comp, mode):
    weighted = comp["weighted"]
    expert = comp["expert"]
    gate = comp["gate"]
    if mode == "full":
        return weighted.sum(dim=-1)
    if mode.startswith("czero_"):
        mask = torch.ones(4, device=weighted.device, dtype=weighted.dtype)
        if mode == "czero_phase_local":
            mask[0] = 0
            mask[3] = 0
        else:
            op = mode.split("_", 1)[1]
            mask[OPS.index(op)] = 0
        return (weighted * mask.view(1, 1, 1, 4)).sum(dim=-1)
    if mode.startswith("grenorm_"):
        mask = torch.ones(4, device=weighted.device, dtype=weighted.dtype)
        op = mode.split("_", 1)[1]
        if op == "phase_local":
            mask[0] = 0
            mask[3] = 0
        elif op == "smooth_only":
            mask[:] = 0
            mask[2] = 1
        elif op == "phase_only":
            mask[:] = 0
            mask[0] = 1
        else:
            mask[OPS.index(op)] = 0
        g = gate * mask.view(1, 1, 1, 4)
        g = g / g.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return (expert * g).sum(dim=-1)
    if mode == "local_linear_decay":
        p = torch.linspace(1.0, 0.0, weighted.shape[2], device=weighted.device, dtype=weighted.dtype)
        w = weighted.clone()
        w[..., 3] = w[..., 3] * p.view(1, 1, -1)
        return w.sum(dim=-1)
    if mode == "local_far_zero":
        w = weighted.clone()
        w[:, :, 480:, 3] = 0
        return w.sum(dim=-1)
    raise ValueError(mode)


def component_and_intervention(setting_key, true, device):
    art = ARTIFACTS[setting_key]
    args, model, _ = load_model(art["stom_cmd"], device)
    _, loader = data_provider(args, "test")
    stats = StreamStats(args.pred_len, args.number_of_targets)
    modes = [
        "full",
        "czero_phase", "czero_spectral", "czero_smooth", "czero_local", "czero_phase_local",
        "grenorm_phase", "grenorm_spectral", "grenorm_smooth", "grenorm_local", "grenorm_phase_local",
        "grenorm_smooth_only", "grenorm_phase_only",
        "local_linear_decay", "local_far_zero",
    ]
    offset = 0
    with torch.no_grad():
        for data in loader:
            batch_x = data[0].float().to(device)
            batch_y = data[1].float().to(device)
            batch_x_mark = data[2].float().to(device)
            batch_y_mark = data[3].float().to(device)
            f_dim = -args.number_of_targets
            if args.features == "MS":
                f_dim = -1
            true_b = batch_y[:, -args.pred_len:, f_dim:].float().cpu().numpy()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.use_amp and device.type == "cuda")):
                comp = forward_components(model, batch_x, batch_x_mark, batch_y_mark)
            stats.update_components(comp)
            mean = comp["target_mean"]
            for mode in modes:
                yn = alt_norm(comp, mode)
                pred = (yn + mean).permute(0, 2, 1).detach().float().cpu().numpy()
                stats.add_intervention(mode, pred, true_b)
            offset += true_b.shape[0]
    align_rows = stats.alignment_rows(art["dataset"])
    int_rows = stats.intervention_rows(art["dataset"])
    write_csv(OUT_DIR / f"ERROR_OPERATOR_ALIGNMENT_{art['dataset']}_720.csv", align_rows, list(align_rows[0].keys()))
    write_csv(OUT_DIR / f"operator_intervention_{art['dataset']}_720.csv", int_rows, list(int_rows[0].keys()))
    return align_rows, int_rows


def summarize_localization(dataset, pos_rows, channel_rows, sample_rows, seg_rows):
    delta = np.array([r["delta_mse"] for r in pos_rows])
    longest = 0
    cur = 0
    for v in delta > 0:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    degraded_channels = sum(1 for r in channel_rows if r["delta_mse"] > 0)
    sample_delta = np.array([r["delta_mse"] for r in sample_rows])
    improved = float((sample_delta < 0).mean())
    worsened = float((sample_delta > 0).mean())
    q = np.quantile(sample_delta, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    return {
        "dataset": dataset,
        "overall_delta_mean": float(sample_delta.mean()),
        "sample_delta_median": float(np.median(sample_delta)),
        "sample_delta_std": float(sample_delta.std()),
        "q05": float(q[0]), "q10": float(q[1]), "q25": float(q[2]), "q50": float(q[3]), "q75": float(q[4]), "q90": float(q[5]), "q95": float(q[6]),
        "improved_sample_ratio": improved,
        "worsened_sample_ratio": worsened,
        "longest_positive_delta_run_positions": int(longest),
        "max_degradation_horizon": int(np.argmax(delta) + 1),
        "max_degradation_delta_mse": float(delta.max()),
        "max_improvement_horizon": int(np.argmin(delta) + 1),
        "max_improvement_delta_mse": float(delta.min()),
        "degraded_channels": int(degraded_channels),
        "num_channels": len(channel_rows),
        "worst_channel": int(max(channel_rows, key=lambda r: r["delta_mse"])["channel"]),
        "worst_channel_delta_mse": float(max(channel_rows, key=lambda r: r["delta_mse"])["delta_mse"]),
        "segment_deltas": {r["segment"]: r["delta_mse"] for r in seg_rows},
    }


def write_reports(align_audit, loc_summaries, operator_notes, decisions):
    md = ["# Artifact Alignment Audit\n\n"]
    md.append("No training was run. ETTh1-720 formal STOM lacked `pred.npy`, so this study generated an inference-only cached prediction from the official best-validation checkpoint.\n\n")
    md.append("| setting | model | pred path | shape | recomputed MSE | recomputed MAE | expected MSE | expected MAE | pass |\n")
    md.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |\n")
    for r in align_audit:
        md.append(f"| {r['setting']} | {r['model']} | `{r['pred_path']}` | {r['shape']} | {r['mse']:.9f} | {r['mae']:.9f} | {r['expected_mse']:.9f} | {r['expected_mae']:.9f} | {r['pass']} |\n")
    (OUT_DIR / "ARTIFACT_ALIGNMENT_AUDIT.md").write_text("".join(md))

    md = ["# Error Localization Report\n\n"]
    for s in loc_summaries:
        md.append(f"## {s['dataset']}-720\n\n")
        md.append(f"- Mean sample delta MSE: `{s['overall_delta_mean']:.6f}`; median `{s['sample_delta_median']:.6f}`; std `{s['sample_delta_std']:.6f}`.\n")
        md.append(f"- Improved / worsened sample ratio: `{s['improved_sample_ratio']:.3f}` / `{s['worsened_sample_ratio']:.3f}`.\n")
        md.append(f"- Segment delta MSE: near `{s['segment_deltas']['near']:.6f}`, middle `{s['segment_deltas']['middle']:.6f}`, far `{s['segment_deltas']['far']:.6f}`.\n")
        md.append(f"- Longest positive position-delta run: `{s['longest_positive_delta_run_positions']}` positions.\n")
        md.append(f"- Worst channel: `{s['worst_channel']}` with delta MSE `{s['worst_channel_delta_mse']:.6f}`; degraded channels `{s['degraded_channels']}/{s['num_channels']}`.\n")
        if s["worsened_sample_ratio"] > 0.65 and s["degraded_channels"] > s["num_channels"] * 0.65:
            cls = "global systematic failure"
        elif s["segment_deltas"]["far"] > max(s["segment_deltas"]["near"], s["segment_deltas"]["middle"]) * 1.5:
            cls = "far-horizon extrapolation failure"
        elif s["worsened_sample_ratio"] < 0.5 and s["q95"] > abs(s["q50"]) * 5:
            cls = "heavy-tail / rare-sample failure"
        elif abs(s["overall_delta_mean"]) < 0.002:
            cls = "statistically small tie-like behavior"
        else:
            cls = "mixed mechanism"
        md.append(f"- Initial classification: **{cls}**.\n\n")
    (OUT_DIR / "ERROR_LOCALIZATION_REPORT.md").write_text("".join(md))

    md = ["# Error Operator Alignment Report\n\n"]
    for note in operator_notes:
        md.append(f"## {note['dataset']}-720\n\n")
        for k, v in note.items():
            if k != "dataset":
                md.append(f"- {k}: {v}\n")
        md.append("\n")
    (OUT_DIR / "ERROR_OPERATOR_ALIGNMENT_REPORT.md").write_text("".join(md))

    md = ["# Causal Failure Diagnosis\n\n"]
    for d in decisions:
        md.append(f"## {d['setting']}\n\n")
        for k, v in d.items():
            if k != "setting":
                md.append(f"- {k}: {v}\n")
        md.append("\n")
    (OUT_DIR / "CAUSAL_FAILURE_DIAGNOSIS.md").write_text("".join(md))


def make_operator_note(setting_key, pos_rows, align_rows, int_rows):
    dataset = ARTIFACTS[setting_key]["dataset"]
    delta = np.array([r["delta_mse"] for r in pos_rows])
    by_op = {op: [] for op in OPS}
    by_gate = {op: [] for op in OPS}
    by_mag = {op: [] for op in OPS}
    for r in align_rows:
        op = r["operator"]
        by_op[op].append(float(r["contribution_share"]))
        by_gate[op].append(float(r["mean_gate"]))
        by_mag[op].append(float(r["mean_abs_operator_output"]))
    corr = {}
    for op in OPS:
        for label, arr in [("share", by_op[op]), ("gate", by_gate[op]), ("magnitude", by_mag[op])]:
            arr = np.asarray(arr)
            if arr.std() < 1e-12 or delta.std() < 1e-12:
                c = 0.0
            else:
                c = float(np.corrcoef(delta, arr)[0, 1])
            corr[f"corr_delta_mse_vs_{op}_{label}"] = f"{c:.4f}"
    full = next(r for r in int_rows if r["intervention"] == "full")
    candidates = sorted([r for r in int_rows if r["intervention"] != "full"], key=lambda r: r["mse"])
    best = candidates[0]
    return {
        "dataset": dataset,
        **corr,
        "best_intervention": f"{best['intervention']} mse={best['mse']:.6f} mae={best['mae']:.6f}",
        "full_intervention_check": f"mse={full['mse']:.6f} mae={full['mae']:.6f}",
        "mechanism_screen": "Correlation is associative only; causal evidence is taken from operator interventions.",
    }


def make_decision(setting_key, loc_summary, int_rows):
    dataset = ARTIFACTS[setting_key]["dataset"]
    full = next(r for r in int_rows if r["intervention"] == "full")
    sorted_rows = sorted([r for r in int_rows if r["intervention"] != "full"], key=lambda r: r["mse"])
    best = sorted_rows[0]
    improvement = full["mse"] - best["mse"]
    if abs(float(ARTIFACTS[setting_key]["expected"]["stom_mse"]) - float(ARTIFACTS[setting_key]["expected"]["lin_mse"])) < 0.003:
        failure_type = "tie-like / negligible degradation"
        paper_decision = "not_a_meaningful_failure"
    elif improvement > 0.01:
        failure_type = "operator contribution failure with actionable masking evidence"
        paper_decision = "actionable_diagnosis_case_study"
    else:
        failure_type = "decomposition-visible but not causally recoverable by simple masking"
        paper_decision = "explainable_limitation"
    suspect = "local" if "local" in best["intervention"] else ("smooth" if "smooth" in best["intervention"] else ("phase" if "phase" in best["intervention"] else "spectral"))
    return {
        "setting": f"{dataset}-720",
        "failure_type": failure_type,
        "error_region": max(loc_summary["segment_deltas"], key=lambda k: loc_summary["segment_deltas"][k]),
        "primary_suspect_operator": suspect,
        "masking_effect": f"best {best['intervention']} improves over full by MSE {improvement:.6f}",
        "confidence": "medium" if improvement > 0.01 else "low-to-medium",
        "paper_decision": paper_decision,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out-dir", default="failure_diagnosis_results")
    args_cli = ap.parse_args()
    global OUT_DIR, CACHE_DIR, FIG_DIR
    OUT_DIR = Path(args_cli.out_dir)
    CACHE_DIR = OUT_DIR / "cache"
    FIG_DIR = OUT_DIR / "figures"
    for path in (OUT_DIR, CACHE_DIR, FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.update({key: {field: Path(value) if field in ("lin_cmd", "stom_cmd", "lin_pred", "stom_pred", "stom_diag") else value
                            for field, value in item.items()}
                      for key, item in json.loads(Path(args_cli.artifacts).read_text()).items()})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"stage": "start", "device": str(device)}, ensure_ascii=False))
    align_audit, loc_summaries, operator_notes, decisions = [], [], [], []
    for key, art in ARTIFACTS.items():
        print(json.dumps({"stage": "setting_start", "setting": key}, ensure_ascii=False))
        true = truth_from_cmd(art["lin_cmd"])
        generate_stom_pred(key, device)
        lin = np.load(art["lin_pred"])
        stom = np.load(art["stom_pred"])
        for model, pred, path in [("LinAttn", lin, art["lin_pred"]), ("STOM", stom, art["stom_pred"])]:
            mse, mae = compute_metrics(pred, true)
            exp_mse = art["expected"]["lin_mse"] if model == "LinAttn" else art["expected"]["stom_mse"]
            exp_mae = art["expected"]["lin_mae"] if model == "LinAttn" else art["expected"]["stom_mae"]
            ok = abs(mse - exp_mse) < 2e-3 and abs(mae - exp_mae) < 2e-3
            align_audit.append({"setting": f"{art['dataset']}-720", "model": model, "pred_path": str(path), "shape": list(pred.shape), "mse": mse, "mae": mae, "expected_mse": exp_mse, "expected_mae": exp_mae, "pass": ok})
            if not ok:
                raise RuntimeError(f"metric alignment failed for {key} {model}: got {mse}/{mae}, expected {exp_mse}/{exp_mae}")
        pos_rows, ch_rows, smp_rows, seg_rows = localization(key, true, lin, stom)
        loc_summary = summarize_localization(art["dataset"], pos_rows, ch_rows, smp_rows, seg_rows)
        loc_summaries.append(loc_summary)
        align_rows, int_rows = component_and_intervention(key, true, device)
        operator_notes.append(make_operator_note(key, pos_rows, align_rows, int_rows))
        decisions.append(make_decision(key, loc_summary, int_rows))
        print(json.dumps({"stage": "setting_done", "setting": key, "delta_mse": loc_summary["overall_delta_mean"], "decision": decisions[-1]}, ensure_ascii=False))
    write_reports(align_audit, loc_summaries, operator_notes, decisions)
    write_csv(OUT_DIR / "FAILURE_STUDY_DECISION.csv", decisions, ["setting", "failure_type", "error_region", "primary_suspect_operator", "masking_effect", "confidence", "paper_decision"])
    
    md = ["# ETTh1 / Weather 720 Failure Study\n\n"]
    md.append("## Motivation\n\nThis study follows an interpretability-guided sequence: artifact alignment, error localization, operator-error alignment, and inference-time operator intervention before any code-level refinement.\n\n")
    for d in decisions:
        md.append(f"## {d['setting']}\n\n")
        md.append(f"- Failure type: {d['failure_type']}\n")
        md.append(f"- Error region: {d['error_region']}\n")
        md.append(f"- Primary suspect operator: {d['primary_suspect_operator']}\n")
        md.append(f"- Masking effect: {d['masking_effect']}\n")
        md.append(f"- Paper decision: `{d['paper_decision']}`\n\n")
    md.append("## Evidence Paths\n\n")
    for name in ["ARTIFACT_ALIGNMENT_AUDIT.md", "ERROR_LOCALIZATION_REPORT.md", "ERROR_OPERATOR_ALIGNMENT_REPORT.md", "CAUSAL_FAILURE_DIAGNOSIS.md", "FAILURE_STUDY_DECISION.csv"]:
        md.append(f"- `{OUT_DIR / name}`\n")
    md.append("\n## Code Correction Decision\n\nNo permanent model-code correction is executed by this script. A correction should only be attempted if ETTh1-720 intervention results show a clear, story-consistent recoverable operator behavior and protection settings are then run separately.\n")
    (OUT_DIR / "ETTh1_Weather_720_FAILURE_STUDY.md").write_text("".join(md))
    print(json.dumps({"stage": "complete", "out_dir": str(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()