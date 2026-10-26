import argparse
import csv
import itertools
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


STOM = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(STOM), str(Path(__file__).resolve().parent)]

from data_provider.data_factory import data_provider
from models import stom as stom_model
from dump_stom_interpretability_raw import parse_command_file, unwrap_state_dict

SEED = 2026
EPS = 1e-12
SETTINGS = {}



def sh(cmd, cwd=None, timeout=20):
    try:
        return subprocess.check_output(cmd, shell=True, cwd=cwd, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
    except Exception as exc:
        return f"N/A: {exc}"


def mkdirs(out):
    for name in ["efficiency", "source_suppression", "temperature", "correctability", "gate_no_magnitude", "bootstrap", "logs"]:
        (out / name).mkdir(parents=True, exist_ok=True)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({k for row in rows for k in row.keys()}) if rows else []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def append_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def env_info(command=""):
    cuda = torch.cuda.is_available()
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "cwd": str(Path.cwd()),
        "git_commit": sh("git rev-parse HEAD"),
        "git_status": sh("git status --short", timeout=10),
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "cpu",
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
    }


def init_out(out_arg):
    if out_arg:
        out = Path(out_arg).resolve()
    else:
        out = Path("final_diagnostics_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    mkdirs(out)
    return out


def setup_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def resolve_root(args):
    if args.root_path and not os.path.isabs(args.root_path):
        args.root_path = str((STOM / args.root_path).resolve())


def load_model(cmd, ckpt, device, use_stom=1, disable_mag=0, batch_size=None):
    args = parse_command_file(Path(cmd))
    resolve_root(args)
    args.use_gpu = device.type == "cuda"
    args.gpu = 0
    args.devices = "0"
    args.use_multi_gpu = False
    args.num_workers = 0
    args.batch_size_test = 0
    args.compile = 0
    args.use_stom = int(use_stom)
    args.disable_operator_magnitude_features = int(disable_mag)
    args.random_seed = SEED
    if batch_size is not None:
        args.batch_size = int(batch_size)
    model = stom_model.Model(args).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(unwrap_state_dict(state), strict=False)
    model.eval()
    return args, model


def model_batch(args, data, device):
    bx, by, bxm, bym = [x.float().to(device, non_blocking=True) for x in data[:4]]
    dec = torch.zeros_like(by[:, -args.pred_len:, :])
    dec = torch.cat([by[:, :args.label_len, :], dec], dim=1).to(device, non_blocking=True)
    return bx, by, bxm, bym, dec


def target_from_batch(args, by, model):
    f_dim = -args.number_of_targets
    if args.features == "MS":
        f_dim = -1
    return by[:, -args.pred_len:, f_dim:].permute(0, 2, 1).contiguous()


def denorm(model, y_norm, mean, std):
    return y_norm * std[:, -model.number_of_targets:] + mean[:, -model.number_of_targets:]


def extract_xhist_yphase(model, batch_x, batch_x_mark, batch_y_mark):
    m = model
    x0 = batch_x.permute(0, 2, 1)
    mean = x0[:, :, [-1]]
    std = torch.ones(x0.size(0), x0.size(1), 1, device=x0.device)
    x_norm = (x0 - mean) / std
    x_hist = x_norm[:, -m.number_of_targets:, :].clone()
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
    return x_hist, y_phase, mean, std


def gate_from_outputs(model, x_hist, y_phase, y_spec, y_smooth, y_local):
    stom = model.stom_head
    smooth = stom.moving_average(x_hist)
    residual = x_hist - smooth
    expert_outputs = {
        "phase": y_phase,
        "spectral": y_spec,
        "smooth": y_smooth,
        "local": y_local,
    }
    hist_mean = x_hist.mean(dim=-1)
    hist_std = x_hist.std(dim=-1, unbiased=False)
    last_value = x_hist[:, :, -1]
    smooth_slope = smooth[:, :, -1] - smooth[:, :, 0]
    residual_energy = (residual.pow(2).mean(dim=-1) + 1e-6).sqrt()
    norms = [(expert_outputs[name].pow(2).mean(dim=-1) + 1e-6).sqrt() for name in stom.operator_set]
    gate_base_feat = torch.stack([hist_mean, hist_std, last_value, smooth_slope, residual_energy, *norms], dim=-1)
    if stom.disable_operator_magnitude_features:
        gate_base_feat = gate_base_feat.clone()
        gate_base_feat[..., 5:5 + stom.n_gate_ops] = 0.0
    horizon = torch.arange(model.pred_len, device=x_hist.device, dtype=gate_base_feat.dtype) / max(model.pred_len, 1)
    horizon = horizon.view(1, 1, model.pred_len, 1).expand(x_hist.size(0), x_hist.size(1), -1, -1)
    if stom.ablation_mode == "no_horizon_gate":
        horizon = torch.zeros_like(horizon)
    gate_feat = torch.cat([gate_base_feat.unsqueeze(2).expand(-1, -1, model.pred_len, -1), horizon], dim=-1)
    logits = stom.gate_mlp(gate_feat.float()).to(dtype=x_hist.dtype)
    mask = stom._expert_mask(x_hist.device, x_hist.dtype).view(1, 1, 1, stom.n_gate_ops) * stom._phase_repair_mask(x_hist.device, x_hist.dtype)
    gate = torch.softmax(logits.masked_fill(mask <= 0, -1e9), dim=-1)
    return logits, gate, gate_feat


def stom_parts(model, x_hist, y_phase, A_override=None, force_fp32=False):
    stom = model.stom_head
    work = x_hist.float() if force_fp32 else x_hist
    y_phase = y_phase.float() if force_fp32 else y_phase
    smooth = stom.moving_average(work)
    residual = work - smooth
    y_smooth = stom.trend_proj(smooth)
    y_local = stom.local_proj(F.gelu(stom.local_conv(residual)))
    A, phi, delta, alpha, phase_strength, recent_strength, sigma_phase, sigma_recent = stom._spectral_kernel(work.device, work.dtype)
    if A_override is not None:
        A = A_override.to(device=work.device, dtype=work.dtype)
    y_spec_h = torch.einsum("hpl,bcl->bhcp", A, residual)
    head_w = torch.softmax(stom.spec_head_logits, dim=0).to(device=work.device, dtype=work.dtype)
    spec_gain = F.softplus(stom.raw_spec_gain).to(device=work.device, dtype=work.dtype)
    y_spec = spec_gain * torch.einsum("h,bhcp->bcp", head_w, y_spec_h)
    logits, gate, gate_feat = gate_from_outputs(model, work, y_phase, y_spec, y_smooth, y_local)
    expert = {"phase": y_phase, "spectral": y_spec, "smooth": y_smooth, "local": y_local}
    stack = torch.stack([expert[name] for name in stom.operator_set], dim=-1)
    y = (gate * stack).sum(dim=-1)
    return {
        "x_hist": work,
        "smooth": smooth,
        "residual": residual,
        "A": A,
        "phi": phi,
        "alpha": alpha,
        "gamma_spec": spec_gain,
        "head_w": head_w,
        "y_spec_h": y_spec_h,
        "y_phase": y_phase,
        "y_spec": y_spec,
        "y_smooth": y_smooth,
        "y_local": y_local,
        "gate_logits": logits,
        "gate": gate,
        "gate_feat": gate_feat,
        "expert_stack": stack,
        "y_norm": y,
        "operator_set": list(stom.operator_set),
    }


def exact_cspec(parts):
    residual = parts["residual"].float()
    A = parts["A"].float()
    head_w = parts["head_w"].float()
    gain = parts["gamma_spec"].float()
    op_index = {name: i for i, name in enumerate(parts["operator_set"])}
    gate_spec = parts["gate"][..., op_index["spectral"]].float()
    per_head_source = A.view(1, A.shape[0], 1, A.shape[1], A.shape[2]) * residual.unsqueeze(1).unsqueeze(3)
    agg_source = gain * torch.einsum("h,bhcpl->bcpl", head_w, per_head_source)
    cspec = gate_spec.unsqueeze(-1) * agg_source
    return agg_source, cspec


def eval_model(args, model, loader, device, collect_parts=False):
    preds, trues = [], []
    part_rows = []
    with torch.inference_mode():
        for i, data in enumerate(loader):
            bx, by, bxm, bym, dec = model_batch(args, data, device)
            pred = model(bx, bxm, dec, bym)
            y = target_from_batch(args, by, model).permute(0, 2, 1)
            pred = pred[:, -args.pred_len:, -model.number_of_targets:]
            preds.append(pred.detach().float().cpu().numpy())
            trues.append(y.detach().float().cpu().numpy())
            if collect_parts and i == 0:
                xh, yp, mean, std = extract_xhist_yphase(model, bx, bxm, bym)
                parts = stom_parts(model, xh, yp, force_fp32=True)
                recon = denorm(model, parts["y_norm"], mean, std).permute(0, 2, 1)
                part_rows.append(float((recon - pred).abs().max().item()))
    p = np.concatenate(preds, axis=0)
    t = np.concatenate(trues, axis=0)
    d = p - t
    return {
        "mse": float(np.mean(d * d)),
        "mae": float(np.mean(np.abs(d))),
        "preds": p,
        "trues": t,
        "first_batch_recon_max_diff": part_rows[0] if part_rows else None,
    }


def pearson(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return "N/A"
    return float(np.corrcoef(a, b)[0, 1])


def ranks(x):
    x = np.asarray(x).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    return r


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def audit(out):
    rows = []
    for setting, cfg in SETTINGS.items():
        for key in ["cmd", "ckpt", "lin_cmd", "lin_ckpt", "raw", "no_mag_cmd", "no_mag_ckpt"]:
            if key in cfg:
                p = Path(cfg[key])
                rows.append({"setting": setting, "item": key, "path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() else ""})
    append_json(out / "checkpoint_manifest.json", {"seed": SEED, "items": rows})
    code_map = [
        "STOM model: models/stom.py: STOMHead and Model.forward.",
        "Local LinAttn: models/stom.py: CAPSAttention._forward_linattn selected by predictor=LinAttn.",
        "phase output: Model.forward builds y_phase from temporal_processor/value_token_mixer.",
        "smooth S and residual R: STOMHead.moving_average(x_hist), residual=x_hist-smooth.",
        "local output: depthwise local_conv(residual) followed by local_proj.",
        "spectral scores: STOMHead._spectral_kernel returns phi and A=softmax(phi).",
        "A/alpha/gamma_spec: _spectral_kernel alpha; gamma_spec=softplus(raw_spec_gain).",
        "gate feature order: hist_mean,hist_std,last_value,smooth_slope,residual_energy,operator output RMS in operator_set order,horizon.",
        "gate logits/probabilities: gate_mlp(gate_feat), masked softmax over operator_set.",
        "normalization: Model.forward subtracts last input value and uses std=1; output is denormalized by adding last value.",
        "test dataloader/metrics: exp/exp_main.py vali/test use data_provider(...,'test') and MSE/MAE over final pred_len target channels.",
        "checkpoint loading: unwrap optional module. prefix, load_state_dict(strict=False).",
    ]
    (out / "code_map.txt").write_text("\n".join(code_map) + "\n")
    old = [
        "# Old Diagnostics Audit",
        "",
        "- Existing `diagnostics/` is treated as legacy output and is not overwritten.",
        "- `diagnostics/source_suppression/source_indices.npz` is a placeholder in the old low-cost script (`placeholder=np.array([1])`), so it is not a formal result.",
        "- Old temperature source sweep sets realized source-profile entropy from the same normalized A-like array and leaves source MSE/MAE as N/A; it must be rerun.",
        "- Correctability uses raw-space operator directions without adding the inverse-normalization mean to each column.",
        "- Old efficiency uses short warmup/iteration defaults and mixes optional source materialization inside the same script; rerun with 30 warmup and 100 measured iterations.",
        "- Old gate-no-magnitude checkpoints can be reused as checkpoints, but the evaluation summaries are rerun through this script.",
        "- Smoke files under `interpretability/raw/*_smoke.*` are smoke tests, not formal outputs.",
        "- Existing full raw files under `interpretability/raw/*_seed2026.npz` may be used for audit comparison only; formal summaries below are regenerated.",
    ]
    (out / "old_results_audit.md").write_text("\n".join(old) + "\n")
    append_json(out / "run_manifest.json", env_info("audit"))


def source_suppression(out, settings, device, max_batches=0):
    raw_rows, summary_rows = [], []
    for setting in settings:
        cfg = SETTINGS[setting]
        args, model = load_model(cfg["cmd"], cfg["ckpt"], device, use_stom=1)
        _, loader = data_provider(args, "test")
        total_samples = 0
        arrays = {}
        stat = {}
        rng = np.random.default_rng(SEED)
        with torch.inference_mode():
            for bidx, data in enumerate(loader):
                if max_batches and bidx >= max_batches:
                    break
                bx, by, bxm, bym, dec = model_batch(args, data, device)
                xh, yp, mean, std = extract_xhist_yphase(model, bx, bxm, bym)
                parts = stom_parts(model, xh, yp, force_fp32=True)
                pred_norm = parts["y_norm"]
                pred = denorm(model, pred_norm, mean, std)
                true = target_from_batch(args, by, model)
                agg, cspec = exact_cspec(parts)
                op_index = {name: i for i, name in enumerate(parts["operator_set"])}
                spec_i = op_index["spectral"]
                z_spec = parts["gate"][..., spec_i].float() * parts["y_spec"].float()
                score = cspec.abs().mean(dim=2)
                B, C, P, L = cspec.shape
                total_samples += B
                for k in [1, 5, 10]:
                    kk = min(k, L)
                    top_idx = torch.topk(score, kk, dim=-1).indices
                    masks = [("topk", 0, top_idx)]
                    for rep in range(10):
                        rand = np.empty((B, C, kk), dtype=np.int64)
                        top_np = top_idx.detach().cpu().numpy()
                        for bi in range(B):
                            for ci in range(C):
                                while True:
                                    cand = rng.choice(L, size=kk, replace=False)
                                    if not np.array_equal(np.sort(cand), np.sort(top_np[bi, ci])):
                                        break
                                rand[bi, ci] = cand
                        masks.append(("random", rep, torch.as_tensor(rand, device=device)))
                    for mask_type, rep, idx in masks:
                        mask = torch.zeros((B, C, L), device=device, dtype=torch.bool).scatter_(-1, idx, True)
                        traced = (cspec * mask.unsqueeze(2)).sum(dim=-1)
                        unweighted_selected = (agg * mask.unsqueeze(2)).sum(dim=-1)
                        y_spec_masked = parts["y_spec"].float() - unweighted_selected
                        z_spec_masked = parts["gate"][..., spec_i].float() * y_spec_masked
                        fixed_norm = pred_norm.float() - z_spec + z_spec_masked
                        delta_actual = pred_norm.float() - fixed_norm
                        delta_traced = traced
                        rec_logits, rec_gate, _ = gate_from_outputs(model, xh.float(), yp.float(), y_spec_masked, parts["y_smooth"].float(), parts["y_local"].float())
                        rec_stack = torch.stack([
                            {"phase": parts["y_phase"].float(), "spectral": y_spec_masked, "smooth": parts["y_smooth"].float(), "local": parts["y_local"].float()}[name]
                            for name in parts["operator_set"]
                        ], dim=-1)
                        rec_norm = (rec_gate.float() * rec_stack).sum(-1)
                        delta_rec = pred_norm.float() - rec_norm
                        fixed_pred = denorm(model, fixed_norm, mean, std)
                        rec_pred = denorm(model, rec_norm, mean, std)
                        diff = (delta_actual - delta_traced).detach().cpu().numpy()
                        key = (setting, "fixed_gate", mask_type, k)
                        stat.setdefault(key, []).append({
                            "consistency_mae": float(np.abs(diff).mean()),
                            "consistency_rmse": float(np.sqrt((diff * diff).mean())),
                            "max_abs_error": float(np.abs(diff).max()),
                            "pearson": pearson(delta_actual.cpu().numpy(), delta_traced.cpu().numpy()),
                            "spearman": spearman(delta_actual.cpu().numpy(), delta_traced.cpu().numpy()),
                            "sign_agreement": float((torch.sign(delta_actual) == torch.sign(delta_traced)).float().mean().item()),
                            "mean_abs_prediction_change": float(delta_actual.abs().mean().item()),
                            "mse_before": float(((pred - true) ** 2).mean().item()),
                            "mse_after": float(((fixed_pred - true) ** 2).mean().item()),
                            "mae_before": float((pred - true).abs().mean().item()),
                            "mae_after": float((fixed_pred - true).abs().mean().item()),
                            "gate_l1": 0.0,
                            "spec_gate_change": 0.0,
                        })
                        key2 = (setting, "recomputed_gate", mask_type, k)
                        stat.setdefault(key2, []).append({
                            "consistency_mae": "N/A",
                            "consistency_rmse": "N/A",
                            "max_abs_error": "N/A",
                            "pearson": pearson(delta_rec.cpu().numpy(), delta_traced.cpu().numpy()),
                            "spearman": spearman(delta_rec.cpu().numpy(), delta_traced.cpu().numpy()),
                            "sign_agreement": float((torch.sign(delta_rec) == torch.sign(delta_traced)).float().mean().item()),
                            "mean_abs_prediction_change": float(delta_rec.abs().mean().item()),
                            "mse_before": float(((pred - true) ** 2).mean().item()),
                            "mse_after": float(((rec_pred - true) ** 2).mean().item()),
                            "mae_before": float((pred - true).abs().mean().item()),
                            "mae_after": float((rec_pred - true).abs().mean().item()),
                            "gate_l1": float((rec_gate.float() - parts["gate"].float()).abs().mean().item()),
                            "spec_gate_change": float((rec_gate[..., spec_i].float() - parts["gate"][..., spec_i].float()).mean().item()),
                        })
                        sid_base = total_samples - B
                        idx_np = idx.detach().cpu().numpy()
                        da_np = delta_actual.detach().cpu().numpy()
                        dt_np = delta_traced.detach().cpu().numpy()
                        dr_np = delta_rec.detach().cpu().numpy()
                        for bi in range(B):
                            for ci in range(C):
                                row_id = len(raw_rows)
                                raw_rows.append({
                                    "row_id": row_id,
                                    "setting": setting,
                                    "sample_id": sid_base + bi,
                                    "variable_id": ci,
                                    "k": k,
                                    "mask_type": mask_type,
                                    "random_repeat_id": rep,
                                    "selected_source_indices": " ".join(map(str, idx_np[bi, ci].tolist())),
                                })
                                arrays[f"delta_actual_{row_id}"] = da_np[bi, ci].astype(np.float32)
                                arrays[f"delta_traced_{row_id}"] = dt_np[bi, ci].astype(np.float32)
                                arrays[f"delta_recomputed_{row_id}"] = dr_np[bi, ci].astype(np.float32)
        for key, vals in stat.items():
            setting, protocol, mask_type, k = key
            row = {"setting": setting, "protocol": protocol, "mask_type": mask_type, "k": k, "num_batches": len(vals), "num_test_samples": total_samples}
            for metric in vals[0].keys():
                vv = [v[metric] for v in vals if isinstance(v[metric], (int, float))]
                row[metric] = float(np.mean(vv)) if vv else vals[0][metric]
            summary_rows.append(row)
        np.savez_compressed(out / "source_suppression" / f"{setting}_deltas.npz", **arrays)
    write_csv(out / "source_suppression/source_suppression_per_sample.csv", raw_rows)
    write_csv(out / "source_suppression/source_suppression_summary.csv", summary_rows)
    np.savez_compressed(out / "source_suppression/source_indices.npz", row_ids=np.arange(len(raw_rows), dtype=np.int64), selected=np.array([r["selected_source_indices"] for r in raw_rows], dtype=object))
    cfg = env_info("source_suppression")
    cfg["settings"] = settings
    cfg["seed"] = SEED
    append_json(out / "source_suppression/config.json", cfg)
    passed = bool(summary_rows) and all((r["protocol"] != "fixed_gate") or float(r["consistency_mae"]) < 1e-5 for r in summary_rows)
    append_json(out / "source_suppression/validation_report.json", {"passed": passed, "rows": len(raw_rows), "summary_rows": len(summary_rows), "placeholder": False})


def temperature(out, settings, device, max_batches=0):
    gate_rows, source_rows = [], []
    curves = {}
    for setting in settings:
        cfg = SETTINGS[setting]
        args, model = load_model(cfg["cmd"], cfg["ckpt"], device, use_stom=1)
        _, loader = data_provider(args, "test")
        g_stats = {}
        s_stats = {}
        for tau in [0.5, 0.75, 1.0, 1.5, 2.0]:
            g_stats[tau] = {"sq": 0.0, "abs": 0.0, "n": 0, "ent": [], "gate": [], "share": [], "mad": [], "maxdiff": 0.0}
            s_stats[tau] = {"sq": 0.0, "abs": 0.0, "n": 0, "tent": [], "qent": [], "top1": [], "top5": [], "top10": [], "qcorr": [], "mad": [], "maxdiff": 0.0}
        with torch.inference_mode():
            for bidx, data in enumerate(loader):
                if max_batches and bidx >= max_batches:
                    break
                bx, by, bxm, bym, dec = model_batch(args, data, device)
                xh, yp, mean, std = extract_xhist_yphase(model, bx, bxm, bym)
                parts = stom_parts(model, xh, yp, force_fp32=True)
                base = denorm(model, parts["y_norm"], mean, std)
                true = target_from_batch(args, by, model)
                logits = parts["gate_logits"].float()
                stack = parts["expert_stack"].float()
                base_gate = parts["gate"].float()
                base_cspec = exact_cspec(parts)[1].abs()
                q_base = base_cspec / (base_cspec.sum(dim=-1, keepdim=True) + EPS)
                for tau in g_stats:
                    gg = torch.softmax(logits / tau, dim=-1)
                    yy = (gg * stack).sum(-1)
                    pred = denorm(model, yy, mean, std)
                    d = pred - true
                    gs = g_stats[tau]
                    gs["sq"] += float((d * d).sum().item())
                    gs["abs"] += float(d.abs().sum().item())
                    gs["n"] += d.numel()
                    gs["ent"].append(float((-(gg.clamp_min(EPS) * gg.clamp_min(EPS).log()).sum(-1)).mean().item()))
                    gs["gate"].append(gg.mean(dim=(0, 1, 2)).detach().cpu().numpy())
                    contrib = (gg * stack).abs().mean(dim=(0, 1, 2))
                    gs["share"].append((contrib / (contrib.sum() + EPS)).detach().cpu().numpy())
                    gs["mad"].append(float((pred - base).abs().mean().item()))
                    gs["maxdiff"] = max(gs["maxdiff"], float((pred - base).abs().max().item()) if tau == 1.0 else gs["maxdiff"])
                for tau in s_stats:
                    A = torch.softmax(parts["phi"].float() / tau, dim=-1)
                    pp = stom_parts(model, xh, yp, A_override=A, force_fp32=True)
                    op_index = {name: i for i, name in enumerate(pp["operator_set"])}
                    spec_i = op_index["spectral"]
                    pred_norm = parts["y_norm"].float() - parts["gate"][..., spec_i].float() * parts["y_spec"].float() + parts["gate"][..., spec_i].float() * pp["y_spec"].float()
                    pred = denorm(model, pred_norm, mean, std)
                    d = pred - true
                    ss = s_stats[tau]
                    ss["sq"] += float((d * d).sum().item())
                    ss["abs"] += float(d.abs().sum().item())
                    ss["n"] += d.numel()
                    transfer_ent = -(A.clamp_min(EPS) * A.clamp_min(EPS).log()).sum(-1)
                    _, cspec = exact_cspec({**pp, "gate": parts["gate"]})
                    q = cspec.abs() / (cspec.abs().sum(dim=-1, keepdim=True) + EPS)
                    qent = -(q.clamp_min(EPS) * q.clamp_min(EPS).log()).sum(-1)
                    top = torch.sort(q, dim=-1, descending=True).values
                    ss["tent"].append(float(transfer_ent.mean().item()))
                    ss["qent"].append(float(qent.mean().item()))
                    ss["top1"].append(float(top[..., :1].sum(-1).mean().item()))
                    ss["top5"].append(float(top[..., :5].sum(-1).mean().item()))
                    ss["top10"].append(float(top[..., :10].sum(-1).mean().item()))
                    ss["qcorr"].append(pearson(q.detach().cpu().numpy(), q_base.detach().cpu().numpy()))
                    ss["mad"].append(float((pred - base).abs().mean().item()))
                    ss["maxdiff"] = max(ss["maxdiff"], float((pred - base).abs().max().item()) if tau == 1.0 else ss["maxdiff"])
        base_gate_curve = np.mean(g_stats[1.0]["gate"], axis=0)
        for tau, gs in g_stats.items():
            gate_curve = np.mean(gs["gate"], axis=0)
            share = np.mean(gs["share"], axis=0)
            row = {"setting": setting, "tau_g": tau, "tau_s": 1.0, "mse": gs["sq"] / gs["n"], "mae": gs["abs"] / gs["n"], "gate_entropy": float(np.mean(gs["ent"])), "prediction_mad_relative_tau1": float(np.mean(gs["mad"])), "prediction_maxdiff_tau1": gs["maxdiff"], "gate_curve_corr_with_tau1": pearson(gate_curve, base_gate_curve)}
            for i, name in enumerate(model.stom_head.operator_set):
                row[f"mean_gate_{name}"] = float(gate_curve[i])
                row[f"absolute_contribution_share_{name}"] = float(share[i])
            gate_rows.append(row)
            curves[f"{setting}_gate_tau_{tau}"] = gate_curve
        for tau, ss in s_stats.items():
            source_rows.append({"setting": setting, "tau_g": 1.0, "tau_s": tau, "mse": ss["sq"] / ss["n"], "mae": ss["abs"] / ss["n"], "transfer_entropy": float(np.mean(ss["tent"])), "realized_source_profile_entropy": float(np.mean(ss["qent"])), "top1_concentration": float(np.mean(ss["top1"])), "top5_concentration": float(np.mean(ss["top5"])), "top10_concentration": float(np.mean(ss["top10"])), "source_profile_corr_relative_tau1": np.mean([x for x in ss["qcorr"] if isinstance(x, float)]) if any(isinstance(x, float) for x in ss["qcorr"]) else "N/A", "prediction_mad_relative_tau1": float(np.mean(ss["mad"])), "prediction_maxdiff_tau1": ss["maxdiff"]})
    write_csv(out / "temperature/temperature_gate_summary.csv", gate_rows)
    write_csv(out / "temperature/temperature_source_summary.csv", source_rows)
    np.savez_compressed(out / "temperature/temperature_curves.npz", **curves)
    append_json(out / "temperature/config.json", env_info("temperature"))
    passed = all(float(r["prediction_maxdiff_tau1"]) < 1e-5 for r in gate_rows if float(r["tau_g"]) == 1.0) and all(float(r["prediction_maxdiff_tau1"]) < 1e-5 for r in source_rows if float(r["tau_s"]) == 1.0)
    append_json(out / "temperature/validation_report.json", {"passed": passed, "gate_rows": len(gate_rows), "source_rows": len(source_rows)})


def solve_box(Z, e):
    best = float(e @ e)
    best_lam = np.zeros(Z.shape[1], dtype=np.float64)
    for st in itertools.product([0, 1, 2], repeat=Z.shape[1]):
        lam = np.array([1.0 if x == 1 else 0.0 for x in st], dtype=np.float64)
        free = [i for i, x in enumerate(st) if x == 2]
        rhs = e - Z @ lam
        if free:
            sol = np.linalg.lstsq(Z[:, free], rhs, rcond=None)[0]
            if (sol < -1e-9).any() or (sol > 1 + 1e-9).any():
                continue
            lam[free] = np.clip(sol, 0, 1)
        obj = float(((e - Z @ lam) ** 2).sum())
        if obj < best:
            best = obj
            best_lam = lam.copy()
    return best_lam, best


def correctability(out, settings, device, max_batches=0):
    per = []
    for setting in settings:
        cfg = SETTINGS[setting]
        args, model = load_model(cfg["cmd"], cfg["ckpt"], device, use_stom=1)
        _, loader = data_provider(args, "test")
        sample_base = 0
        with torch.inference_mode():
            for bidx, data in enumerate(loader):
                if max_batches and bidx >= max_batches:
                    break
                bx, by, bxm, bym, dec = model_batch(args, data, device)
                xh, yp, mean, std = extract_xhist_yphase(model, bx, bxm, bym)
                parts = stom_parts(model, xh, yp, force_fp32=True)
                pred = denorm(model, parts["y_norm"], mean, std)
                true = target_from_batch(args, by, model)
                
                
                Z = parts["gate"].float() * parts["expert_stack"].float() * std[:, -model.number_of_targets:].unsqueeze(-1)
                H = model.pred_len
                regions = {"full": (0, H), "near": (0, H // 3), "middle": (H // 3, 2 * H // 3), "far": (2 * H // 3, H)}
                Znp = Z.detach().cpu().numpy()
                enp = (pred - true).detach().cpu().numpy()
                B, C = enp.shape[:2]
                for bi in range(B):
                    for region, (lo, hi) in regions.items():
                        e = enp[bi, :, lo:hi].reshape(-1).astype(np.float64)
                        M = Znp[bi, :, lo:hi, :].reshape(-1, Znp.shape[-1]).astype(np.float64)
                        norm = float(e @ e) + EPS
                        pinv = np.linalg.pinv(M, rcond=1e-8)
                        proj = M @ (pinv @ e)
                        Cw = float((proj @ proj) / norm)
                        lam, obj = solve_box(M, e)
                        removal, shrinkage = [], []
                        for oi in range(M.shape[1]):
                            z = M[:, oi]
                            removal.append(max(0.0, (norm - float(((e - z) ** 2).sum())) / norm))
                            s = min(1.0, max(0.0, float(e @ z) / (float(z @ z) + EPS)))
                            shrinkage.append(max(0.0, (norm - float(((e - s * z) ** 2).sum())) / norm))
                        per.append({"setting": setting, "sample_id": sample_base + bi, "region": region, "C_omega": Cw, "rank": int(np.linalg.matrix_rank(M)), "condition_number": float(np.linalg.cond(M)), "projected_error_energy": float(proj @ proj), "uncorrectable_error_energy": float(norm - proj @ proj), "box_improvement": max(0.0, (norm - obj) / norm), "best_single_removal": float(max(removal)), "best_single_shrinkage": float(max(shrinkage)), "lambda_box": json.dumps(lam.tolist())})
                sample_base += B
    write_csv(out / "correctability/correctability_per_sample.csv", per)
    summary = []
    for setting in sorted({r["setting"] for r in per}):
        for region in ["full", "near", "middle", "far"]:
            rr = [r for r in per if r["setting"] == setting and r["region"] == region]
            C = np.array([r["C_omega"] for r in rr])
            B = np.array([r["box_improvement"] for r in rr])
            summary.append({"setting": setting, "region": region, "num_samples": len(rr), "mean_C_omega": float(C.mean()), "median_C_omega": float(np.median(C)), "std_C_omega": float(C.std()), "mean_box_improvement": float(B.mean()), "mean_best_single_removal": float(np.mean([r["best_single_removal"] for r in rr])), "mean_best_single_shrinkage": float(np.mean([r["best_single_shrinkage"] for r in rr])), "pearson_C_vs_box": pearson(C, B), "spearman_C_vs_box": spearman(C, B), "box_over_C_ratio": float(B.mean() / (C.mean() + EPS)), "condition_number_median": float(np.median([r["condition_number"] for r in rr])), "rank_distribution": json.dumps({str(i): int(sum(r["rank"] == i for r in rr)) for i in range(1, 5)})})
    write_csv(out / "correctability/correctability_summary.csv", summary)
    np.savez_compressed(out / "correctability/correctability_arrays.npz", C=np.array([r["C_omega"] for r in per]), box=np.array([r["box_improvement"] for r in per]))
    append_json(out / "correctability/config.json", env_info("correctability"))
    passed = all(-1e-6 <= r["C_omega"] <= 1 + 1e-5 and r["best_single_shrinkage"] + 1e-5 >= r["best_single_removal"] for r in per)
    append_json(out / "correctability/validation_report.json", {"passed": passed, "rows": len(per)})


def efficiency(out, device):
    rows, raw = [], []
    cfg = SETTINGS["ETTm1_96"]
    modes = [
        ("local_forecast", cfg["lin_cmd"], cfg["lin_ckpt"], 0, "forecast"),
        ("stom_forecast", cfg["cmd"], cfg["ckpt"], 1, "forecast"),
        ("stom_retain_Z", cfg["cmd"], cfg["ckpt"], 1, "Z"),
        ("stom_retain_Z_Cspec", cfg["cmd"], cfg["ckpt"], 1, "Cspec"),
    ]
    base_pred = None
    for mode_name, cmd, ckpt, use_stom, mode in modes:
        args, model = load_model(cmd, ckpt, device, use_stom=use_stom)
        _, loader = data_provider(args, "test")
        data = next(iter(loader))
        bx, by, bxm, bym, dec = model_batch(args, data, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            base_alloc = torch.cuda.memory_allocated()
            base_reserved = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
        times = []
        with torch.inference_mode():
            for i in range(130):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                if use_stom and mode != "forecast":
                    xh, yp, mean, std = extract_xhist_yphase(model, bx, bxm, bym)
                    parts = stom_parts(model, xh, yp, force_fp32=False)
                    pred = denorm(model, parts["y_norm"], mean, std).permute(0, 2, 1)
                    if mode == "Cspec":
                        exact_cspec(parts)
                else:
                    pred = model(bx, bxm, dec, bym)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if i >= 30:
                    times.append((time.perf_counter() - t0) * 1000)
        pred_cpu = pred.detach().float().cpu().numpy()
        if mode_name == "stom_forecast":
            base_pred = pred_cpu
        diff = "" if base_pred is None or not mode_name.startswith("stom") else float(np.max(np.abs(pred_cpu - base_pred)))
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        pres = torch.cuda.max_memory_reserved() if device.type == "cuda" else 0
        row = {"setting": "ETTm1_96", "mode": mode_name, "batch_size": int(bx.shape[0]), "dtype": str(bx.dtype), "device": str(device), "total_params": sum(p.numel() for p in model.parameters()), "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad), "flops": "N/A", "flops_field": "N/A", "latency_mean_ms": float(np.mean(times)), "latency_std_ms": float(np.std(times)), "latency_median_ms": float(np.median(times)), "latency_p95_ms": float(np.percentile(times, 95)), "per_sample_latency_ms": float(np.mean(times) / bx.shape[0]), "throughput_samples_per_sec": float(1000 * bx.shape[0] / np.mean(times)), "baseline_allocated_mb": float((base_alloc if device.type == "cuda" else 0) / 2**20), "peak_allocated_mb": float(peak / 2**20), "incremental_peak_mb": float(((peak - base_alloc) if device.type == "cuda" else 0) / 2**20), "peak_reserved_mb": float(pres / 2**20), "prediction_maxdiff_vs_stom_forecast": diff}
        rows.append(row)
        for i, x in enumerate(times):
            raw.append({"mode": mode_name, "iteration": i, "latency_ms": x})
    write_csv(out / "efficiency/efficiency_summary.csv", rows)
    write_csv(out / "efficiency/efficiency_raw.csv", raw)
    append_json(out / "efficiency/config.json", env_info("efficiency"))
    passed = len(raw) == 400 and all((not str(r["mode"]).startswith("stom_retain")) or float(r["prediction_maxdiff_vs_stom_forecast"]) < 1e-5 for r in rows if r["prediction_maxdiff_vs_stom_forecast"] != "")
    append_json(out / "efficiency/validation_report.json", {"passed": passed, "raw_rows": len(raw)})


def gate_no_magnitude(out, settings, device, max_batches=0):
    rows = []
    for setting in settings:
        cfg = SETTINGS[setting]
        if "no_mag_ckpt" not in cfg or not Path(cfg["no_mag_ckpt"]).exists():
            rows.append({"setting": setting, "status": "FAILED", "reason": "no no-magnitude checkpoint"})
            continue
        for variant, cmd, ckpt, disable in [("full", cfg["cmd"], cfg["ckpt"], 0), ("no_magnitude", cfg["no_mag_cmd"], cfg["no_mag_ckpt"], 1)]:
            args, model = load_model(cmd, ckpt, device, use_stom=1, disable_mag=disable)
            _, loader = data_provider(args, "test")
            sq = ab = n = 0
            gates, shares, maxp, frac80, entropies, argmax = [], [], [], [], [], []
            assertion_max = 0.0
            with torch.inference_mode():
                for bidx, data in enumerate(loader):
                    if max_batches and bidx >= max_batches:
                        break
                    bx, by, bxm, bym, dec = model_batch(args, data, device)
                    xh, yp, mean, std = extract_xhist_yphase(model, bx, bxm, bym)
                    parts = stom_parts(model, xh, yp, force_fp32=True)
                    pred = denorm(model, parts["y_norm"], mean, std)
                    true = target_from_batch(args, by, model)
                    d = pred - true
                    sq += float((d * d).sum().item())
                    ab += float(d.abs().sum().item())
                    n += d.numel()
                    gate = parts["gate"].float()
                    gates.append(gate.mean(dim=(0, 1, 2)).cpu().numpy())
                    contrib = (gate * parts["expert_stack"].float()).abs().mean(dim=(0, 1, 2))
                    shares.append((contrib / (contrib.sum() + EPS)).cpu().numpy())
                    max_gate = gate.max(dim=-1).values
                    maxp.append(float(max_gate.mean().item()))
                    frac80.append(float((max_gate > 0.8).float().mean().item()))
                    entropies.append(float((-(gate.clamp_min(EPS) * gate.clamp_min(EPS).log()).sum(dim=-1)).mean().item()))
                    argmax.append(np.bincount(gate.argmax(dim=-1).detach().cpu().numpy().reshape(-1), minlength=gate.shape[-1]) / gate[..., 0].numel())
                    if disable:
                        assertion_max = max(assertion_max, float(parts["gate_feat"][..., 5:5 + model.stom_head.n_gate_ops].abs().max().item()))
            gate_mean = np.mean(gates, axis=0)
            share_mean = np.mean(shares, axis=0)
            arg_mean = np.mean(argmax, axis=0)
            row = {"setting": setting, "variant": variant, "status": "SUCCESS", "mse": sq / n, "mae": ab / n, "gate_entropy": float(np.mean(entropies)), "mean_max_gate_probability": float(np.mean(maxp)), "fraction_max_gate_gt_0_8": float(np.mean(frac80)), "magnitude_feature_assertion_max_abs": assertion_max}
            for i, name in enumerate(model.stom_head.operator_set):
                row[f"mean_gate_{name}"] = float(gate_mean[i])
                row[f"absolute_contribution_share_{name}"] = float(share_mean[i])
                row[f"argmax_fraction_{name}"] = float(arg_mean[i])
            rows.append(row)
    by_setting = {}
    for row in rows:
        by_setting.setdefault(row["setting"], {})[row.get("variant")] = row
    for setting, d in by_setting.items():
        if "full" in d and "no_magnitude" in d:
            d["no_magnitude"]["delta_mse_vs_full"] = float(d["no_magnitude"]["mse"] - d["full"]["mse"])
            d["no_magnitude"]["delta_mae_vs_full"] = float(d["no_magnitude"]["mae"] - d["full"]["mae"])
    write_csv(out / "gate_no_magnitude/gate_no_magnitude_results.csv", rows)
    append_json(out / "gate_no_magnitude/config.json", env_info("gate_no_magnitude"))
    passed = all(r.get("status") == "SUCCESS" and (r.get("variant") != "no_magnitude" or float(r["magnitude_feature_assertion_max_abs"]) == 0.0) for r in rows)
    append_json(out / "gate_no_magnitude/validation_report.json", {"passed": passed, "rows": len(rows)})


def bootstrap(out, settings, device, max_batches=0):
    rows, summary = [], []
    rng = np.random.default_rng(SEED)
    for setting in settings:
        cfg = SETTINGS[setting]
        args, stom = load_model(cfg["cmd"], cfg["ckpt"], device, use_stom=1)
        largs, lin = load_model(cfg["lin_cmd"], cfg["lin_ckpt"], device, use_stom=0)
        _, loader = data_provider(args, "test")
        sid = 0
        with torch.inference_mode():
            for bidx, data in enumerate(loader):
                if max_batches and bidx >= max_batches:
                    break
                bx, by, bxm, bym, dec = model_batch(args, data, device)
                pred_s = stom(bx, bxm, dec, bym)[:, -args.pred_len:, -stom.number_of_targets:]
                pred_l = lin(bx, bxm, dec, bym)[:, -args.pred_len:, -lin.number_of_targets:]
                true = target_from_batch(args, by, stom).permute(0, 2, 1)
                ds = (pred_s - true).detach().cpu().numpy()
                dl = (pred_l - true).detach().cpu().numpy()
                for i in range(ds.shape[0]):
                    rows.append({"setting": setting, "sample_id": sid, "stom_mse": float(np.mean(ds[i] ** 2)), "stom_mae": float(np.mean(np.abs(ds[i]))), "local_mse": float(np.mean(dl[i] ** 2)), "local_mae": float(np.mean(np.abs(dl[i]))), "delta_mse_stom_minus_local": float(np.mean(ds[i] ** 2) - np.mean(dl[i] ** 2)), "delta_mae_stom_minus_local": float(np.mean(np.abs(ds[i])) - np.mean(np.abs(dl[i])))})
                    sid += 1
        dm = np.array([r["delta_mse_stom_minus_local"] for r in rows if r["setting"] == setting])
        da = np.array([r["delta_mae_stom_minus_local"] for r in rows if r["setting"] == setting])
        boot_m, boot_a = [], []
        for _ in range(2000):
            idx = rng.integers(0, len(dm), size=len(dm))
            boot_m.append(float(dm[idx].mean()))
            boot_a.append(float(da[idx].mean()))
        summary.append({"setting": setting, "num_samples": len(dm), "mean_delta_mse": float(dm.mean()), "mse_ci_low": float(np.percentile(boot_m, 2.5)), "mse_ci_high": float(np.percentile(boot_m, 97.5)), "median_delta_mse": float(np.median(dm)), "prob_delta_mse_lt_0": float(np.mean(np.array(boot_m) < 0)), "sample_mse_win_rate": float(np.mean(dm < 0)), "mean_delta_mae": float(da.mean()), "mae_ci_low": float(np.percentile(boot_a, 2.5)), "mae_ci_high": float(np.percentile(boot_a, 97.5)), "median_delta_mae": float(np.median(da)), "prob_delta_mae_lt_0": float(np.mean(np.array(boot_a) < 0)), "sample_mae_win_rate": float(np.mean(da < 0))})
    write_csv(out / "bootstrap/bootstrap_per_sample.csv", rows)
    write_csv(out / "bootstrap/bootstrap_summary.csv", summary)
    append_json(out / "bootstrap/config.json", env_info("bootstrap"))
    append_json(out / "bootstrap/validation_report.json", {"passed": bool(summary), "rows": len(rows), "repetitions": 2000})


def summarize(out):
    items = []
    for exp in ["source_suppression", "temperature", "efficiency", "correctability", "gate_no_magnitude", "bootstrap"]:
        report = out / exp / "validation_report.json"
        status = "FAILED"
        if report.exists():
            data = json.loads(report.read_text())
            status = "SUCCESS" if data.get("passed") else "PARTIAL"
        items.append({"experiment": exp, "status": status, "validation_report": str(report), "exists": report.exists()})
    write_csv(out / "summary_all.csv", items)
    readme = ["# Final STOM Low-Cost Diagnostics", "", f"Seed: {SEED}", "", "Outputs are generated from existing checkpoints; no benchmark retraining is performed.", ""]
    for item in items:
        readme.append(f"- {item['experiment']}: {item['status']}")
    readme.extend(["", "These diagnostics support checkpoint-level behavioral and traceability claims only. They do not establish causal source explanations or multi-seed training stability."])
    (out / "README.md").write_text("\n".join(readme) + "\n")
    manifest = env_info("summarize")
    manifest["summary"] = items
    append_json(out / "run_manifest.json", manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True, choices=["audit", "source_suppression", "temperature", "efficiency", "correctability", "gate_no_magnitude", "bootstrap", "summarize", "all"])
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--settings", default="ETTm1_96,ETTh2_336")
    ap.add_argument("--max-batches", type=int, default=0)
    args = ap.parse_args()
    SETTINGS.update({setting: {field: Path(value) if field in ("cmd", "ckpt", "lin_cmd", "lin_ckpt", "raw", "no_mag_cmd", "no_mag_ckpt") else value
                               for field, value in spec.items()}
                     for setting, spec in json.loads(Path(args.config).read_text()).items()})
    setup_seed()
    out = init_out(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    settings = [s for s in args.settings.split(",") if s]
    log = out / "logs" / f"{args.task}.log"
    log.write_text(json.dumps({"task": args.task, "out": str(out), "device": str(device), "settings": settings, "started": datetime.now().isoformat()}, indent=2) + "\n")
    if args.task in ["audit", "all"]:
        audit(out)
    if args.task in ["source_suppression", "all"]:
        source_suppression(out, settings, device, args.max_batches)
    if args.task in ["temperature", "all"]:
        temperature(out, settings, device, args.max_batches)
    if args.task in ["efficiency", "all"]:
        efficiency(out, device)
    if args.task in ["correctability", "all"]:
        corr_settings = settings + (["ETTh1_720"] if "ETTh1_720" not in settings and SETTINGS.get("ETTh1_720", {}).get("ckpt", Path("missing_checkpoint")).exists() else [])
        correctability(out, corr_settings, device, args.max_batches)
    if args.task in ["gate_no_magnitude", "all"]:
        gate_no_magnitude(out, settings, device, args.max_batches)
    if args.task in ["bootstrap", "all"]:
        bootstrap(out, settings, device, args.max_batches)
    if args.task in ["summarize", "all"]:
        summarize(out)
    with log.open("a") as f:
        f.write(json.dumps({"finished": datetime.now().isoformat()}, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
