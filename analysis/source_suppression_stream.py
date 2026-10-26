import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

STOM = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(STOM), str(STOM / "scripts_local"), str(Path(__file__).resolve().parent)]

from data_provider.data_factory import data_provider
from final_diagnostics import (
    SEED, SETTINGS, EPS, exact_cspec, extract_xhist_yphase, gate_from_outputs,
    init_out, load_model, model_batch, setup_seed, stom_parts, target_from_batch,
    denorm, env_info, write_csv, append_json, pearson, spearman,
)


def init_acc():
    return defaultdict(float) | {"n_batches": 0, "n_values": 0, "max_abs_error": 0.0, "pearson_x": [], "pearson_y": [], "spearman_x": [], "spearman_y": []}


def update_acc(acc, delta, traced, pred, masked_pred, true, gate_l1=0.0, spec_gate_change=0.0, consistency=True):
    d = (delta - traced).detach().float().cpu().numpy().reshape(-1)
    da = delta.detach().float().cpu().numpy().reshape(-1)
    tr = traced.detach().float().cpu().numpy().reshape(-1)
    acc["n_batches"] += 1
    acc["n_values"] += d.size
    if consistency:
        acc["abs_error"] += float(np.abs(d).sum())
        acc["sq_error"] += float((d * d).sum())
        acc["max_abs_error"] = max(acc["max_abs_error"], float(np.abs(d).max()))
    acc["sign_agree"] += float((np.sign(da) == np.sign(tr)).sum())
    acc["mean_abs_prediction_change_sum"] += float(np.abs(da).sum())
    before = (pred - true).detach().float().cpu().numpy().reshape(-1)
    after = (masked_pred - true).detach().float().cpu().numpy().reshape(-1)
    acc["mse_before_sum"] += float((before * before).sum())
    acc["mse_after_sum"] += float((after * after).sum())
    acc["mae_before_sum"] += float(np.abs(before).sum())
    acc["mae_after_sum"] += float(np.abs(after).sum())
    acc["metric_n"] += before.size
    acc["gate_l1_sum"] += float(gate_l1)
    acc["spec_gate_change_sum"] += float(spec_gate_change)
    keep = min(20000, da.size)
    acc["pearson_x"].append(da[:keep])
    acc["pearson_y"].append(tr[:keep])
    acc["spearman_x"].append(da[:keep])
    acc["spearman_y"].append(tr[:keep])


def acc_row(key, acc):
    setting, protocol, mask_type, k = key
    n = max(1, acc["n_values"])
    m = max(1, acc["metric_n"])
    x = np.concatenate(acc["pearson_x"]) if acc["pearson_x"] else np.array([])
    y = np.concatenate(acc["pearson_y"]) if acc["pearson_y"] else np.array([])
    row = {
        "setting": setting,
        "protocol": protocol,
        "mask_type": mask_type,
        "k": k,
        "num_batches": int(acc["n_batches"]),
        "num_values": int(acc["n_values"]),
        "consistency_mae": acc["abs_error"] / n if protocol == "fixed_gate" else "N/A",
        "consistency_rmse": (acc["sq_error"] / n) ** 0.5 if protocol == "fixed_gate" else "N/A",
        "max_abs_error": acc["max_abs_error"] if protocol == "fixed_gate" else "N/A",
        "pearson": pearson(x, y),
        "spearman": spearman(x, y),
        "sign_agreement": acc["sign_agree"] / n,
        "mean_abs_prediction_change": acc["mean_abs_prediction_change_sum"] / n,
        "mse_before": acc["mse_before_sum"] / m,
        "mse_after": acc["mse_after_sum"] / m,
        "mse_delta": (acc["mse_after_sum"] - acc["mse_before_sum"]) / m,
        "mae_before": acc["mae_before_sum"] / m,
        "mae_after": acc["mae_after_sum"] / m,
        "mae_delta": (acc["mae_after_sum"] - acc["mae_before_sum"]) / m,
        "mean_gate_l1_change": acc["gate_l1_sum"] / max(1, acc["n_batches"]),
        "mean_spec_gate_change": acc["spec_gate_change_sum"] / max(1, acc["n_batches"]),
    }
    return row


def run(out, settings, device, max_batches=0):
    out = init_out(out)
    setup_seed()
    meta_path = out / "source_suppression" / "source_suppression_per_sample.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_id", "setting", "sample_id", "variable_id", "k", "mask_type", "random_repeat_id", "delta_actual_file", "delta_traced_file", "delta_recomputed_file", "selected_indices_file", "selected_indices_row"]
    accs = defaultdict(init_acc)
    row_id_global = 0
    manifest = {"settings": settings, "files": {}, "seed": SEED, "started": datetime.now().isoformat()}
    with meta_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        for setting in settings:
            cfg = SETTINGS[setting]
            args, model = load_model(cfg["cmd"], cfg["ckpt"], device, use_stom=1)
            _, loader = data_provider(args, "test")
            n_samples = len(loader.dataset) if not max_batches else min(max_batches * args.batch_size, len(loader.dataset))
            C, P, L = model.number_of_targets, model.pred_len, model.seq_len
            rows_setting = n_samples * C * 33
            base = out / "source_suppression"
            actual_path = base / f"{setting}_delta_actual.npy"
            traced_path = base / f"{setting}_delta_traced.npy"
            recomp_path = base / f"{setting}_delta_recomputed.npy"
            selected_path = base / f"{setting}_selected_indices.npy"
            actual_mm = np.lib.format.open_memmap(actual_path, mode="w+", dtype=np.float16, shape=(rows_setting, P))
            traced_mm = np.lib.format.open_memmap(traced_path, mode="w+", dtype=np.float16, shape=(rows_setting, P))
            recomp_mm = np.lib.format.open_memmap(recomp_path, mode="w+", dtype=np.float16, shape=(rows_setting, P))
            selected_mm = np.lib.format.open_memmap(selected_path, mode="w+", dtype=np.int16, shape=(rows_setting, 10))
            selected_mm[:] = -1
            rng = np.random.default_rng(SEED)
            row_local = 0
            sample_base = 0
            with torch.inference_mode():
                for bidx, data in enumerate(loader):
                    if max_batches and bidx >= max_batches:
                        break
                    bx, by, bxm, bym, dec = model_batch(args, data, device)
                    xh, yp, mean, std = extract_xhist_yphase(model, bx, bxm, bym)
                    parts = stom_parts(model, xh, yp, force_fp32=True)
                    pred_norm = parts["y_norm"].float()
                    pred = denorm(model, pred_norm, mean, std)
                    true = target_from_batch(args, by, model)
                    agg, cspec = exact_cspec(parts)
                    op_index = {name: i for i, name in enumerate(parts["operator_set"])}
                    spec_i = op_index["spectral"]
                    z_spec = parts["gate"][..., spec_i].float() * parts["y_spec"].float()
                    score = cspec.abs().mean(dim=2)
                    B = bx.shape[0]
                    for k in [1, 5, 10]:
                        kk = min(k, L)
                        top_idx = torch.topk(score, kk, dim=-1).indices
                        masks = [("topk", 0, top_idx)]
                        top_np = top_idx.detach().cpu().numpy()
                        for rep in range(10):
                            rand = np.empty((B, C, kk), dtype=np.int64)
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
                            fixed_norm = pred_norm - z_spec + z_spec_masked
                            delta_actual = pred_norm - fixed_norm
                            fixed_pred = denorm(model, fixed_norm, mean, std)
                            rec_logits, rec_gate, _ = gate_from_outputs(model, xh.float(), yp.float(), y_spec_masked, parts["y_smooth"].float(), parts["y_local"].float())
                            rec_stack = torch.stack([{"phase": parts["y_phase"].float(), "spectral": y_spec_masked, "smooth": parts["y_smooth"].float(), "local": parts["y_local"].float()}[name] for name in parts["operator_set"]], dim=-1)
                            rec_norm = (rec_gate.float() * rec_stack).sum(-1)
                            delta_rec = pred_norm - rec_norm
                            rec_pred = denorm(model, rec_norm, mean, std)
                            update_acc(accs[(setting, "fixed_gate", mask_type, k)], delta_actual, traced, pred, fixed_pred, true, consistency=True)
                            update_acc(accs[(setting, "recomputed_gate", mask_type, k)], delta_rec, traced, pred, rec_pred, true, gate_l1=float((rec_gate.float() - parts["gate"].float()).abs().mean().item()), spec_gate_change=float((rec_gate[..., spec_i].float() - parts["gate"][..., spec_i].float()).mean().item()), consistency=False)
                            idx_np = idx.detach().cpu().numpy()
                            da_np = delta_actual.detach().cpu().numpy().astype(np.float16)
                            dt_np = traced.detach().cpu().numpy().astype(np.float16)
                            dr_np = delta_rec.detach().cpu().numpy().astype(np.float16)
                            for bi in range(B):
                                for ci in range(C):
                                    actual_mm[row_local] = da_np[bi, ci]
                                    traced_mm[row_local] = dt_np[bi, ci]
                                    recomp_mm[row_local] = dr_np[bi, ci]
                                    selected_mm[row_local, :kk] = idx_np[bi, ci]
                                    writer.writerow({"row_id": row_id_global, "setting": setting, "sample_id": sample_base + bi, "variable_id": ci, "k": k, "mask_type": mask_type, "random_repeat_id": rep, "delta_actual_file": str(actual_path), "delta_traced_file": str(traced_path), "delta_recomputed_file": str(recomp_path), "selected_indices_file": str(selected_path), "selected_indices_row": row_local})
                                    row_id_global += 1
                                    row_local += 1
                    sample_base += B
            actual_mm.flush(); traced_mm.flush(); recomp_mm.flush(); selected_mm.flush()
            manifest["files"][setting] = {"rows_expected": rows_setting, "rows_written": row_local, "delta_actual": str(actual_path), "delta_traced": str(traced_path), "delta_recomputed": str(recomp_path), "selected_indices": str(selected_path)}
    summary = [acc_row(key, acc) for key, acc in sorted(accs.items())]
    write_csv(out / "source_suppression" / "source_suppression_summary.csv", summary)
    append_json(out / "source_suppression" / "config.json", env_info("source_suppression_stream") | manifest)
    passed = bool(summary) and all((r["protocol"] != "fixed_gate") or float(r["consistency_mae"]) < 1e-5 and float(r["max_abs_error"]) < 1e-5 for r in summary)
    append_json(out / "source_suppression" / "validation_report.json", {"passed": passed, "rows": row_id_global, "summary_rows": len(summary), "placeholder": False, "streamed_vectors": True})
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--settings", default="ETTm1_96,ETTh2_336")
    ap.add_argument("--max-batches", type=int, default=0)
    args = ap.parse_args()
    SETTINGS.update({setting: {field: Path(value) if field in ("cmd", "ckpt", "lin_cmd", "lin_ckpt", "raw", "no_mag_cmd", "no_mag_ckpt") else value
                               for field, value in spec.items()}
                     for setting, spec in json.loads(Path(args.config).read_text()).items()})
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    run(args.out_dir, [s for s in args.settings.split(",") if s], device, args.max_batches)