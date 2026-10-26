from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.data_factory import data_provider  
from final_diagnostics import (  
    denorm,
    extract_xhist_yphase,
    load_model,
    model_batch,
    setup_seed,
    stom_parts,
    target_from_batch,
)

OPS = ("phase", "spectral", "smooth", "local")
SHRINK_GRID = np.linspace(0.0, 1.0, 21)
SEEDS = (2026, 42, 0, 1, 2)

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def metric_update(acc: dict, pred: torch.Tensor, true: torch.Tensor) -> None:
    d = (pred.float() - true.float()).detach()
    acc["sq"] += float(d.square().sum().item())
    acc["abs"] += float(d.abs().sum().item())
    acc["n"] += int(d.numel())

def metric_finish(acc: dict) -> tuple[float, float]:
    n = max(1, acc["n"])
    return acc["sq"] / n, acc["abs"] / n

def checkpoint_inventory(config_path):
    inventory = json.loads(Path(config_path).read_text())
    return {setting: {int(seed): (Path(item["command"]), Path(item["checkpoint"]))
                      for seed, item in seeds.items()}
            for setting, seeds in inventory.items()}


def operator_effects(args, model, data, device):
    bx, by, bxm, bym, _ = model_batch(args, data, device)
    xh, yp, mean, std = extract_xhist_yphase(model, bx, bxm, bym)
    parts = stom_parts(model, xh, yp, force_fp32=True)
    order = [str(x) for x in parts["operator_set"]]
    perm = [order.index(op) for op in OPS]
    stack = parts["expert_stack"].float()[..., perm]
    gate = parts["gate"].float()[..., perm]
    full_norm = parts["y_norm"].float()
    full_pred = denorm(model, full_norm, mean, std).float()
    true = target_from_batch(args, by, model).float()
    effects = []
    for k in range(len(OPS)):
        ablated = full_norm - gate[..., k] * stack[..., k]
        effects.append(full_pred - denorm(model, ablated, mean, std).float())
    return full_pred, true, torch.stack(effects, dim=-1)

def collect_validation(args, model, device) -> dict:
    _, loader = data_provider(args, "val")
    n_ops = len(OPS)
    c = 0.0
    b = np.zeros(n_ops, dtype=np.float64)
    gram = np.zeros((n_ops, n_ops), dtype=np.float64)
    full = {"sq": 0.0, "abs": 0.0, "n": 0}
    with torch.inference_mode():
        for data in loader:
            pred, true, effects = operator_effects(args, model, data, device)
            e = (pred - true).detach().float().cpu().numpy().reshape(-1).astype(np.float64)
            z = effects.detach().float().cpu().numpy().reshape(-1, n_ops).astype(np.float64)
            c += float(e @ e)
            b += z.T @ e
            gram += z.T @ z
            metric_update(full, pred, true)

    def quadratic(lam: np.ndarray) -> float:
        return float(c - 2.0 * lam @ b + lam @ gram @ lam)

    removal_losses = c - 2.0 * b + np.diag(gram)
    removal_i = int(np.argmin(removal_losses))
    shrink_scores = np.zeros((n_ops, len(SHRINK_GRID)), dtype=np.float64)
    for k in range(n_ops):
        shrink_scores[k] = c - 2.0 * SHRINK_GRID * b[k] + SHRINK_GRID**2 * gram[k, k]
    shrink_i, shrink_grid_i = np.unravel_index(np.argmin(shrink_scores), shrink_scores.shape)
    shrink_s = float(SHRINK_GRID[shrink_grid_i])
    joint = minimize(
        quadratic,
        x0=np.zeros(n_ops, dtype=np.float64),
        bounds=[(0.0, 1.0)] * n_ops,
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000},
    )
    joint_lam = np.clip(joint.x, 0.0, 1.0)
    full_mse, full_mae = metric_finish(full)
    return {
        "full_val_mse": full_mse,
        "full_val_mae": full_mae,
        "selected_removal": OPS[removal_i],
        "removal_val_mse": float(removal_losses[removal_i] / max(1, full["n"])),
        "selected_shrink_operator": OPS[int(shrink_i)],
        "selected_shrink_s": shrink_s,
        "shrink_val_mse": float(shrink_scores[shrink_i, shrink_grid_i] / max(1, full["n"])),
        "selected_joint_lambda": joint_lam.tolist(),
        "joint_val_mse": float(quadratic(joint_lam) / max(1, full["n"])),
        "selection_objective": "validation_mse",
        "shrink_grid": SHRINK_GRID.tolist(),
        "joint_optimizer_success": bool(joint.success),
        "joint_optimizer_message": str(joint.message),
    }

def evaluate_test(args, model, device, selection: dict) -> dict:
    _, loader = data_provider(args, "test")
    metrics = {
        "full": {"sq": 0.0, "abs": 0.0, "n": 0},
        "removal": {"sq": 0.0, "abs": 0.0, "n": 0},
        "shrink": {"sq": 0.0, "abs": 0.0, "n": 0},
        "joint": {"sq": 0.0, "abs": 0.0, "n": 0},
    }
    removal_i = OPS.index(selection["selected_removal"])
    shrink_i = OPS.index(selection["selected_shrink_operator"])
    shrink_s = float(selection["selected_shrink_s"])
    joint_lam = torch.tensor(selection["selected_joint_lambda"], device=device).view(1, 1, 1, -1)
    with torch.inference_mode():
        for data in loader:
            pred, true, effects = operator_effects(args, model, data, device)
            removal = pred - effects[..., removal_i]
            shrink = pred - shrink_s * effects[..., shrink_i]
            joint = pred - (effects * joint_lam).sum(dim=-1)
            metric_update(metrics["full"], pred, true)
            metric_update(metrics["removal"], removal, true)
            metric_update(metrics["shrink"], shrink, true)
            metric_update(metrics["joint"], joint, true)
    out = {}
    for key, acc in metrics.items():
        out[f"{key}_test_mse"], out[f"{key}_test_mae"] = metric_finish(acc)
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out-dir", default="validation_intervention_results")
    ap.add_argument("--models", required=True)
    args_cli = ap.parse_args()
    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")
    inventory = checkpoint_inventory(args_cli.models)
    out = Path(args_cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selections: dict[str, dict[str, dict]] = {}

    
    for setting, seed_map in inventory.items():
        selections[setting] = {}
        for seed in SEEDS:
            command, checkpoint = seed_map[seed]
            if not command.exists() or not checkpoint.exists():
                raise FileNotFoundError(f"{setting} seed={seed}: {command} / {checkpoint}")
            print(f"validation {setting} seed={seed}", flush=True)
            args, model = load_model(command, checkpoint, device, use_stom=1)
            args.random_seed = seed
            setup_seed()
            selection = collect_validation(args, model, device)
            selections[setting][str(seed)] = {
                "dataset": "Weather" if setting.startswith("Weather") else "ETTh1",
                "horizon": 720,
                "seed": seed,
                "command_file": str(command.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                **selection,
            }
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    (out / "selection_details.json").write_text(json.dumps(selections, indent=2))

    
    rows = []
    for setting, seed_map in inventory.items():
        for seed in SEEDS:
            sel = selections[setting][str(seed)]
            command, checkpoint = seed_map[seed]
            print(f"test {setting} seed={seed}", flush=True)
            args, model = load_model(command, checkpoint, device, use_stom=1)
            args.random_seed = seed
            setup_seed()
            test = evaluate_test(args, model, device, sel)
            row = {
                "dataset": sel["dataset"],
                "horizon": 720,
                "seed": seed,
                "full_val_mse": sel["full_val_mse"],
                "full_val_mae": sel["full_val_mae"],
                "selected_removal_operator": sel["selected_removal"],
                "removal_val_mse": sel["removal_val_mse"],
                "removal_test_mse": test["removal_test_mse"],
                "selected_shrink_operator": sel["selected_shrink_operator"],
                "selected_shrink_s": sel["selected_shrink_s"],
                "shrink_val_mse": sel["shrink_val_mse"],
                "shrink_test_mse": test["shrink_test_mse"],
                "joint_lambda_phase": sel["selected_joint_lambda"][0],
                "joint_lambda_spectral": sel["selected_joint_lambda"][1],
                "joint_lambda_smooth": sel["selected_joint_lambda"][2],
                "joint_lambda_local": sel["selected_joint_lambda"][3],
                "joint_val_mse": sel["joint_val_mse"],
                "joint_test_mse": test["joint_test_mse"],
                "full_test_mse": test["full_test_mse"],
                "full_test_mae": test["full_test_mae"],
                "removal_test_mae": test["removal_test_mae"],
                "shrink_test_mae": test["shrink_test_mae"],
                "joint_test_mae": test["joint_test_mae"],
                "command_file": str(command.resolve()),
                "checkpoint": str(checkpoint.resolve()),
            }
            rows.append(row)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(out / "per_seed_results.csv", rows)
    summary = []
    for dataset in ("ETTh1", "Weather"):
        rr = [r for r in rows if r["dataset"] == dataset]
        for metric in (
            "full_val_mse",
            "removal_val_mse",
            "shrink_val_mse",
            "joint_val_mse",
            "full_test_mse",
            "removal_test_mse",
            "shrink_test_mse",
            "joint_test_mse",
            "full_test_mae",
            "removal_test_mae",
            "shrink_test_mae",
            "joint_test_mae",
        ):
            vals = np.asarray([float(r[metric]) for r in rr], dtype=np.float64)
            summary.append({
                "dataset": dataset,
                "metric": metric,
                "num_seeds": len(vals),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
            })
        for policy, metric in (
            ("removal", "removal_test_mse"),
            ("shrink", "shrink_test_mse"),
            ("joint", "joint_test_mse"),
        ):
            base = np.asarray([float(r["full_test_mse"]) for r in rr])
            cand = np.asarray([float(r[metric]) for r in rr])
            imp = base - cand
            summary.append({
                "dataset": dataset,
                "metric": f"{policy}_test_improvement",
                "num_seeds": len(imp),
                "mean": float(imp.mean()),
                "std": float(imp.std(ddof=1)),
                "improved_seeds": int((imp > 0).sum()),
                "improvement_rate": float((imp > 0).mean()),
                "mean_relative_improvement": float(np.mean(imp / base)),
            })
    write_csv(out / "mean_std_results.csv", summary)
    print(out)

if __name__ == "__main__":
    main()
