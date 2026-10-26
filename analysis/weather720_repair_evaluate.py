"""Evaluate Weather-720 repair checkpoints and build paper-ready CSVs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

STOM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STOM_ROOT))
OUT_DIR = Path("weather_repair_results")
os.chdir(STOM_ROOT)
os.environ.setdefault('WANDB_MODE', 'offline')
os.environ.setdefault('WANDB_SILENT', 'true')
os.environ.setdefault('WANDB_DIR', str(OUT_DIR / 'wandb'))

from exp.exp_main import Exp_Main
from utils.metrics import metric
RESULT_DIR = OUT_DIR / "results"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

REFERENCE = {
    "Full_STOM_reference": {"mse": 0.3864375244, "mae": 0.3857067038},
    "Local_LinAttn_reference": {"mse": 0.381485, "mae": 0.382616},
}

EXPERIMENTS = {
    "global_without_phase": {
        "model_id": "2026_repair_weather720_global_without_phase_STOM_LinAttn_Weather_96_720",
        "repair_type": "global_without_phase",
        "stom_ablation": "without_phase",
        "stom_phase_repair_mode": "none",
        "phase_active_near": False,
        "phase_active_middle": False,
        "phase_active_far": False,
    },
    "regional_phase_repair": {
        "model_id": "2026_repair_weather720_regional_phase_repair_STOM_LinAttn_Weather_96_720",
        "repair_type": "regional_phase_repair",
        "stom_ablation": "full",
        "stom_phase_repair_mode": "regional_phase_repair",
        "phase_active_near": True,
        "phase_active_middle": False,
        "phase_active_far": False,
    },
    "Full_STOM_reference": {
        "model_id": "2026_stom_full_Weather_720_STOM_LinAttn_Weather_96_720",
        "repair_type": "reference_full_stom",
        "stom_ablation": "full",
        "stom_phase_repair_mode": "none",
        "checkpoint": Path("checkpoints/full_stom/checkpoint.pth"),
        "phase_active_near": True,
        "phase_active_middle": True,
        "phase_active_far": True,
    },
}


def base_args(spec: dict) -> SimpleNamespace:
    return SimpleNamespace(
        random_seed=2026, is_training=0, model_id=spec["model_id"], model="STOM", model_type="DL",
        data="custom", root_path="./dataset/", data_path="weather/weather.csv", features="M", target="OT",
        freq="h", checkpoints=str(CHECKPOINT_DIR), scale=1, train_ratio=0.7, test_ratio=0.2,
        plot_every=5, iterative_prediction=0, number_of_targets=0, seq_len=96, label_len=0,
        pred_len=720, fc_dropout=0.05, head_dropout=0.0, patch_len=16, stride=8,
        padding_patch="end", revin=1, affine=0, subtract_last=0, decomposition=0, kernel_size=25,
        individual=0, embed_type=0, enc_in=21, dec_in=7, c_out=7, d_model=0, n_heads=4,
        e_layers=3, d_layers=1, d_ff=2048, moving_avg=25, factor=1, distil=True,
        dropout=0.0, embed="timeF", activation="gelu", output_attention=False, do_predict=False,
        time_emb_dim=4, mlp_ratio=8, num_workers=6, itr=1, train_epochs=100, batch_size=32,
        patience=12, learning_rate=0.0008, des="Exp", loss="mse", lradj="TST", pct_start=0.0,
        use_amp=True, max_grad_norm=1.0, test_every=0, gradient_accumulation=1, batch_size_test=0,
        predictor="LinAttn", resume="none", compile=0, save_test_predictions=1, save_test_visuals=0,
        disable_train_test_eval=0, skip_final_test_after_train=0, use_gpu=torch.cuda.is_available(), gpu=0,
        use_multi_gpu=False, devices="0,1,2,3", test_flop=False, random_drop_training=0,
        model_label="Model", univariate=0, channel_independence=1, decomp_method="dft_decomp",
        use_norm=1, down_sampling_layers=2, down_sampling_window=2, down_sampling_method="avg",
        channel_mixing=1, percent=100, seasonal_patterns="Monthly", inverse=False,
        task_name="long_term_forecast", top_k=3, num_kernels=6, use_future_temporal_feature=0,
        random_channel_dropout=1, weight_decay=0.0, specmix_training_curve_path="",
        specmix_tensorboard_dir="", specmix_diagnostics_path="", specmix_gate_reg=0.0,
        specmix_gate_rho=0.2, specmix_gate_reg_weight=1e-4, use_stom=1, stom_ma_kernel=25,
        stom_local_kernel=5, stom_spec_gain_init=6.0, stom_gate_prior="0.35,0.10,0.30,0.25",
        stom_gate_hidden_dim=32, stom_rank=4, stom_n_spec_heads=4, stom_precision_variant="none",
        stom_ablation=spec["stom_ablation"], stom_operator_set="phase,spectral,smooth,local",
        stom_phase_repair_mode=spec["stom_phase_repair_mode"], stom_phase_repair_boundary=240,
        stom_residual_mode="none", stom_residual_alpha_cap=0.10, stom_residual_alpha_init=0.01,
        stom_residual_dict_k=8, stom_spec_injection_mode="none", stom_spec_injection_beta_cap=0.10,
        stom_spec_injection_beta_init=0.01,
    )


def setting_name(spec: dict) -> str:
    return f"{spec['model_id']}_STOM_custom_ftM_sl96_ll0_pl720_dm0_nh4_el3_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0"


def parse_best(log_file: Path) -> tuple[str, str]:
    if not log_file.exists():
        return "", ""
    best_epoch = ""
    best_val = ""
    current_epoch = ""
    for line in log_file.read_text(errors="ignore").splitlines():
        m_epoch = re.search(r"Starting Training Epoch: (\d+)", line)
        if m_epoch:
            current_epoch = str(int(m_epoch.group(1)))
        m_val = re.search(r"Validation loss decreased .* --> ([0-9.eE+-]+)\)", line)
        if m_val:
            best_epoch = current_epoch
            best_val = m_val.group(1)
    return best_epoch, best_val


def load_state(model, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if unexpected:
        print(f"warning unexpected keys for {checkpoint}: {unexpected[:5]}")


def evaluate_checkpoint(name: str, spec: dict, gpu: int) -> dict:
    args = base_args(spec)
    args.gpu = gpu
    args.use_gpu = torch.cuda.is_available()
    exp = Exp_Main(args)
    setting = setting_name(spec)
    checkpoint = Path(spec.get("checkpoint", CHECKPOINT_DIR / setting / "checkpoint.pth"))
    load_state(exp.model, checkpoint)
    data, loader = exp._get_data(flag="test")
    exp.model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].float().to(exp.device)
            batch_y = batch[1].float().to(exp.device)
            batch_x_mark = batch[2].float().to(exp.device)
            batch_y_mark = batch[3].float().to(exp.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
            out = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            out = out[:, -args.pred_len:, -args.enc_in:]
            true = batch_y[:, -args.pred_len:, -args.enc_in:]
            preds.append(out.float().cpu().numpy())
            trues.append(true.float().cpu().numpy())
    pred = np.concatenate(preds, axis=0).reshape(-1, args.pred_len, args.enc_in)
    true = np.concatenate(trues, axis=0).reshape(-1, args.pred_len, args.enc_in)
    mae, mse, rmse, mape, mspe, rse, corr = metric(pred, true)
    diff = pred - true
    pos_mse = np.mean(np.square(diff), axis=(0, 2))
    pos_mae = np.mean(np.abs(diff), axis=(0, 2))
    def region(start, end):
        d = diff[:, start:end, :]
        return float(np.mean(np.square(d))), float(np.mean(np.abs(d)))
    near_mse, near_mae = region(0, 240)
    mid_mse, mid_mae = region(240, 480)
    far_mse, far_mae = region(480, 720)
    params = sum(p.numel() for p in exp.model.parameters())
    log_file = OUT_DIR / "logs" / f"{spec['repair_type']}.log"
    cmd_file = OUT_DIR / "commands" / f"weather720_{spec['repair_type']}.cmd"
    if name == "Full_STOM_reference":
        log_file = OUT_DIR / "logs" / "stom_full_Weather_720.log"
        cmd_file = OUT_DIR / "commands" / "stom_full_Weather_720.cmd"
    best_epoch, best_val = parse_best(log_file)
    row = {
        "model": name, "seed": 2026, "dataset": "Weather", "pred_len": 720,
        "repair_type": spec["repair_type"], "phase_active_near": spec["phase_active_near"],
        "phase_active_middle": spec["phase_active_middle"], "phase_active_far": spec["phase_active_far"],
        "best_epoch": best_epoch, "best_val_loss": best_val,
        "test_mse": float(np.mean(mse)), "test_mae": float(np.mean(mae)),
        "near_mse": near_mse, "near_mae": near_mae, "middle_mse": mid_mse, "middle_mae": mid_mae,
        "far_mse": far_mse, "far_mae": far_mae, "params": params,
        "checkpoint": str(checkpoint), "log_file": str(log_file), "command_file": str(cmd_file),
        "notes": "best-validation checkpoint inference; retrospective Weather-720 within-case repair" if name != "Full_STOM_reference" else "provided Full STOM reference checkpoint inference",
    }
    return {"row": row, "position_mse": pos_mse, "position_mae": pos_mae}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default="weather_repair_results")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--models", nargs="*", default=["Full_STOM_reference", "global_without_phase", "regional_phase_repair"])
    args = ap.parse_args()
    global OUT_DIR, RESULT_DIR, CHECKPOINT_DIR
    OUT_DIR = Path(args.out_dir)
    RESULT_DIR = OUT_DIR / "results"
    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    EXPERIMENTS["Full_STOM_reference"]["checkpoint"] = CHECKPOINT_DIR / "full_stom" / "checkpoint.pth"
    rows = []
    pos = {"forecast_position": np.arange(720)}
    for ref_name, ref in REFERENCE.items():
        if ref_name == "Local_LinAttn_reference":
            rows.append({
                "model": ref_name, "seed": 2026, "dataset": "Weather", "pred_len": 720,
                "repair_type": "reference_linattn", "phase_active_near": "NA", "phase_active_middle": "NA",
                "phase_active_far": "NA", "best_epoch": "", "best_val_loss": "",
                "test_mse": ref["mse"], "test_mae": ref["mae"], "near_mse": "", "near_mae": "",
                "middle_mse": "", "middle_mae": "", "far_mse": "", "far_mae": "", "params": "",
                "checkpoint": "", "log_file": "", "command_file": "", "notes": "local LinAttn reference from existing result table",
            })
    for name in args.models:
        spec = EXPERIMENTS[name]
        out = evaluate_checkpoint(name, spec, args.gpu)
        rows.append(out["row"])
        pos[f"{name}_mse"] = out["position_mse"]
        pos[f"{name}_mae"] = out["position_mae"]
    write_csv(RESULT_DIR / "weather720_repair_summary.csv", rows)
    pos_rows = []
    for i in range(720):
        row = {k: (int(v[i]) if k == "forecast_position" else float(v[i])) for k, v in pos.items()}
        pos_rows.append(row)
    write_csv(RESULT_DIR / "positionwise_mse.csv", pos_rows)
    report = ["# Weather-720 Repair Results\n\n"]
    report.append("This is a retrospective, test-informed, within-case repair study. The Weather-720 test targets were used to select the repair structure. The repaired scores are not unbiased estimates of repair generalization.\n\n")
    report.append("| model | MSE | MAE | near MSE | middle MSE | far MSE |\n| --- | ---: | ---: | ---: | ---: | ---: |\n")
    for r in rows:
        report.append(f"| {r['model']} | {r['test_mse']} | {r['test_mae']} | {r['near_mse']} | {r['middle_mse']} | {r['far_mse']} |\n")
    RESULT_DIR.joinpath("weather720_repair_report.md").write_text("".join(report))
    print(RESULT_DIR / "weather720_repair_summary.csv")
    print(RESULT_DIR / "positionwise_mse.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())