from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from dump_stom_interpretability_raw import parse_command_file

SETTINGS = {"ETTm1-96": 24, "ETTh2-336": 32, "ETTh1-720": 48, "Weather-720": 48}


def cli_args(config, setting, seed, width, out_dir):
    values = dict(config)
    values.update({
        "is_training": 1,
        "model": "STOM",
        "model_id": f"L_native_pm_{setting}_seed{seed}",
        "d_model": width,
        "random_seed": seed,
        "checkpoints": str(out_dir / "checkpoints"),
        "use_stom": 0,
        "save_test_predictions": 0,
        "save_test_visuals": 0,
        "skip_final_test_after_train": 0,
        "disable_train_test_eval": 1,
        "gpu": 0,
    })
    args = []
    for key, value in values.items():
        if key in {"use_gpu", "use_multi_gpu"}:
            continue
        if isinstance(value, bool):
            if key == "distil":
                if not value:
                    args.append("--distil")
            elif value:
                args.append(f"--{key}")
            continue
        if value is not None:
            args.extend((f"--{key}", str(value)))
    return args

def run_setting(setting, seed, gpu, command_file, out_dir):
    width = SETTINGS[setting]
    config = vars(parse_command_file(command_file))
    args = cli_args(config, setting, seed, width, out_dir)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WANDB_MODE="offline", WANDB_SILENT="true")
    command = [sys.executable, "-u", "run_longExp.py", *args]
    log = out_dir / "logs" / f"{setting}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        proc = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, stdout=fh, stderr=subprocess.STDOUT)
    setting_name = (
        f"L_native_pm_{setting}_seed{seed}_STOM_{config['data']}_ftM_sl96_ll0_"
        f"pl{config['pred_len']}_dm{width}_nh{config['n_heads']}_el{config['e_layers']}_"
        f"dl{config.get('d_layers', 1)}_df{config.get('d_ff', 2048)}_"
        f"fc{config.get('factor', 1)}_eb{config.get('embed', 'timeF')}_"
        f"dt{config.get('distil', True)}_{config.get('des', 'Exp')}_0"
    )
    metrics = Path(__file__).resolve().parents[1] / "results" / setting_name / "metrics.json"
    data = json.loads(metrics.read_text()) if metrics.exists() else {}
    return {"setting": setting, "seed": seed, "width": width,
            "returncode": proc.returncode, "MSE": data.get("MSE", ""),
            "MAE": data.get("MAE", "")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--seeds", default="2026,42,0")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--commands", required=True)
    ap.add_argument("--out-dir", default="capacity_matched_results")
    args = ap.parse_args()
    command_files = json.loads(Path(args.commands).read_text())
    out_dir = Path(args.out_dir)
    rows = [run_setting(setting, int(seed), args.gpu, Path(command_files[setting]), out_dir)
            for setting in args.settings.split(",") if setting
            for seed in args.seeds.split(",") if seed]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_seed_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", "seed", "width", "returncode", "MSE", "MAE"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
