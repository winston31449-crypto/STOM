from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch

STOM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STOM_DIR))
sys.path.insert(0, str(STOM_DIR / "scripts_local"))

from data_provider.data_factory import data_provider
from dump_stom_interpretability_raw import parse_command_file, unwrap_state_dict
from models import stom as stom_model

FIELDS = ["seed", "model", "mse", "mae", "checkpoint", "selected_by", "notes"]


def setting_from_args(args):
    return "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
        args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len,
        args.pred_len, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
        args.factor, args.embed, args.distil, args.des, 0,
    )


def prepare_args(command_file: Path):
    args = parse_command_file(command_file)
    if args.root_path and not os.path.isabs(args.root_path):
        args.root_path = str((STOM_DIR / args.root_path).resolve())
    if getattr(args, "checkpoints", "./checkpoints/") and not os.path.isabs(str(args.checkpoints)):
        args.checkpoints = str((STOM_DIR / args.checkpoints).resolve())
    args.num_workers = 0
    args.batch_size_test = 0
    args.compile = 0
    if not getattr(args, "number_of_targets", 0):
        args.number_of_targets = args.enc_in
    if not hasattr(args, "stom_operator_set"):
        args.stom_operator_set = "phase,spectral,smooth,local"
    return args


def checkpoint_from_args(args):
    return Path(args.checkpoints) / setting_from_args(args) / "checkpoint.pth"


def append_row(path: Path, row):
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--command-file", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--outcsv", default="retrained_repair_results.csv")
    ap.add_argument("--decision", required=True)
    ap.add_argument("--selected-by", default="validation_loss_checkpoint_after_frozen_selection")
    ap.add_argument("--notes", default="final test once")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--allow-cpu", action="store_true")
    args_cli = ap.parse_args()

    decision = Path(args_cli.decision)
    if not decision.exists():
        raise RuntimeError(f"selection decision is not frozen yet: {decision}")
    frozen = json.loads(decision.read_text()).get("decision_frozen_before_test")
    if frozen is not True:
        raise RuntimeError("selection_decision.json is not marked frozen before test")

    args = prepare_args(Path(args_cli.command_file).resolve())
    checkpoint = Path(args_cli.checkpoint).resolve() if args_cli.checkpoint else checkpoint_from_args(args)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if args_cli.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args_cli.device)
    if device.type == "cpu" and not args_cli.allow_cpu:
        raise RuntimeError("CPU final test requested implicitly; pass --allow-cpu to acknowledge it")
    args.use_gpu = device.type == "cuda"
    args.gpu = 0

    model = stom_model.Model(args).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(unwrap_state_dict(state), strict=False)
    model.eval()
    _, loader = data_provider(args, "test")
    sq = 0.0
    ab = 0.0
    n = 0
    with torch.no_grad():
        for data in loader:
            batch_x = data[0].float().to(device)
            batch_y = data[1].float().to(device)
            batch_x_mark = data[2].float().to(device)
            batch_y_mark = data[3].float().to(device)
            pred_len = args.pred_len
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -args.number_of_targets
            if args.features == "MS":
                f_dim = -1
            pred = outputs[:, -pred_len:, f_dim:].float()
            true = batch_y[:, -pred_len:, f_dim:].float()
            diff = pred - true
            sq += float(diff.pow(2).sum().item())
            ab += float(diff.abs().sum().item())
            n += int(diff.numel())
    mse = sq / max(n, 1)
    mae = ab / max(n, 1)
    row = {
        "seed": args_cli.seed,
        "model": args_cli.model_name,
        "mse": f"{mse:.10f}",
        "mae": f"{mae:.10f}",
        "checkpoint": str(checkpoint),
        "selected_by": args_cli.selected_by,
        "notes": args_cli.notes,
    }
    append_row(Path(args_cli.outcsv), row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()