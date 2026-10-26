import argparse
import csv
import json
import math
import os
import shlex
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

STOM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STOM_DIR))

from data_provider.data_factory import data_provider
from models import stom as stom_model


DEFAULTS = dict(
    random_seed=2026,
    model_id="interpretability_dump",
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
INT_KEYS = {"random_seed", "is_training", "scale", "plot_every", "iterative_prediction", "number_of_targets", "seq_len", "label_len", "pred_len", "enc_in", "dec_in", "c_out", "d_model", "n_heads", "e_layers", "d_layers", "d_ff", "factor", "time_emb_dim", "num_workers", "batch_size", "batch_size_test", "compile", "gpu", "random_channel_dropout", "use_stom", "stom_ma_kernel", "stom_local_kernel", "stom_gate_hidden_dim", "stom_rank", "stom_n_spec_heads"}
FLOAT_KEYS = {"train_ratio", "test_ratio", "dropout", "weight_decay", "stom_spec_gain_init", "stom_residual_alpha_cap", "stom_residual_alpha_init", "stom_spec_injection_beta_cap", "stom_spec_injection_beta_init"}


def parse_command_file(path: Path) -> SimpleNamespace:
    args = dict(DEFAULTS)
    text = path.read_text()
    cmd_parts = []
    collecting = False
    for line in text.splitlines():
        stripped = line.strip()
        if not collecting:
            if "run_longExp.py" in stripped and "python" in stripped:
                collecting = True
            else:
                continue
        if not stripped or stripped.startswith(">"):
            break
        continued = stripped.endswith("\\")
        if continued:
            stripped = stripped[:-1].strip()
        if " > " in stripped:
            stripped = stripped.split(" > ", 1)[0].strip()
            continued = False
        cmd_parts.append(stripped)
        if not continued:
            break
    py_line = " ".join(cmd_parts)
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
    args["use_gpu"] = bool(torch.cuda.is_available())
    args["num_workers"] = 0
    args["batch_size_test"] = 0
    return SimpleNamespace(**args)


def unwrap_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def forward_stom_raw(model, batch_x, batch_x_mark, dec_inp, batch_y_mark, use_amp=False):
    del dec_inp
    m = model
    x = batch_x.permute(0, 2, 1)
    mean = x[:, :, [-1]]
    std = torch.ones(x.size(0), x.size(1), 1, device=x.device)
    x = (x - mean) / std
    x_hist_norm_targets = x[:, -m.number_of_targets:, :].clone()

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
    x = torch.einsum('bcli,cid->bcld', x, m.value_token_mixer)
    x = torch.cat([channel_token, x], dim=-1)

    B, C, N, _ = x.shape
    pos = torch.arange(0, N, dtype=torch.long, device=x.device)
    x = x + m.channel_emb_2 + m.pos_emb(pos)
    x = x.reshape(B * C, N, -1)
    x = m.temporal_processor(x)
    x = x[:, -m.pred_len:].reshape(B, C, m.pred_len, -1)
    y_phase = torch.einsum('bcld,cdi->bcli', x[:, :, :, -m.seq_emb_dim:], m.value_token_mixer.transpose(-2, -1)).squeeze(-1)

    stom = m.stom_head
    smooth = stom.moving_average(x_hist_norm_targets)
    residual = x_hist_norm_targets - smooth
    y_smooth = stom.trend_proj(smooth)
    local_feat = F.gelu(stom.local_conv(residual))
    y_local = stom.local_proj(local_feat)
    a, phi, delta, alpha, phase_strength, recent_strength, sigma_phase, sigma_recent = stom._spectral_kernel(x_hist_norm_targets.device, x_hist_norm_targets.dtype)
    y_spec_h = torch.einsum('hpl,bcl->bhcp', a, residual)
    spec_head_mix = torch.softmax(stom.spec_head_logits, dim=0).to(device=x_hist_norm_targets.device, dtype=x_hist_norm_targets.dtype)
    spec_gain = F.softplus(stom.raw_spec_gain).to(device=x_hist_norm_targets.device, dtype=x_hist_norm_targets.dtype)
    y_spec = spec_gain * torch.einsum('h,bhcp->bcp', spec_head_mix, y_spec_h)

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
    if stom.ablation_mode == 'no_horizon_gate':
        horizon = torch.zeros_like(horizon)
    gate_feat = torch.cat([gate_base_feat.unsqueeze(2).expand(-1, -1, m.pred_len, -1), horizon], dim=-1)
    gate_logits = stom.gate_mlp(gate_feat.float()).to(dtype=x_hist_norm_targets.dtype)
    expert_mask = stom._expert_mask(x_hist_norm_targets.device, x_hist_norm_targets.dtype).view(1, 1, 1, 4)
    gate = torch.softmax(gate_logits.masked_fill(expert_mask <= 0, -1e9), dim=-1)
    expert_output = torch.stack([y_phase, y_spec, y_smooth, y_local], dim=-1)
    weighted = expert_output * gate
    y_norm = weighted.sum(dim=-1)
    y_final = (y_norm * std[:, -m.number_of_targets:] + mean[:, -m.number_of_targets:]).permute(0, 2, 1)
    y_phase_actual = (y_phase * std[:, -m.number_of_targets:] + mean[:, -m.number_of_targets:]).permute(0, 2, 1)
    spectral_effective = torch.einsum('h,hpl->pl', spec_head_mix, a)
    raw = {
        'gate': gate.detach().cpu().to(torch.float16).numpy(),
        'expert_output': expert_output.detach().cpu().to(torch.float16).numpy(),
        'weighted_expert_output': weighted.detach().cpu().to(torch.float16).numpy(),
        'y_phase_or_backbone': y_phase_actual.detach().cpu().float().numpy(),
        'y_final': y_final.detach().cpu().float().numpy(),
        'y_final_norm': y_norm.detach().cpu().to(torch.float16).numpy(),
        'spectral_kernel_by_head': a.detach().cpu().float().numpy(),
        'spectral_kernel_effective': spectral_effective.detach().cpu().float().numpy(),
    }
    return y_final, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--command-file', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-samples', type=int, default=0, help='0 means full test set; positive uses deterministic prefix subset')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args_cli = ap.parse_args()

    args = parse_command_file(Path(args_cli.command_file))
    device = torch.device('cuda:0' if args_cli.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    args.use_gpu = device.type == 'cuda'
    args.gpu = 0
    args.batch_size_test = 0
    args.num_workers = 0

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    model = stom_model.Model(args).to(device)
    state = torch.load(args_cli.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(unwrap_state_dict(state), strict=False)
    if missing:
        print('missing_keys', missing[:20], 'count', len(missing))
    if unexpected:
        print('unexpected_keys', unexpected[:20], 'count', len(unexpected))
    model.eval()

    _, loader = data_provider(args, 'test')
    stores = {k: [] for k in ['gate', 'expert_output', 'weighted_expert_output', 'y_true', 'y_phase_or_backbone', 'y_final', 'y_final_norm', 'sample_mse_stom', 'sample_mse_linattn']}
    sample_indices = []
    seen = 0
    f_dim = -args.number_of_targets
    if args.features == 'MS':
        f_dim = -1

    with torch.no_grad():
        for data in loader:
            batch_x = data[0].float().to(device)
            batch_y = data[1].float().to(device)
            batch_x_mark = data[2].float().to(device)
            batch_y_mark = data[3].float().to(device)
            pred_len = args.pred_len
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            y_final, raw = forward_stom_raw(model, batch_x, batch_x_mark, dec_inp, batch_y_mark, use_amp=False)
            true = batch_y[:, -pred_len:, f_dim:].detach().cpu().float().numpy()
            y_final_np = raw['y_final'][:, :, f_dim if f_dim == -1 else 0:]
            y_phase_np = raw['y_phase_or_backbone'][:, :, f_dim if f_dim == -1 else 0:]
            if f_dim != -1:
                y_final_np = raw['y_final']
                y_phase_np = raw['y_phase_or_backbone']
            bsz = true.shape[0]
            take = bsz
            if args_cli.max_samples > 0:
                remain = args_cli.max_samples - seen
                if remain <= 0:
                    break
                take = min(take, remain)
            sl = slice(0, take)
            stores['gate'].append(raw['gate'][sl])
            stores['expert_output'].append(raw['expert_output'][sl])
            stores['weighted_expert_output'].append(raw['weighted_expert_output'][sl])
            stores['y_true'].append(true[sl])
            stores['y_phase_or_backbone'].append(y_phase_np[sl])
            stores['y_final'].append(y_final_np[sl])
            stores['y_final_norm'].append(raw['y_final_norm'][sl])
            stores['sample_mse_stom'].append(((y_final_np[sl] - true[sl]) ** 2).mean(axis=1))
            stores['sample_mse_linattn'].append(((y_phase_np[sl] - true[sl]) ** 2).mean(axis=1))
            sample_indices.extend(range(seen, seen + take))
            seen += take
            if args_cli.max_samples > 0 and seen >= args_cli.max_samples:
                break

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.concatenate(v, axis=0) for k, v in stores.items()}
    channel_count = arrays['y_true'].shape[-1]
    np.savez_compressed(
        out,
        **arrays,
        spectral_kernel_by_head=raw['spectral_kernel_by_head'][None, ...],
        spectral_kernel_effective=raw['spectral_kernel_effective'][None, ...],
        operator_order=np.array(['phase', 'spectral', 'smooth', 'local'], dtype=object),
        sample_index=np.array(sample_indices, dtype=np.int64),
        channel_index=np.arange(channel_count, dtype=np.int64),
        checkpoint_path=np.array(str(args_cli.checkpoint), dtype=object),
        config_path=np.array(str(args_cli.command_file), dtype=object),
    )
    mse = float(((arrays['y_final'] - arrays['y_true']) ** 2).mean())
    mae = float(np.abs(arrays['y_final'] - arrays['y_true']).mean())
    meta = {
        'out': str(out),
        'dataset': args.data,
        'pred_len': args.pred_len,
        'samples_dumped': int(arrays['y_true'].shape[0]),
        'channels': int(channel_count),
        'subset_mse': mse,
        'subset_mae': mae,
        'checkpoint_path': str(args_cli.checkpoint),
        'config_path': str(args_cli.command_file),
    }
    with out.with_suffix('.json').open('w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()