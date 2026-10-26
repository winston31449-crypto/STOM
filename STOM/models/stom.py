import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from types import SimpleNamespace

class SimpleRoPE(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, pos_scale: float = 1.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.pos_scale = float(pos_scale)

    @staticmethod
    def _rot(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    def _cos_sin(self, L: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(L, device=device, dtype=self.inv_freq.dtype) * self.pos_scale  # (L,)
        freqs = torch.einsum("t,f->tf", t, self.inv_freq)                      # (L, d/2)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype=dtype, device=device) # (L, d)

        cos = emb.cos()[None, :, None, :]  # (1, L, 1, d)
        sin = emb.sin()[None, :, None, :]
        return cos, sin

    def apply_rotary(self, x: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        B, L, h, d = x.shape
        cos, sin = self._cos_sin(L, device=x.device, dtype=x.dtype)
        if transpose:
            sin = -sin  # R^T = R(-θ)
        return x * cos + self._rot(x) * sin

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        q_out = self.apply_rotary(q, transpose=False)
        k_out = self.apply_rotary(k, transpose=False)
        return q_out, k_out

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        From: https://github.com/meta-llama/llama/blob/main/llama/model.py
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @classmethod
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

    def forward(self, x):
        output = self.norm(x.float()).type_as(x)
        return output * self.weight

NORM = RMSNorm

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CAPSAttention(nn.Module):


    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0,
                 attn_type: str = 'softmax', use_time_decay_gate: bool = False,
                 spd_dropout: float = 0.0, predictor: str = 'CAPS'):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout_rate = dropout
        self.attn_type = attn_type
        self.use_time_decay_gate = use_time_decay_gate
        self.predictor = predictor

        assert d_model % n_head == 0
        self.d_head = d_model // n_head

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # SPD gating projection
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)

        # Input projection for scan recurrence
        self.p_proj = nn.Linear(d_model, d_model, bias=False)

        # Clock mechanism
        self.clock = nn.Linear(d_model, n_head, bias=False)

        # Output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.spd_dropout = nn.Dropout(spd_dropout) if spd_dropout > 0 else None

        self.rope = SimpleRoPE(dim=d_model // n_head, base=10000, pos_scale=1.0)

    def forward(self, x: torch.Tensor, use_mult_gate: bool = False, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, D]
            use_mult_gate: True for multiplicative mode, False for additive mode
        """
        if self.predictor == 'LinAttn':
            return self._forward_linattn(x)
        return self._forward_add(x)

    def _forward_linattn(self, x: torch.Tensor) -> torch.Tensor:
        """Plain causal linear attention + RoPE (no gating paths)."""
        B, T, D = x.shape
        q = self.W_q(x).view(B, T, self.n_head, -1)
        k = self.W_k(x).view(B, T, self.n_head, -1)
        v = self.W_v(x).view(B, T, self.n_head, -1)
        q, k = self.rope(q, k)
        qk = torch.einsum('blhd,bihd->bhli', q, k)
        qk = torch.tril(qk)
        output = torch.einsum('bhli,bihd->blhd', qk, v)
        output = output.reshape(B, T, -1)
        output = self.dropout(self.c_proj(output))
        return output

    def _forward_add(self, x: torch.Tensor) -> torch.Tensor:
        """Additive mode: Two Q/K pairs (p_exp only vs SPD only) combined as qk1 + qk2."""
        B, T, D = x.shape
        times = torch.arange(T, device=x.device)

        # Project Q, K, V
        q_out = self.W_q(x)
        k_out = self.W_k(x)
        v_out = self.W_v(x)

        # Apply time-decay gating if enabled
        # Reshape for multi-head
        q_out = q_out.view(B, T, self.n_head, -1)
        k_out = k_out.view(B, T, self.n_head, -1)
        v = v_out.view(B, T, self.n_head, -1)

        # Clock mechanism
        clock = F.softplus(self.clock(x)).view(B, T, self.n_head, -1) + 1e-6
        log_clock = torch.log(clock)

        # Normalize Q, K
        # q_out = F.normalize(q_out, dim=-1, p=2)
        # k_out = F.normalize(k_out, dim=-1, p=2)
        q_out, k_out = self.rope(q_out, k_out)

        # === First Q/K pair: p_exp scaling only (no SPD) ===
        q1 = q_out.clone()
        k1 = k_out.clone()

        p = self.p_proj(x).view(B, T, self.n_head, -1) + log_clock
        p_max = p.max(dim=1, keepdim=True).values
        p_exp = torch.exp(p - p_max)
        p_exp_cumsum = p_exp.cumsum(dim=1)
        q1 = q1 / (p_exp_cumsum + 1e-8)
        k1 = k1 * p_exp

        # === Second Q/K pair: SPD gating only (no p_exp) ===
        q2 = q_out.clone()
        k2 = k_out.clone()

        gj = -F.softplus(self.gate_proj(x)).view(B, T, self.n_head, -1) 
        gj = gj * clock
        gj_cumsum = gj.cumsum(dim=1).clip(-50, 40)
        gj_cumprod = torch.exp(gj_cumsum)

        q2 = q2 * gj_cumprod
        k2 = k2 / (gj_cumprod + 1e-8)

        clock_cumsum = clock.cumsum(dim=1)
        q3 = q_out / clock_cumsum
        k3 = k_out * clock

        # Attention computation with additive combination
        if self.attn_type == 'softmax':
            output = self._softmax_attention_additive(q1, k1, q2, k2, q3, k3, v)
        else:
            output = self._linear_attention_additive(q1, k1, q2, k2, q3, k3, v)

        # Output projection (no internal norm/residual/MLP for additive mode)
        output = output.reshape(B, T, -1)
        output = self.dropout(self.c_proj(output))

        return output

    def _softmax_attention_additive(self, q1, k1, q2, k2, q3, k3, v):
        """Additive attention: qk1 + qk2 combined before softmax."""

        q_cat = torch.cat([q1, q2, q3], dim=-1)  # head_dim -> 2*head_dim
        k_cat = torch.cat([k1, k2, k3], dim=-1)

        out = F.scaled_dot_product_attention(q_cat.transpose(1, 2), k_cat.transpose(1, 2), v.transpose(1, 2), 
                                             is_causal=False, dropout_p=self.dropout_rate if self.training else 0.0).transpose(1, 2).contiguous()
        return out


    def _linear_attention_additive(self, q1, k1, q2, k2, q3, k3, v, use_fla=False):
        if use_fla:
            from fla.ops.simple_gla import chunk_simple_gla
            o1, _ = chunk_simple_gla(q=q1, k=k1, v=v, output_final_state=False)
            o2, _ = chunk_simple_gla(q=q2, k=k2, v=v, output_final_state=False)
            o3, _ = chunk_simple_gla(q=q3, k=k3, v=v, output_final_state=False)
            out = o1 + o2 + o3
            return out
        else:
            # Original quadratic (faster for small T)
            q_cat = torch.cat([q1, q2, q3], dim=-1)
            k_cat = torch.cat([k1, k2, k3], dim=-1)
            qk = torch.einsum('blhd,bihd->bhli', q_cat, k_cat)
            qk = torch.tril(qk)
            out = torch.einsum('bhli,bihd->blhd', qk, v)
            return out

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layer = config.n_layer
        self.n_layer = n_layer
        self.predictor = config.predictor
        self.attn = nn.ModuleList([CAPSAttention(d_model=config.n_embd, n_head=config.n_head, dropout=config.dropout, attn_type='linear', predictor=config.predictor) for i in range(n_layer)])
        self.pns = nn.ModuleList([NORM(config.n_embd) for i in range(n_layer)])
        self.lns = nn.ModuleList([NORM(config.n_embd) for i in range(n_layer)])
        self.mlps = nn.ModuleList([MLP(config) for i in range(n_layer)])
        self.n_head = config.n_head
        self.d = config.n_embd // self.n_head

    def forward(self, x):
        for attn, pn, ln, mlp in zip(self.attn, self.pns, self.lns, self.mlps):
            x = x + attn(pn(x))
            x = x + mlp(ln(x))
        return x

def random_ratio_channel_dropout_vectorized(x, training=True):
    if not training:
        return x

    B, C, _ = x.shape
    device = x.device
    r = torch.rand(B, device=device)
    scores = torch.rand(B, C, device=device)
    threshold = r.unsqueeze(1)
    # mask: True = keep
    mask = (scores > threshold).float()
    # (B, C, 1)
    mask = mask.unsqueeze(-1)
    keep_ratio = mask.mean(dim=1, keepdim=True).clamp(min=1e-6)
    mask = mask / keep_ratio
    return x * mask

def nearest_power_of_two_to_sqrt(x: float) -> float:
    if x <= 0:
        raise ValueError("x must be positive")

    sqrt_x = math.sqrt(x)
    n = round(math.log2(sqrt_x))
    n = max(1, n)
    return 2 ** n

class STOMHead(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, channels: int, n_heads: int = 4, rank: int = 4,
                 ma_kernel: int = 25, local_kernel: int = 5, spec_gain_init: float = 6.0,
                 gate_prior='0.35,0.10,0.30,0.25', gate_hidden_dim: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.n_heads = n_heads
        self.rank = rank
        self.ma_kernel = ma_kernel
        self.local_kernel = local_kernel
        self.spec_gain_init = float(spec_gain_init)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.register_buffer("gate_prior", self._parse_gate_prior(gate_prior), persistent=False)

        self.trend_proj = nn.Linear(seq_len, pred_len)
        self.local_conv = nn.Conv1d(channels, channels, kernel_size=local_kernel, padding=local_kernel // 2, groups=channels)
        self.local_proj = nn.Linear(seq_len, pred_len)

        self.raw_alpha = nn.Parameter(self._inverse_softplus(torch.logspace(math.log10(1.0 / 16.0), math.log10(1.0 / 256.0), steps=n_heads)))
        self.raw_phase_strength = nn.Parameter(self._inverse_softplus(torch.tensor([2.0, 4.0, 4.0, 2.0])[:n_heads].clone() if n_heads <= 4 else torch.linspace(2.0, 4.0, steps=n_heads)))
        self.raw_recent_strength = nn.Parameter(self._inverse_softplus(torch.tensor([1.5, 1.0, 0.5, 0.25])[:n_heads].clone() if n_heads <= 4 else torch.linspace(1.5, 0.25, steps=n_heads)))
        self.raw_sigma_phase = nn.Parameter(self._inverse_softplus(torch.tensor([3.0, 6.0, 12.0, 24.0])[:n_heads].clone() if n_heads <= 4 else torch.linspace(3.0, 24.0, steps=n_heads)))
        self.raw_sigma_recent = nn.Parameter(self._inverse_softplus(torch.tensor([4.0, 8.0, 16.0, 32.0])[:n_heads].clone() if n_heads <= 4 else torch.linspace(4.0, 32.0, steps=n_heads)))
        self.low_rank_u = nn.Parameter(torch.zeros(n_heads, pred_len, rank))
        self.low_rank_v = nn.Parameter(torch.zeros(n_heads, seq_len, rank))
        self.spec_head_logits = nn.Parameter(torch.zeros(n_heads))
        self.raw_spec_gain = nn.Parameter(self._inverse_softplus(torch.tensor(self.spec_gain_init)))

        self.gate_mlp = nn.Sequential(
            nn.Linear(10, self.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, 4),
        )
        self._last_diag = None
        self.reset_parameters()

    @staticmethod
    def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.expm1(x.clamp_min(1e-6)))

    @staticmethod
    def _parse_gate_prior(gate_prior) -> torch.Tensor:
        if isinstance(gate_prior, str):
            values = [float(v.strip()) for v in gate_prior.replace(';', ',').split(',') if v.strip()]
        else:
            values = [float(v) for v in gate_prior]
        if len(values) != 4:
            raise ValueError(f"stom_gate_prior must have 4 values, got {values}")
        prior = torch.tensor(values, dtype=torch.float32).clamp_min(1e-6)
        return prior / prior.sum()

    def reset_parameters(self):
        with torch.no_grad():
            self.trend_proj.weight.zero_()
            self.trend_proj.bias.zero_()
            self.trend_proj.weight[:, -1] = 1.0
            self.local_conv.weight.zero_()
            center = self.local_kernel // 2
            self.local_conv.weight[:, 0, center] = 1.0
            if self.local_conv.bias is not None:
                self.local_conv.bias.zero_()
            self.local_proj.weight.zero_()
            self.local_proj.bias.zero_()
            self.low_rank_u.normal_(0.0, 0.01)
            self.low_rank_v.normal_(0.0, 0.01)
            self.spec_head_logits.zero_()
            for module in self.gate_mlp:
                if isinstance(module, nn.Linear):
                    module.weight.normal_(0.0, 1e-3)
                    module.bias.zero_()
            self.gate_mlp[-1].bias.copy_(torch.log(self.gate_prior.to(dtype=self.gate_mlp[-1].bias.dtype, device=self.gate_mlp[-1].bias.device)))

    def moving_average(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.ma_kernel // 2
        padded = F.pad(x, (pad, pad), mode='replicate')
        return F.avg_pool1d(padded, kernel_size=self.ma_kernel, stride=1)

    def _spectral_kernel(self, device, dtype):
        L, P, H = self.seq_len, self.pred_len, self.n_heads
        p = torch.arange(P, device=device, dtype=dtype)
        j = torch.arange(L, device=device, dtype=dtype)
        delta = L + p[:, None] - j[None, :]
        same = j[None, :] - p[:, None]
        recent = j - (L - 1)

        alpha = F.softplus(self.raw_alpha).to(device=device, dtype=dtype)
        phase_strength = F.softplus(self.raw_phase_strength).to(device=device, dtype=dtype)
        recent_strength = F.softplus(self.raw_recent_strength).to(device=device, dtype=dtype)
        sigma_phase = (F.softplus(self.raw_sigma_phase) + 1e-4).to(device=device, dtype=dtype)
        sigma_recent = (F.softplus(self.raw_sigma_recent) + 1e-4).to(device=device, dtype=dtype)

        decay_bias = -alpha[:, None, None] * delta[None, :, :]
        phase_bias = -0.5 * (same[None, :, :] / sigma_phase[:, None, None]) ** 2
        recent_bias = -0.5 * (recent[None, None, :] / sigma_recent[:, None, None]) ** 2
        low_rank_bias = torch.einsum('hpr,hlr->hpl', self.low_rank_u.to(dtype=dtype), self.low_rank_v.to(dtype=dtype)) / math.sqrt(self.rank)
        phi = decay_bias + phase_strength[:, None, None] * phase_bias + recent_strength[:, None, None] * recent_bias + low_rank_bias
        a = torch.softmax(phi, dim=-1)
        return a, phi, delta, alpha, phase_strength, recent_strength, sigma_phase, sigma_recent

    def forward(self, x_hist: torch.Tensor, y_phase: torch.Tensor, record_diag: bool = False):
        # x_hist/y_phase are in normalized space: x_hist [B,C,L], y_phase [B,C,P].
        smooth = self.moving_average(x_hist)
        residual = x_hist - smooth
        y_smooth = self.trend_proj(smooth)
        local_feat = F.gelu(self.local_conv(residual))
        y_local = self.local_proj(local_feat)

        a, phi, delta, alpha, phase_strength, recent_strength, sigma_phase, sigma_recent = self._spectral_kernel(x_hist.device, x_hist.dtype)
        y_spec_h = torch.einsum('hpl,bcl->bhcp', a, residual)
        spec_head_mix = torch.softmax(self.spec_head_logits, dim=0).to(device=x_hist.device, dtype=x_hist.dtype)
        spec_gain = F.softplus(self.raw_spec_gain).to(device=x_hist.device, dtype=x_hist.dtype)
        y_spec = spec_gain * torch.einsum('h,bhcp->bcp', spec_head_mix, y_spec_h)

        hist_mean = x_hist.mean(dim=-1)
        hist_std = x_hist.std(dim=-1, unbiased=False)
        last_value = x_hist[:, :, -1]
        smooth_slope = smooth[:, :, -1] - smooth[:, :, 0]
        residual_energy = (residual.pow(2).mean(dim=-1) + 1e-6).sqrt()
        y_phase_norm = (y_phase.pow(2).mean(dim=-1) + 1e-6).sqrt()
        y_spec_norm = (y_spec.pow(2).mean(dim=-1) + 1e-6).sqrt()
        y_smooth_norm = (y_smooth.pow(2).mean(dim=-1) + 1e-6).sqrt()
        y_local_norm = (y_local.pow(2).mean(dim=-1) + 1e-6).sqrt()
        gate_base_feat = torch.stack([
            hist_mean, hist_std, last_value, smooth_slope, residual_energy,
            y_phase_norm, y_spec_norm, y_smooth_norm, y_local_norm,
        ], dim=-1)
        horizon = torch.arange(self.pred_len, device=x_hist.device, dtype=gate_base_feat.dtype) / max(self.pred_len, 1)
        horizon = horizon.view(1, 1, self.pred_len, 1).expand(x_hist.size(0), x_hist.size(1), -1, -1)
        gate_feat = torch.cat([
            gate_base_feat.unsqueeze(2).expand(-1, -1, self.pred_len, -1),
            horizon,
        ], dim=-1)
        gate = torch.softmax(self.gate_mlp(gate_feat.float()).to(dtype=x_hist.dtype), dim=-1)
        y = (
            gate[..., 0] * y_phase
            + gate[..., 1] * y_spec
            + gate[..., 2] * y_smooth
            + gate[..., 3] * y_local
        )

        diag = None
        if record_diag:
            diag = self._make_diag(
                gate, y_phase, y_spec, y_smooth, y_local, a, phi, delta,
                alpha, phase_strength, recent_strength, sigma_phase, sigma_recent, spec_head_mix,
            )
            self._last_diag = diag
        return y, diag

    def _make_diag(self, gate, y_phase, y_spec, y_smooth, y_local, a, phi, delta,
                   alpha, phase_strength, recent_strength, sigma_phase, sigma_recent, spec_head_mix):
        with torch.no_grad():
            gate_f = gate.detach().float()
            a_f = a.detach().float()
            phi_f = phi.detach().float()
            delta_f = delta.detach().float().view(1, self.pred_len, self.seq_len)
            entropy = -(a_f.clamp_min(1e-12) * a_f.clamp_min(1e-12).log()).sum(dim=-1)
            eml = (a_f * delta_f).sum(dim=-1)
            p = torch.arange(self.pred_len, device=a.device).view(1, self.pred_len, 1)
            j = torch.arange(self.seq_len, device=a.device).view(1, 1, self.seq_len)
            same_window = torch.clamp(sigma_phase.detach().float().view(self.n_heads, 1, 1), min=4.0)
            same_mask = (j - p).abs().float() <= same_window
            recent_mask = j >= (self.seq_len - 8)
            same_phase_mass = a_f.masked_fill(~same_mask, 0.0).sum(dim=-1)
            recent_mass = a_f.masked_fill(~recent_mask, 0.0).sum(dim=-1)
            gate_entropy = -(gate_f.clamp_min(1e-12) * gate_f.clamp_min(1e-12).log()).sum(dim=-1)
            gate_by_horizon = gate_f.mean(dim=(0, 1))
            gate_entropy_by_horizon = gate_entropy.mean(dim=(0, 1))

            y_phase_f = y_phase.detach().float()
            y_spec_f = y_spec.detach().float()
            y_smooth_f = y_smooth.detach().float()
            y_local_f = y_local.detach().float()
            contrib_phase = (gate_f[..., 0] * y_phase_f).abs().mean()
            contrib_spec = (gate_f[..., 1] * y_spec_f).abs().mean()
            contrib_smooth = (gate_f[..., 2] * y_smooth_f).abs().mean()
            contrib_local = (gate_f[..., 3] * y_local_f).abs().mean()
            contrib_total = contrib_phase + contrib_spec + contrib_smooth + contrib_local + 1e-12
            return {
                'use_stom': True,
                'operator': 'spectral_transfer_operator_mixture_stepgate',
                'ma_kernel': int(self.ma_kernel),
                'local_kernel': int(self.local_kernel),
                'rank': int(self.rank),
                'n_spec_heads': int(self.n_heads),
                'gate_hidden_dim': int(self.gate_hidden_dim),
                'gate_prior': self.gate_prior.detach().float().cpu().tolist(),
                'gate_shape': list(gate_f.shape),
                'gate_phase_mean': float(gate_f[..., 0].mean().item()),
                'gate_spec_mean': float(gate_f[..., 1].mean().item()),
                'gate_smooth_mean': float(gate_f[..., 2].mean().item()),
                'gate_local_mean': float(gate_f[..., 3].mean().item()),
                'gate_entropy': float(gate_entropy.mean().item()),
                'gate_phase_by_horizon': gate_by_horizon[:, 0].cpu().tolist(),
                'gate_spec_by_horizon': gate_by_horizon[:, 1].cpu().tolist(),
                'gate_smooth_by_horizon': gate_by_horizon[:, 2].cpu().tolist(),
                'gate_local_by_horizon': gate_by_horizon[:, 3].cpu().tolist(),
                'gate_entropy_by_horizon': gate_entropy_by_horizon.cpu().tolist(),
                'y_phase_norm': float((y_phase_f.pow(2).mean(dim=-1) + 1e-6).sqrt().mean().item()),
                'y_spec_norm': float((y_spec_f.pow(2).mean(dim=-1) + 1e-6).sqrt().mean().item()),
                'y_smooth_norm': float((y_smooth_f.pow(2).mean(dim=-1) + 1e-6).sqrt().mean().item()),
                'y_local_norm': float((y_local_f.pow(2).mean(dim=-1) + 1e-6).sqrt().mean().item()),
                'contribution_phase_abs': float(contrib_phase.item()),
                'contribution_spec_abs': float(contrib_spec.item()),
                'contribution_smooth_abs': float(contrib_smooth.item()),
                'contribution_local_abs': float(contrib_local.item()),
                'contribution_ratio_phase': float((contrib_phase / contrib_total).item()),
                'contribution_ratio_spec': float((contrib_spec / contrib_total).item()),
                'contribution_ratio_smooth': float((contrib_smooth / contrib_total).item()),
                'contribution_ratio_local': float((contrib_local / contrib_total).item()),
                'A_entropy': float(entropy.mean().item()),
                'A_entropy_per_head': entropy.mean(dim=-1).cpu().tolist(),
                'EML': float(eml.mean().item()),
                'EML_per_head': eml.mean(dim=-1).cpu().tolist(),
                'same_phase_mass': float(same_phase_mass.mean().item()),
                'same_phase_mass_per_head': same_phase_mass.mean(dim=-1).cpu().tolist(),
                'recent_mass': float(recent_mass.mean().item()),
                'recent_mass_per_head': recent_mass.mean(dim=-1).cpu().tolist(),
                'pressure_logZ_mean': float(torch.logsumexp(phi_f, dim=-1).mean().item()),
                'alpha_h': alpha.detach().float().cpu().tolist(),
                'phase_strength_h': phase_strength.detach().float().cpu().tolist(),
                'recent_strength_h': recent_strength.detach().float().cpu().tolist(),
                'sigma_phase_h': sigma_phase.detach().float().cpu().tolist(),
                'sigma_recent_h': sigma_recent.detach().float().cpu().tolist(),
                'spec_head_mix': spec_head_mix.detach().float().cpu().tolist(),
                'spec_gain': float(F.softplus(self.raw_spec_gain.detach()).float().item()),
            }

    def get_diagnostics(self, reset: bool = False):
        diag = self._last_diag
        if reset:
            self._last_diag = None
        return diag

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.time_emb_dim = configs.time_emb_dim
        self.enc_in = configs.enc_in + configs.time_emb_dim
        self.number_of_targets = configs.number_of_targets if configs.number_of_targets else configs.enc_in
        n_series = self.number_of_targets
        self.n_series = n_series
        self.d_input = nearest_power_of_two_to_sqrt(self.enc_in) * 4 if not configs.d_model else configs.d_model       # int(np.sqrt(self.enc_in)) * 4

        # Use ARX Tokenization
        self.dropout = nn.Dropout(configs.dropout)
        self.n_head = self.d_input // 8 if not configs.n_heads else configs.n_heads
        self.d = self.d_input // self.n_head
        self.local_attention = getattr(configs, "local_attention", False)   # Disabled
        self.local_attention_window = getattr(configs, "local_attention_window", 8)

        self.seq_emb_dim = nearest_power_of_two_to_sqrt(self.enc_in) * 4
        self.channel_token_mixer_seq = nn.Linear(self.enc_in, self.d_input, bias=False)
        self.value_token_mixer = nn.Parameter(0.02*torch.randn(self.number_of_targets, 1, self.seq_emb_dim))

        hidden_dim = self.d_input + self.seq_emb_dim
        transformer_config = SimpleNamespace(n_layer=configs.e_layers, n_embd=hidden_dim, n_head=self.n_head, dropout=configs.dropout,
                                             bias=False, max_len=self.seq_len+self.pred_len, predictor=configs.predictor, decay=True, enc_in=self.number_of_targets,
                                             local_attention=self.local_attention, local_attention_window=self.local_attention_window)
        self.temporal_processor = nn.Sequential(
                                                #NORM(hidden_dim),
                                                Transformer(transformer_config),
                                                #NORM(hidden_dim)
                                                )

        self.pos_emb = nn.Embedding(self.seq_len+self.pred_len, hidden_dim)
        self.channel_emb_2 = nn.Parameter(torch.zeros(1, self.number_of_targets, 1, hidden_dim))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * configs.e_layers))

        self.out_fill = nn.Linear(self.seq_len, self.pred_len, bias=False)

        self.random_channel_dropout = configs.random_channel_dropout
        self.use_stom = bool(getattr(configs, "use_stom", 0))
        self.stom_diagnostics_enabled = False
        self._last_stom_diag = None
        self.stom_head = STOMHead(
            self.seq_len,
            self.pred_len,
            self.number_of_targets,
            n_heads=int(getattr(configs, 'stom_n_spec_heads', 4)),
            rank=int(getattr(configs, 'stom_rank', 4)),
            ma_kernel=int(getattr(configs, 'stom_ma_kernel', 25)),
            local_kernel=int(getattr(configs, 'stom_local_kernel', 5)),
            spec_gain_init=float(getattr(configs, 'stom_spec_gain_init', 6.0)),
            gate_prior=getattr(configs, 'stom_gate_prior', '0.35,0.10,0.30,0.25'),
            gate_hidden_dim=int(getattr(configs, 'stom_gate_hidden_dim', 32)),
        ) if self.use_stom else None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.00)

    def set_stom_diagnostics(self, enabled: bool):
        self.stom_diagnostics_enabled = enabled

    def get_stom_diagnostics(self, reset: bool = False):
        diag = self._last_stom_diag
        if reset:
            self._last_stom_diag = None
        return diag

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # B C L

        # preprocessing
        mean = x[:, :, [-1]] # x[:, :, -self.pred_len:].mean(dim=2, keepdim=True) # # B C 1
        std = torch.ones(x.size(0), x.size(1), 1, device=x.device) # x.std(dim=2, keepdim=True) + 1e-8 # torch.ones_like(x.std(dim=2, keepdim=True)) #                    # B C 1
        x = (x - mean) / std                                       # B C L
        x_hist_norm_targets = x[:, -self.number_of_targets:, :].clone()

        out_fill = self.out_fill(x)  
        x = torch.cat([x, out_fill], dim=2)  # B C L+pred_len
        if x_mark_enc is not None and x_mark_dec is not None and self.time_emb_dim:
            mark = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:]], dim=1).permute(0, 2, 1) # B C L
            mark = (mark - x_mark_enc.permute(0, 2, 1)[:, :, [-1]]) / (x_mark_enc.permute(0, 2, 1).std(dim=-1, keepdim=True) + 1e-8)  
            x = torch.cat([mark, x], dim=1)  # B C L


        x_dropout = random_ratio_channel_dropout_vectorized(x, self.training and self.random_channel_dropout)
        channel_token = self.channel_token_mixer_seq(x_dropout.permute(0, 2, 1))   # B L C -> B L d
        channel_token = channel_token.unsqueeze(1).expand(-1, self.number_of_targets, -1, -1)
        x = x[:, -self.number_of_targets:].unsqueeze(-1) # B C L 1
        x = torch.einsum('bcli,cid->bcld', x, self.value_token_mixer)   # B C L d
        x = torch.cat([channel_token, x], dim=-1)    # B C L 2d

        B, C, N, _ = x.shape
        pos = torch.arange(0, N, dtype=torch.long, device=x.device)
        pos_emb = self.pos_emb(pos)
        x = x + self.channel_emb_2 + pos_emb
        x = x.reshape(B * C, N, -1)                   # B*C L 2d    

        x = self.temporal_processor(x)              # B*C L 2d
        x = x[:, -self.pred_len:].reshape(B, C, self.pred_len, -1)   # B C pred_len 2d
        y_phase = torch.einsum('bcld,cdi->bcli', x[:, :, :, -self.seq_emb_dim:], self.value_token_mixer.transpose(-2, -1)).squeeze(-1)  # B C L_P
        x = y_phase
        if self.use_stom:
            x, stom_diag = self.stom_head(x_hist_norm_targets, y_phase, record_diag=self.stom_diagnostics_enabled)
            if stom_diag is not None:
                self._last_stom_diag = stom_diag

        # inverse processing
        x = (x*std[:, -self.number_of_targets:] + mean[:, -self.number_of_targets:]).permute(0, 2, 1)                  # B s Ct

        if self.training:
            return x, torch.tensor(0, device=x.device)
        else:
            return x