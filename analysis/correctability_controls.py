from __future__ import annotations

import csv
import argparse
from pathlib import Path

import numpy as np

OPS = ("phase", "spectral", "smooth", "local")
REPS = 100
BASE_SEED = 20260928
MAX_SAMPLES = 250


def c_many(z: np.ndarray, e: np.ndarray) -> np.ndarray:
    gram = np.einsum("snk,snl->skl", z, z)
    rhs = np.einsum("snk,sn->sk", z, e)
    eye = np.eye(z.shape[-1], dtype=z.dtype)
    coef = np.linalg.solve(gram + 1e-8 * eye, rhs[..., None])[..., 0]
    proj = np.einsum("snk,sk->sn", z, coef)
    den = np.sum(e * e, axis=1)
    return np.divide(np.sum(proj * proj, axis=1), den, out=np.zeros_like(den), where=den > 1e-12)

def load_arrays(path: Path):
    z = np.load(path, allow_pickle=True)
    weighted = z["weighted_expert_output"].astype(np.float32)
    y_norm = z["y_final_norm"].astype(np.float32)
    y_final = z["y_final"].astype(np.float32).transpose(0, 2, 1)
    y_true = z["y_true"].astype(np.float32).transpose(0, 2, 1)
    order = ["smooth" if str(x) == "trend" else str(x) for x in z["operator_order"].tolist()]
    perm = [order.index(k) for k in OPS]
    weighted = weighted[..., perm]
    offset = y_final - y_norm
    error = y_norm - (y_true - offset)
    return weighted, error

def process(setting: str, seed: int, path: Path, rng: np.random.Generator) -> list[dict]:
    weighted, error = load_arrays(path)
    weighted = weighted[:MAX_SAMPLES]
    error = error[:MAX_SAMPLES]
    b, c, h, k = weighted.shape
    rows = []
    lo, hi = 0, h
    for start in range(0, b, 128):
        stop = min(b, start + 128)
        zs = weighted[start:stop, :, lo:hi, :].reshape(stop - start, -1, k).astype(np.float64)
        es = error[start:stop, :, lo:hi].reshape(stop - start, -1).astype(np.float64)
        original = c_many(zs, es)
        s, n = zs.shape[:2]
        rand = rng.normal(size=(REPS, s, n, k))
        zn = np.linalg.norm(zs, axis=1)
        rn = np.linalg.norm(rand, axis=2)
        rand *= np.divide(zn[None], rn, out=np.zeros_like(rn), where=rn > 1e-12)[:, :, None, :]
        random_values = c_many(rand.reshape(REPS * s, n, k), np.broadcast_to(es[None], (REPS, s, n)).reshape(REPS * s, n)).reshape(REPS, s).T
        perms = np.stack([rng.permutation(n) for _ in range(REPS * k)], axis=0).reshape(REPS, k, n)
        permuted = np.stack(
            [np.stack([zs[:, perms[rep, j, :], j] for j in range(k)], axis=-1) for rep in range(REPS)],
            axis=0,
        )
        perm_values = c_many(permuted.reshape(REPS * s, n, k), np.broadcast_to(es[None], (REPS, s, n)).reshape(REPS * s, n)).reshape(REPS, s).T
        for i in range(stop - start):
            rows.append({
                "setting": setting, "seed": seed, "sample_id": start + i,
                "C_original": float(original[i]),
                "C_random_mean": float(random_values[i].mean()),
                "C_random_std": float(random_values[i].std(ddof=1)),
                "C_random_p95": float(np.percentile(random_values[i], 95)),
                "C_permuted_mean": float(perm_values[i].mean()),
                "C_permuted_std": float(perm_values[i].std(ddof=1)),
                "C_permuted_p95": float(np.percentile(perm_values[i], 95)),
            })
    return rows

def write_csv(path: Path, rows: list[dict]):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="append", nargs=3, metavar=("SETTING", "SEED", "NPZ"), required=True)
    parser.add_argument("--out-dir", default="correctability_control_results")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    requested = set()
    for setting, seed_str, filename in args.cache:
        seed = int(seed_str)
        requested.add(setting)
        rows.extend(process(setting, seed, Path(filename), np.random.default_rng(BASE_SEED + seed)))
    write_csv(out_dir / "per_sample_controls.csv", rows)
    summary = []
    for setting in sorted(requested):
        for seed in sorted({r["seed"] for r in rows if r["setting"] == setting}):
            rr = [r for r in rows if r["setting"] == setting and r["seed"] == seed]
            item = {"setting": setting, "seed": seed, "num_samples": len(rr)}
            for key in ("C_original", "C_random_mean", "C_random_std", "C_random_p95",
                        "C_permuted_mean", "C_permuted_std", "C_permuted_p95"):
                item[key] = float(np.mean([r[key] for r in rr]))
            summary.append(item)
    write_csv(out_dir / "per_seed_controls.csv", summary)
    mean_rows = []
    for setting in sorted(requested):
        rr = [r for r in summary if r["setting"] == setting]
        if rr:
            item = {"setting": setting, "num_seeds": len(rr)}
            for key in ("C_original", "C_random_mean", "C_random_p95", "C_permuted_mean", "C_permuted_p95"):
                values = np.asarray([r[key] for r in rr], dtype=float)
                item[key + "_mean"] = float(values.mean())
                item[key + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            mean_rows.append(item)
    write_csv(out_dir / "mean_std_controls.csv", mean_rows)

if __name__ == "__main__":
    main()
