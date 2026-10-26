from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

def load_cache(path):
    with np.load(path, allow_pickle=True) as z:
        return {
            "source": z["spectral_source_profile"].astype(np.float32),
            "sample_index": z["sample_index"].astype(np.int64),
            "channel_index": z["channel_index"].astype(np.int64),
        }

def source_topk(caches: dict[str, dict], setting: str, top_k: int) -> list[dict]:
    rows = []
    for si, sj in combinations(sorted(caches, key=lambda x: int(x)), 2):
        a = caches[si]
        b = caches[sj]
        
        x = a["source"].reshape(-1, a["source"].shape[-1]).astype(np.float64)
        y = b["source"].reshape(-1, b["source"].shape[-1]).astype(np.float64)
        rx = rankdata(x, axis=1)
        ry = rankdata(y, axis=1)
        rx -= rx.mean(axis=1, keepdims=True)
        ry -= ry.mean(axis=1, keepdims=True)
        den = np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))
        rho = np.divide((rx * ry).sum(axis=1), den, out=np.zeros_like(den), where=den > 1e-12)
        ix = np.argpartition(x, -top_k, axis=1)[:, -top_k:]
        iy = np.argpartition(y, -top_k, axis=1)[:, -top_k:]
        overlap = []
        for left, right in zip(ix, iy):
            ls, rs = set(left.tolist()), set(right.tolist())
            overlap.append(len(ls & rs) / len(ls | rs) if ls | rs else 1.0)
        rows.append({
            "setting": setting,
            "seed_i": int(si),
            "seed_j": int(sj),
            "top_k": top_k,
            "num_sample_channels": len(rho),
            "mean_source_spearman": float(np.mean(rho)),
            "std_source_spearman": float(np.std(rho, ddof=1)) if len(rho) > 1 else 0.0,
            "mean_top_k_jaccard": float(np.mean(overlap)),
            "std_top_k_jaccard": float(np.std(overlap, ddof=1)) if len(overlap) > 1 else 0.0,
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="source_stability_results")
    ap.add_argument("--max-samples", type=int, default=250)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    rows = []
    for setting, paths in sorted(manifest.items()):
        caches = {str(seed): load_cache(Path(path)) for seed, path in paths.items()}
        for cache in caches.values():
            for field in ("source", "sample_index"):
                cache[field] = cache[field][:args.max_samples]
        ref = next(iter(caches.values()))
        for seed, cache in caches.items():
            if not np.array_equal(cache["sample_index"], ref["sample_index"]):
                raise ValueError(f"unaligned samples: {setting}, {seed}")
            if not np.array_equal(cache["channel_index"], ref["channel_index"]):
                raise ValueError(f"unaligned channels: {setting}, {seed}")
        rows.extend(source_topk(caches, setting, 10))
    write_csv(Path(args.out_dir) / "source_profile_stability.csv", rows)

if __name__ == "__main__":
    main()
