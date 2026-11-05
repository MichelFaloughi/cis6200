"""Recursive-training loop for the bigram LM.

generation 0 : n sentences from the real source p
generation t : n sentences, a fraction lam fresh from p, the rest sampled from
               q_{t-1}; an optional filter is applied to the synthetic part
               only (a pipeline cleans scraped/synthetic text, not its curated
               real data).  The model is refit from scratch every generation.

Usage:  python lm/run_bigram_loop.py [--quick]
Writes lm/out/bigram_loop.csv
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import bigram  # noqa: E402
import metrics  # noqa: E402
from factworld import FactWorld, drop_singletons  # noqa: E402


def apply_filter(pairs, world, filt, rng):
    if filt == "none" or len(pairs) == 0:
        return pairs
    if filt == "drop_singletons":
        return drop_singletons(pairs, world)
    if filt.startswith("verify"):
        # oracle fact-checker that catches each false sentence with prob r
        r = float(filt.split("_")[1])
        false = ~world.is_true[pairs[:, 0], pairs[:, 1]]
        caught = false & (rng.random(len(pairs)) < r)
        return pairs[~caught]
    raise ValueError(filt)


def run_one(world, n, lam, rule, filt, generations, seed):
    rng = np.random.default_rng(seed)
    pairs = world.sample_from(world.p, n, rng)
    src = world.p
    rows = []
    for t in range(generations + 1):
        counts = world.counts(pairs)
        q = bigram.fit(counts, rule)
        rows.append({"t": t, **metrics.compute(q, counts, world, src)})
        # next generation's training data
        n_fresh = rng.binomial(n, lam)
        fresh = world.sample_from(world.p, n_fresh, rng) if n_fresh else np.empty((0, 2), int)
        # keep generating until n - n_fresh synthetic sentences survive the filter
        need, kept = n - n_fresh, []
        for _ in range(50):
            if need <= 0:
                break
            batch = apply_filter(world.sample_from(q, 2 * (n - n_fresh) + 10, rng), world, filt, rng)
            kept.append(batch[:need])
            need -= len(kept[-1])
        pairs = np.concatenate([fresh, *kept])
        src = lam * world.p + (1 - lam) * q
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    world = FactWorld(n_entities=1000, n_values=50, zipf_s=1.0, seed=0)
    n = 2000
    G = 10 if args.quick else 30
    seeds = range(2 if args.quick else 10)
    grid = list(itertools.product(
        ["mle", "gt_uniform", "gt_backoff"],
        [0.0, 0.1, 0.3, 0.5],
        ["none", "drop_singletons", "verify_0.5"],
    ))
    out = []
    t0 = time.time()
    for rule, lam, filt in grid:
        for s in seeds:
            for r in run_one(world, n, lam, rule, filt, G, seed=1000 + s):
                out.append({"rule": rule, "lam": lam, "filter": filt, "seed": s, **r})
    df = pd.DataFrame(out)
    os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "out", "bigram_loop.csv")
    df.to_csv(path, index=False)
    print(f"wrote {path}: {len(df)} rows in {time.time() - t0:.1f}s  (rho={world.rho:.3f})")
    g = df[(df["filter"] == "none")].groupby(["rule", "lam", "t"])[["h", "acc_p", "monofact", "q_unseen", "missing_mass_src", "false_topk"]].mean()
    print(g.loc[(slice(None), slice(None), [0, 1, 5, 10, G]), :].round(3).to_string())


if __name__ == "__main__":
    main()
