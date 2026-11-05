"""Small-GPT experiments.

  python lm/run_gpt.py breadth   # single generation: epochs, Zipf/monofact, size, temperature
  python lm/run_gpt.py loop      # recursive retraining, 6 generations, several settings

Outputs CSVs in lm/out/.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
import gpt  # noqa: E402
import metrics  # noqa: E402
from factworld import FactWorld  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "3")))
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
N = 2000
KEEP = ["h", "h_unseen_entity", "h_seen_entity", "q_unseen", "missing_mass_src", "missing_mass_p",
        "monofact", "acc_p", "acc_u", "miscal", "coverage", "false_topk", "false_repeated_frac",
        "h_sample", "eff_support", "n_false_distinct"]


def pick(r):
    return {k: r[k] for k in KEEP}


def fresh_model(voc, d, seed):
    torch.manual_seed(seed)
    return gpt.TinyGPT(voc.size, d=d, n_head=4 if d >= 16 else 2)


def breadth():
    rows = []
    t0 = time.time()
    # (a)+(b): Zipf sweep changes the monofact rate; log metrics along training
    for s in [0.4, 0.7, 1.0, 1.3, 1.6]:
        for seed in range(2):
            w = FactWorld(1000, 50, s, seed=0)
            pairs = w.sample_from(w.p, N, np.random.default_rng(100 + seed))
            counts, voc = w.counts(pairs), gpt.Vocab(w)
            m = fresh_model(voc, 64, seed)

            def cb(ep, loss, m=m, w=w, counts=counts, voc=voc, s=s, seed=seed):
                for T in ([1.0] if ep < 200 else [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]):
                    q, fm = gpt.factoid_grid(m, voc, T)
                    rows.append({"exp": "zipf", "zipf": s, "seed": seed, "d": 64, "epoch": ep,
                                 "T": T, "loss": loss, "format": fm,
                                 **pick(metrics.compute(q, counts, w, w.p))})
            gpt.train(m, voc.encode(pairs), epochs=200, seed=seed, callback=cb, every=5)
            print(f"zipf {s} seed {seed} done {time.time() - t0:.0f}s", flush=True)
    # (c): model size at fixed data (Zipf 1.0)
    w = FactWorld(1000, 50, 1.0, seed=0)
    for d in [4, 8, 16, 32, 128]:
        for seed in range(2):
            pairs = w.sample_from(w.p, N, np.random.default_rng(100 + seed))
            counts, voc = w.counts(pairs), gpt.Vocab(w)
            m = fresh_model(voc, d, seed)
            gpt.train(m, voc.encode(pairs), epochs=200, seed=seed)
            q, fm = gpt.factoid_grid(m, voc, 1.0)
            rows.append({"exp": "size", "zipf": 1.0, "seed": seed, "d": d, "epoch": 200, "T": 1.0,
                         "params": m.n_params(), "format": fm,
                         **pick(metrics.compute(q, counts, w, w.p))})
            print(f"d {d} seed {seed} done {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "gpt_breadth.csv"), index=False)


LOOP_SETTINGS = {
    # name: (epochs, decode temperature, lambda fresh)
    "converged_T1": (100, 1.0, 0.0),
    "early_stop_T1": (20, 1.0, 0.0),
    "converged_T1.5": (100, 1.5, 0.0),
    "converged_T1.5_lam0.3": (100, 1.5, 0.3),
}


def loop(names, seeds, G=6):
    w = FactWorld(1000, 50, 1.0, seed=0)
    voc = gpt.Vocab(w)
    rows = []
    t0 = time.time()
    for name in names:
        epochs, T, lam = LOOP_SETTINGS[name]
        for seed in seeds:
            rng = np.random.default_rng(500 + seed)
            pairs = w.sample_from(w.p, N, rng)
            src = w.p
            for t in range(G):
                counts = w.counts(pairs)
                m = fresh_model(voc, 64, 1000 * seed + t)
                gpt.train(m, voc.encode(pairs), epochs=epochs, seed=1000 * seed + t)
                q, fm = gpt.factoid_grid(m, voc, T)       # distribution that writes the text
                q1, _ = gpt.factoid_grid(m, voc, 1.0)
                r = metrics.compute(q, counts, w, src)
                rows.append({"setting": name, "seed": seed, "t": t, "format": fm,
                             "h_T1": float(q1[~w.is_true].sum()), **pick(r)})
                print(f"{name} seed {seed} gen {t}: h={r['h']:.3f} acc={r['acc_p']:.3f} "
                      f"mono={r['monofact']:.3f} [{time.time() - t0:.0f}s]", flush=True)
                n_fresh = rng.binomial(N, lam)
                parts = [w.sample_from(w.p, n_fresh, rng)] if n_fresh else []
                parts.append(w.sample_from(q, N - n_fresh, rng))
                pairs = np.concatenate(parts)
                src = lam * w.p + (1 - lam) * q
            pd.DataFrame(rows).to_csv(os.path.join(OUT, f"gpt_loop_{'_'.join(names)}.csv"), index=False)


if __name__ == "__main__":
    if sys.argv[1] == "breadth":
        breadth()
    else:
        loop(sys.argv[2].split(","), range(int(sys.argv[3]) if len(sys.argv) > 3 else 3))
