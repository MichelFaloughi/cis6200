"""W3 phase 2: longer GPT loops, accumulate vs replace, interventions, distribution changes.

One job = one (setting, seed) pair, so jobs can run in parallel:

    python lm/run_gpt_v2.py <setting> <seed> [generations]
    python lm/run_gpt_v2.py --list

Writes lm/out/v2/<setting>_s<seed>.csv (one row per generation).

Settings combine five knobs:
  epochs     100 = trained to convergence (memorises its data), 20 = early-stopped
  T          decoding temperature used to write the next generation's text
  data       "replace": each generation trains on N fresh sentences only
             "accumulate": each generation trains on ALL data so far
             (the real generation-0 data plus every synthetic batch), as the
             web does (Gerstgrasser et al. 2024)
  lam/filter fresh-real-data fraction, or a filter on the synthetic text
             (drop_singletons, or an oracle fact-checker verify_r that catches
             each false sentence with probability r)
  zipf/para  source long-tail exponent; para=True writes each fact in one of
             three word orders (see gpt.MultiVocab)

The random-number call order for plain "replace" runs is identical to
run_gpt.py: generation 0 reproduces the midterm runs exactly; later
generations drift by <= 0.01 in hallucination because the CPU thread count
(THREADS) changes floating-point summation order. Use THREADS=2 to match
the committed phase-2 results.
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
from run_bigram_loop import apply_filter  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
OUT = os.path.join(os.path.dirname(__file__), "out", "v2")
os.makedirs(OUT, exist_ok=True)
N = 2000
KEEP = ["h", "h_unseen_entity", "h_seen_entity", "q_unseen", "missing_mass_src", "missing_mass_p",
        "monofact", "acc_p", "acc_u", "miscal", "coverage", "false_topk", "false_repeated_frac",
        "h_sample", "eff_support", "n_false_distinct"]


def S(epochs, T, data="replace", lam=0.0, filt="none", zipf=1.0, para=False, fixed_compute=False):
    return dict(epochs=epochs, T=T, data=data, lam=lam, filt=filt, zipf=zipf, para=para,
                fixed_compute=fixed_compute)


SETTINGS = {
    # 1. longer loops (12 generations, 5 seeds)
    "conv_T1": S(100, 1.0),
    "early_T1": S(20, 1.0),
    "conv_T1.5": S(100, 1.5),
    # 2. accumulate instead of replace
    "conv_T1_acc": S(100, 1.0, data="accumulate"),
    "early_T1_acc": S(20, 1.0, data="accumulate"),
    "conv_T1.5_acc": S(100, 1.5, data="accumulate"),
    # control: accumulate, but with the SAME number of gradient steps as replace
    # (epochs scaled by N / data size), so "more data" is not confused with "more training"
    "conv_T1_acc_fc": S(100, 1.0, data="accumulate", fixed_compute=True),
    "early_T1_acc_fc": S(20, 1.0, data="accumulate", fixed_compute=True),
    "conv_T1.5_acc_fc": S(100, 1.5, data="accumulate", fixed_compute=True),
    # 3. interventions inside the GPT loop
    "conv_T1.5_lam0.3": S(100, 1.5, lam=0.3),
    "conv_T1.5_drop1": S(100, 1.5, filt="drop_singletons"),
    "conv_T1.5_verify0.5": S(100, 1.5, filt="verify_0.5"),
    "early_T1_lam0.3": S(20, 1.0, lam=0.3),
    "early_T1_drop1": S(20, 1.0, filt="drop_singletons"),
    "early_T1_verify0.5": S(20, 1.0, filt="verify_0.5"),
    # 4. distribution changes: flatter / steeper long tail, paraphrased text
    "conv_T1.5_zipf0.7": S(100, 1.5, zipf=0.7),
    "conv_T1.5_zipf1.3": S(100, 1.5, zipf=1.3),
    "early_T1_zipf0.7": S(20, 1.0, zipf=0.7),
    "early_T1_zipf1.3": S(20, 1.0, zipf=1.3),
    "conv_T1_para": S(100, 1.0, para=True),
    "conv_T1.5_para": S(100, 1.5, para=True),
    "early_T1_para": S(20, 1.0, para=True),
}


def text_check(m, voc, w, cfg, seed):
    """Hallucination rate of actually generated text (string-matched), to
    confirm the exact grid describes what the model writes."""
    if cfg["para"]:
        parsed = gpt.multi_sample_text(m, voc, w, k=2000, temperature=cfg["T"], seed=seed)
        ok = [p for p in parsed if p is not None]
    else:
        parsed, _ = gpt.sample_text(m, voc, w, k=2000, temperature=cfg["T"], seed=seed)
        ok = [p for p in parsed if p is not None]
    if not ok:
        return float("nan"), 1.0
    false = np.mean([not w.is_true[e, v] for e, v, *_ in ok])
    return float(false), 1 - len(ok) / len(parsed)


def run(name, seed, G=12):
    cfg = SETTINGS[name]
    w = FactWorld(1000, 50, cfg["zipf"], seed=0)
    voc = gpt.MultiVocab(w) if cfg["para"] else gpt.Vocab(w)
    rng = np.random.default_rng(500 + seed)
    trng = np.random.default_rng(700 + seed)     # template choice (para only)
    frng = np.random.default_rng(900 + seed)     # filter randomness

    pairs = w.sample_from(w.p, N, rng)
    tmpl = trng.integers(0, 3, len(pairs))
    blocks, tblocks, srcs = [pairs], [tmpl], [w.p]   # used by "accumulate"
    src = w.p
    rows, t0 = [], time.time()
    for t in range(G):
        if cfg["data"] == "accumulate":
            pairs, tmpl = np.concatenate(blocks), np.concatenate(tblocks)
            src = sum(srcs) / len(srcs)     # every block has N sentences
        counts = w.counts(pairs)
        ms = 1000 * seed + t
        torch.manual_seed(ms)
        m = gpt.TinyGPT(voc.size, d=64, n_head=4)
        data = voc.encode(pairs, tmpl) if cfg["para"] else voc.encode(pairs)
        epochs = cfg["epochs"]
        if cfg["fixed_compute"]:
            epochs = max(1, round(epochs * N / len(pairs)))
        loss = gpt.train(m, data, epochs=epochs, seed=ms,
                         ignore_index=voc.pad if cfg["para"] else -100)
        extra = {}
        if cfg["para"]:
            q, parts = gpt.multi_factoid_grid(m, voc, cfg["T"])
            q1, _ = gpt.multi_factoid_grid(m, voc, 1.0)
            for k, nm in enumerate(gpt.TEMPLATES):
                extra[f"pi_{nm}"] = float(parts["pi"][k])
                extra[f"h_{nm}"] = float(parts[nm][~w.is_true].sum() / parts[nm].sum())
                extra[f"acc_{nm}"] = float((w.p_entity * (parts[nm].argmax(1) == w.truth)).sum())
            extra["format"] = float("nan")
        else:
            q, fm = gpt.factoid_grid(m, voc, cfg["T"])
            q1, _ = gpt.factoid_grid(m, voc, 1.0)
            extra["format"] = fm
        r = metrics.compute(q, counts, w, src)
        if t in (0, G - 1):
            extra["h_text"], extra["malformed"] = text_check(m, voc, w, cfg, seed=ms)
        rows.append({"setting": name, "seed": seed, "t": t, "n_train": len(pairs), "epochs": epochs, "loss": loss,
                     "h_T1": float(q1[~w.is_true].sum()), **{k: r[k] for k in KEEP}, **extra})
        print(f"{name} s{seed} t{t}: n={len(pairs)} h={r['h']:.3f} acc={r['acc_p']:.3f} "
              f"cov={r['coverage']:.3f} mono={r['monofact']:.3f} [{time.time() - t0:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{name}_s{seed}.csv"), index=False)

        # ---- next generation's new data -----------------------------------
        lam = cfg["lam"]
        n_fresh = rng.binomial(N, lam)
        if cfg["filt"] == "none":
            parts_ = [w.sample_from(w.p, n_fresh, rng)] if n_fresh else []
            parts_.append(w.sample_from(q, N - n_fresh, rng))
            new = np.concatenate(parts_)
        else:   # filter the synthetic part only, refill until N - n_fresh survive
            fresh = w.sample_from(w.p, n_fresh, rng) if n_fresh else np.empty((0, 2), int)
            need, kept = N - n_fresh, []
            for _ in range(50):
                if need <= 0:
                    break
                b = apply_filter(w.sample_from(q, 2 * N + 10, rng), w, cfg["filt"], frng)
                kept.append(b[:need])
                need -= len(kept[-1])
            assert need <= 0, "filter removed too much"
            new = np.concatenate([fresh, *kept])
        new_t = trng.integers(0, 3, len(new))
        if cfg["data"] == "accumulate":
            blocks.append(new)
            tblocks.append(new_t)
            srcs.append(q)
        else:
            pairs, tmpl = new, new_t
            src = lam * w.p + (1 - lam) * q
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "--list":
        print("\n".join(SETTINGS))
    else:
        run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 12)
