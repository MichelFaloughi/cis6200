"""Exact per-generation metrics. All are computable because the world knows L*.

Notation (matches shared/model.md):
  q      model distribution on U (grid E x V)
  counts training-sample counts on U
  src    distribution the training sample was drawn from
         (p at generation 0, lambda*p + (1-lambda)*q_{t-1} afterwards)
"""
from __future__ import annotations

import numpy as np

from factworld import FactWorld


def miscalibration(q: np.ndarray, p: np.ndarray, n_bins: int = 20) -> float:
    """Binned semantic-calibration error  sum_B |q(B) - p(B)|.

    Factoids are grouped by the probability the model gives them (log-spaced
    bins, plus a bin for q == 0), the discrete analogue of the B_z sets in
    Topic 1B Section 2.1 and of the binned metric in Miao & Kearns (2026).
    0 means perfectly calibrated.
    """
    q, p = q.ravel(), p.ravel()
    edges = np.logspace(-10, 0, n_bins + 1)
    b = np.digitize(q, edges)  # 0 => q < 1e-10 (effectively zero)
    qb = np.bincount(b, weights=q, minlength=n_bins + 2)
    pb = np.bincount(b, weights=p, minlength=n_bins + 2)
    return float(np.abs(qb - pb).sum())


def compute(q: np.ndarray, counts: np.ndarray, world: FactWorld,
            src: np.ndarray, top_k: int = 100) -> dict:
    T = world.is_true
    n = int(counts.sum())
    seen = counts > 0
    ent_count = counts.sum(axis=1)
    seen_ent = ent_count > 0

    q_ent = q.sum(axis=1)
    # hallucination mass, split by where it comes from
    h = float(q[~T].sum())
    h_unseen_entity = float(q[~seen_ent][~T[~seen_ent]].sum())
    h_seen_entity = h - h_unseen_entity

    # answer accuracy: does the model's most likely capital for e equal the truth?
    correct = (q.argmax(axis=1) == world.truth) & (q_ent > 0)   # no mass = no answer
    acc_p = float((world.p_entity * correct).sum())   # weighted by how often e is asked
    acc_u = float(correct.mean())                     # every entity equally

    # sample composition
    false_in_sample = counts[~T]
    false_mass = false_in_sample.sum()
    flat_q = q.ravel()
    top = np.argsort(flat_q)[::-1][:min(top_k, int((flat_q > 0).sum()))]
    ent = -(flat_q[flat_q > 0] * np.log(flat_q[flat_q > 0])).sum()

    return {
        "n": n,
        "h": h,
        "h_unseen_entity": h_unseen_entity,
        "h_seen_entity": h_seen_entity,
        "h_sample": float(false_mass / max(n, 1)),
        # monofact rate (Good-Turing estimate of missing mass), Topic 1B Def 3.1
        "monofact": float((counts == 1).sum() / max(n, 1)),
        # breadth check relative to the distribution the sample was ACTUALLY drawn
        # from (feedback: breadth is w.r.t. the previous generation, not p)
        "q_unseen": float(q[~seen].sum()),
        "missing_mass_src": float(src[~seen].sum()),
        # the same with respect to the real source p
        "missing_mass_p": float(world.p[~seen].sum()),
        "coverage": float(T[seen].sum() / T.sum()),   # truths present in the sample
        "acc_p": acc_p,
        "acc_u": acc_u,
        "miscal": miscalibration(q, world.p),
        # feedback: false facts need not stay singletons
        "false_repeated_frac": float(false_in_sample[false_in_sample >= 2].sum() / false_mass)
        if false_mass else 0.0,
        "n_false_distinct": int((false_in_sample > 0).sum()),
        # collapse onto false facts: how many of the model's top-k factoids are false
        "false_topk": float((~T.ravel()[top]).mean()),
        "eff_support": float(np.exp(ent)),
        "q_mass_unseen_entities": float(q_ent[~seen_ent].sum()),
    }
