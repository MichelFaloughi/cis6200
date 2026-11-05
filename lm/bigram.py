"""Smoothed bigram language model over the template's two content slots.

The template tokens ("The capital of", "is", ".") are deterministic, so a
bigram model of the sentence reduces to
    q(E, V) = P(E | "of") * P(V | E ... "is")  =:  P(e) * P(v | e).

Rules (how much probability goes to things never seen in training):

  mle         P(e) = c(e)/n, P(v|e) = c(e,v)/c(e).  No breadth: unseen
              factoids get 0.  Analogue of the 'empirical' rule in sim/.
  gt_uniform  Good-Turing on the entity slot (mass N1/n spread uniformly on
              unseen entities) and absolute discounting on the value slot
              (Ney's D = n1/(n1+2 n2)); freed mass goes uniformly to values not
              seen with that entity.  Has breadth.
  gt_backoff  Same, but freed mass backs off to the value unigram
              distribution (Katz / Kneser-Ney style), the way real n-gram LMs
              smooth.  Has breadth, and its guesses favour popular values, so
              the SAME false fact can be produced many times.
"""
from __future__ import annotations

import numpy as np


def _entity_unigram_gt(ent_count: np.ndarray) -> np.ndarray:
    n = ent_count.sum()
    n1 = (ent_count == 1).sum()
    unseen = ent_count == 0
    m = n1 / n if unseen.any() else 0.0
    pe = (1 - m) * ent_count / n
    if unseen.any():
        pe[unseen] = m / unseen.sum()
    return pe


def fit(counts: np.ndarray, rule: str) -> np.ndarray:
    """Return the model's full distribution q over U as an (E, V) grid."""
    E, V = counts.shape
    ent_count = counts.sum(axis=1)
    seen_e = ent_count > 0

    if rule == "mle":
        return counts / counts.sum()

    pe = _entity_unigram_gt(ent_count)

    n1 = (counts == 1).sum()
    n2 = (counts == 2).sum()
    D = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) > 0 else 0.5

    if rule == "gt_uniform":
        base = np.full(V, 1.0 / V)
    elif rule == "gt_backoff":
        vc = counts.sum(axis=0).astype(float)
        base = (vc + 0.5) / (vc + 0.5).sum()   # tiny floor so no value is impossible
    else:
        raise ValueError(rule)

    pv = np.tile(base, (E, 1))                  # unseen entities: pure back-off
    c = counts[seen_e].astype(float)
    ce = ent_count[seen_e][:, None].astype(float)
    disc = np.maximum(c - D, 0.0) / ce
    freed = 1.0 - disc.sum(axis=1, keepdims=True)
    unseen_v = (c == 0)
    b = base[None, :] * unseen_v
    b_sum = b.sum(axis=1, keepdims=True)
    b = np.where(b_sum > 0, b / np.where(b_sum > 0, b_sum, 1), 0.0)
    pv[seen_e] = disc + freed * b
    pv /= pv.sum(axis=1, keepdims=True)
    return pe[:, None] * pv
