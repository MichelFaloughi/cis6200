"""Factoid-universe recursive training simulator.

Everything here mirrors shared/model.md. If the two disagree, model.md wins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Universe:
    """Finite universe U = L* ∪ H with a long-tailed source p on L*."""

    n_true: int
    n_false: int
    zipf_s: float
    rng: np.random.Generator

    size: int = field(init=False)
    p: np.ndarray = field(init=False)  # source on U (zero on H)
    is_true: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.size = self.n_true + self.n_false
        ranks = np.arange(1, self.n_true + 1, dtype=float)
        w = ranks ** (-self.zipf_s)
        self.p = np.zeros(self.size)
        self.p[: self.n_true] = w / w.sum()
        self.is_true = np.zeros(self.size, dtype=bool)
        self.is_true[: self.n_true] = True

    @property
    def rho(self) -> float:
        return self.n_true / self.n_false


# ---------------------------------------------------------------------------
# Generator rules: sample counts -> distribution q on U
# ---------------------------------------------------------------------------

def rule_empirical(counts: np.ndarray, n: int, **_) -> np.ndarray:
    return counts / n


def rule_good_turing(counts: np.ndarray, n: int, **_) -> np.ndarray:
    n1 = int((counts == 1).sum())
    m = n1 / n
    unseen = counts == 0
    q = (1.0 - m) * counts / n
    k = int(unseen.sum())
    if k > 0 and m > 0:
        q[unseen] = m / k
    return q / q.sum()


def rule_add_alpha(counts: np.ndarray, n: int, alpha: float = 0.01, **_) -> np.ndarray:
    q = counts + alpha
    return q / q.sum()


RULES = {
    "empirical": rule_empirical,
    "good_turing": rule_good_turing,
    "add_alpha": rule_add_alpha,
}


# ---------------------------------------------------------------------------
# Filters: sample counts -> filtered counts (applied before the rule)
# ---------------------------------------------------------------------------

def filter_none(counts: np.ndarray, **_) -> np.ndarray:
    return counts


def filter_drop_singletons(counts: np.ndarray, **_) -> np.ndarray:
    out = counts.copy()
    out[out == 1] = 0
    return out


FILTERS = {
    "none": filter_none,
    "drop_singletons": filter_drop_singletons,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(q: np.ndarray, counts: np.ndarray, n: int, U: Universe) -> dict:
    support = q > 0
    return {
        "h": float(q[~U.is_true].sum()),
        "c": float((support & U.is_true).sum() / U.n_true),
        "m": float((counts == 1).sum() / n),
        "support_size": int(support.sum()),
        "n_false_in_sample": int(counts[~U.is_true].sum()),
    }


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run(
    U: Universe,
    n: int,
    lam: float,
    rule: str,
    generations: int,
    filt: str = "none",
    rule_kwargs: dict | None = None,
) -> list[dict]:
    rng = U.rng
    rule_fn = RULES[rule]
    filt_fn = FILTERS[filt]
    rule_kwargs = rule_kwargs or {}

    # generation 0: n draws from p
    counts = np.bincount(rng.choice(U.size, size=n, p=U.p), minlength=U.size)
    counts = filt_fn(counts)
    q = rule_fn(counts, counts.sum(), **rule_kwargs)
    rows = [{"t": 0, **metrics(q, counts, n, U)}]

    for t in range(1, generations + 1):
        n_fresh = rng.binomial(n, lam)
        fresh = rng.choice(U.size, size=n_fresh, p=U.p)
        synth = rng.choice(U.size, size=n - n_fresh, p=q)
        counts = np.bincount(np.concatenate([fresh, synth]), minlength=U.size)
        counts = filt_fn(counts)
        q = rule_fn(counts, counts.sum(), **rule_kwargs)
        rows.append({"t": t, **metrics(q, counts, n, U)})

    return rows
