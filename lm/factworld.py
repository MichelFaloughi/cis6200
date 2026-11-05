"""Templated factoid world for W3.

A factoid is a sentence "The capital of <E> is <V> ." with a synthetic entity
name E and a synthetic city name V. Each entity has exactly one true capital,
so the universe is U = entities x values, the truth set L* has one fact per
entity, and every other (E, V) pair is a plausible-but-false factoid in H.

    |U| = n_entities * n_values,  |L*| = n_entities,
    rho = |L*| / |H| = 1 / (n_values - 1).

The source p picks an entity from a Zipf law (in a random order, so a name
says nothing about popularity: the "regularity" condition of Topic 1B) and
always states its true capital. Distributions over U are stored as an
(n_entities, n_values) grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
           "br", "dr", "kl", "pl", "st", "tr", "gr", "sk"]
_VOWELS = ["a", "e", "i", "o", "u", "ia", "ou", "ae"]
_CODAS = ["", "", "n", "r", "s", "l", "th", "x", "nd"]


def _names(k: int, rng: np.random.Generator, syllables: int, taken: set[str]) -> list[str]:
    out: list[str] = []
    while len(out) < k:
        w = "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS) for _ in range(syllables))
        w = (w + rng.choice(_CODAS)).capitalize()
        if w not in taken:
            taken.add(w)
            out.append(w)
    return out


@dataclass
class FactWorld:
    n_entities: int = 1000
    n_values: int = 50
    zipf_s: float = 1.0
    seed: int = 0

    entities: list[str] = field(init=False)
    values: list[str] = field(init=False)
    truth: np.ndarray = field(init=False)      # truth[e] = index of e's true capital
    p_entity: np.ndarray = field(init=False)   # source marginal over entities
    p: np.ndarray = field(init=False)          # source on U, shape (E, V)
    is_true: np.ndarray = field(init=False)    # bool grid, True on L*

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        taken: set[str] = set()
        self.entities = _names(self.n_entities, rng, 3, taken)
        self.values = _names(self.n_values, rng, 2, taken)
        self.truth = rng.integers(0, self.n_values, size=self.n_entities)
        w = np.arange(1, self.n_entities + 1, dtype=float) ** (-self.zipf_s)
        self.p_entity = rng.permutation(w / w.sum())
        self.is_true = np.zeros((self.n_entities, self.n_values), dtype=bool)
        self.is_true[np.arange(self.n_entities), self.truth] = True
        self.p = self.p_entity[:, None] * self.is_true

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_entities, self.n_values)

    @property
    def rho(self) -> float:
        return 1.0 / (self.n_values - 1)

    # --- text -------------------------------------------------------------
    def sentence(self, e: int, v: int) -> str:
        return f"The capital of {self.entities[e]} is {self.values[v]} ."

    def parse(self, s: str) -> tuple[int, int] | None:
        """String-match a generated sentence back to a factoid (None if malformed)."""
        if not hasattr(self, "_eidx"):
            self._eidx = {w: i for i, w in enumerate(self.entities)}
            self._vidx = {w: i for i, w in enumerate(self.values)}
        t = s.split()
        if len(t) != 7 or t[:3] != ["The", "capital", "of"] or t[4] != "is" or t[6] != ".":
            return None
        e, v = self._eidx.get(t[3]), self._vidx.get(t[5])
        return None if e is None or v is None else (e, v)

    # --- sampling ---------------------------------------------------------
    def sample_from(self, q: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """n i.i.d. draws from a grid distribution q; returns an (n, 2) array of (e, v)."""
        flat = q.ravel() / q.sum()
        idx = rng.choice(flat.size, size=n, p=flat)
        return np.stack(np.unravel_index(idx, self.shape), axis=1)

    def counts(self, pairs: np.ndarray) -> np.ndarray:
        c = np.zeros(self.shape, dtype=np.int64)
        np.add.at(c, (pairs[:, 0], pairs[:, 1]), 1)
        return c


def drop_singletons(pairs: np.ndarray, world: FactWorld) -> np.ndarray:
    """Filter intervention: remove every factoid that appears exactly once."""
    c = world.counts(pairs)
    keep = c[pairs[:, 0], pairs[:, 1]] >= 2
    return pairs[keep]
