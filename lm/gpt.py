"""A tiny word-level GPT (< 1M parameters) trained on the factoid sentences.

Each sentence is tokenised as
    <bos> The capital of <E> is <V> . <eos>
with one token per entity name and one per city name.  Because the template is
fixed, the model's probability of any factoid can be read off exactly:
    q(e, v) = P(<E>=e | <bos> The capital of) * P(<V>=v | ... <E>=e is)
(both restricted to the slot's legal tokens; the probability of leaving the
template is reported separately as 1 - format_mass).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from factworld import FactWorld

FIXED = ["<bos>", "The", "capital", "of", "is", ".", "<eos>"]
BOS, THE, CAP, OF, IS, DOT, EOS = range(7)


class Vocab:
    def __init__(self, world: FactWorld):
        self.E, self.V = world.n_entities, world.n_values
        self.e0 = len(FIXED)
        self.v0 = self.e0 + self.E
        self.size = self.v0 + self.V

    def encode(self, pairs: np.ndarray) -> torch.Tensor:
        n = len(pairs)
        x = np.empty((n, 9), dtype=np.int64)
        x[:, :4] = [BOS, THE, CAP, OF]
        x[:, 4] = self.e0 + pairs[:, 0]
        x[:, 5] = IS
        x[:, 6] = self.v0 + pairs[:, 1]
        x[:, 7] = DOT
        x[:, 8] = EOS
        return torch.from_numpy(x)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d=64, n_layer=2, n_head=4, max_len=8):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        L = max_len
        self.register_buffer("mask", torch.triu(torch.ones(L, L, dtype=torch.bool), 1))

    def forward(self, idx):
        L = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(L, device=idx.device))
        for b in self.blocks:
            x = b(x, self.mask[:L, :L])
        return self.head(self.ln(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train(model, data: torch.Tensor, epochs: int, lr=2e-3, wd=0.0, batch=128,
          seed=0, callback=None, every=None, ignore_index=-100):
    """Plain next-token cross-entropy on every position (MLE).
    ignore_index: target token not scored (padding in the multi-template corpus)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(data)
    steps = epochs * math.ceil(n / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            xb = data[perm[i:i + batch]]
            logits = model(xb[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1),
                                   ignore_index=ignore_index)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if callback is not None and every and (ep % every == 0 or ep == epochs):
            model.eval()
            callback(ep, loss.item())
            model.train()
    model.eval()
    return loss.item()


@torch.no_grad()
def factoid_grid(model, vocab: Vocab, temperature=1.0) -> tuple[np.ndarray, float]:
    """Exact model distribution over U (E x V grid) and the in-template mass."""
    model.eval()
    prefix = torch.tensor([[BOS, THE, CAP, OF]])
    pe_full = F.softmax(model(prefix)[0, -1] / temperature, dim=-1)
    pe = pe_full[vocab.e0:vocab.v0]
    E = vocab.E
    x = torch.empty((E, 6), dtype=torch.long)
    x[:, :4] = torch.tensor([BOS, THE, CAP, OF])
    x[:, 4] = torch.arange(vocab.e0, vocab.v0)
    x[:, 5] = IS
    logits = model(x)
    pis = F.softmax(logits[:, 4] / temperature, dim=-1)[:, IS]      # P("is" | E)
    pv_full = F.softmax(logits[:, 5] / temperature, dim=-1)
    pv = pv_full[:, vocab.v0:]
    format_mass = float((pe * pis * pv.sum(1)).sum())
    pe_n = pe / pe.sum()
    pv_n = pv / pv.sum(dim=1, keepdim=True)
    q = (pe_n[:, None] * pv_n).double().numpy()
    return q / q.sum(), format_mass


@torch.no_grad()
def sample_text(model, vocab: Vocab, world: FactWorld, k=500, temperature=1.0, seed=0):
    """Free-running sampling, then string-match parsing: sanity check that the
    exact grid matches what the model actually writes."""
    torch.manual_seed(seed)
    inv = FIXED + world.entities + world.values
    x = torch.full((k, 1), BOS, dtype=torch.long)
    for _ in range(8):
        p = F.softmax(model(x)[:, -1] / temperature, dim=-1)
        x = torch.cat([x, torch.multinomial(p, 1)], dim=1)
    sents = [" ".join(inv[t] for t in row[1:8]) for row in x.tolist()]
    return [world.parse(s) for s in sents], sents


# ---------------------------------------------------------------------------
# Paraphrase corpus: the same fact (e, v) can be written in three word orders
#   A: <bos> The capital of <E> is <V> . <eos>
#   B: <bos> <E> 's capital is <V> . <eos> <pad>
#   C: <bos> <V> is the capital of <E> . <eos>
# Template C states the fact "backwards" (city first), so a model that only saw
# a fact in A/B must reverse it to answer in C. The first generated token picks
# the template (The -> A, an entity -> B, a city -> C), so the model's
# distribution over facts is still exact:  q = sum_k pi_k * q_k.
# ---------------------------------------------------------------------------
MFIXED = ["<bos>", "The", "capital", "of", "is", ".", "<eos>", "'s", "the", "<pad>"]
M_BOS, M_THE, M_CAP, M_OF, M_IS, M_DOT, M_EOS, M_S, M_the, M_PAD = range(10)
TEMPLATES = ("A", "B", "C")


class MultiVocab:
    def __init__(self, world: FactWorld):
        self.E, self.V = world.n_entities, world.n_values
        self.e0 = len(MFIXED)
        self.v0 = self.e0 + self.E
        self.size = self.v0 + self.V
        self.pad = M_PAD

    def encode(self, pairs: np.ndarray, templates: np.ndarray) -> torch.Tensor:
        """pairs (n, 2) of (e, v); templates (n,) in {0, 1, 2} for A, B, C."""
        n = len(pairs)
        E, V = self.e0 + pairs[:, 0], self.v0 + pairs[:, 1]
        x = np.full((n, 9), M_PAD, dtype=np.int64)
        a, b, c = templates == 0, templates == 1, templates == 2
        x[a] = np.stack([np.full(a.sum(), M_BOS), np.full(a.sum(), M_THE), np.full(a.sum(), M_CAP),
                         np.full(a.sum(), M_OF), E[a], np.full(a.sum(), M_IS), V[a],
                         np.full(a.sum(), M_DOT), np.full(a.sum(), M_EOS)], 1)
        x[b, :8] = np.stack([np.full(b.sum(), M_BOS), E[b], np.full(b.sum(), M_S),
                             np.full(b.sum(), M_CAP), np.full(b.sum(), M_IS), V[b],
                             np.full(b.sum(), M_DOT), np.full(b.sum(), M_EOS)], 1)
        x[c] = np.stack([np.full(c.sum(), M_BOS), V[c], np.full(c.sum(), M_IS), np.full(c.sum(), M_the),
                         np.full(c.sum(), M_CAP), np.full(c.sum(), M_OF), E[c],
                         np.full(c.sum(), M_DOT), np.full(c.sum(), M_EOS)], 1)
        return torch.from_numpy(x)

    def sentence(self, world: FactWorld, e: int, v: int, k: int) -> str:
        E, V = world.entities[e], world.values[v]
        return [f"The capital of {E} is {V} .", f"{E} 's capital is {V} .",
                f"{V} is the capital of {E} ."][k]


@torch.no_grad()
def multi_factoid_grid(model, vocab: MultiVocab, temperature=1.0):
    """Exact distribution over facts for the paraphrase model.

    Returns (q, parts) where q is the (E, V) grid and parts holds the template
    weights pi and each template's own grid q_A, q_B, q_C (each slot is
    renormalised over its legal tokens, as in factoid_grid)."""
    model.eval()
    T = temperature
    sm = lambda z: F.softmax(z / T, dim=-1)  # noqa: E731
    e0, v0, E, V = vocab.e0, vocab.v0, vocab.E, vocab.V
    first = sm(model(torch.tensor([[M_BOS]]))[0, -1])
    pi = torch.stack([first[M_THE], first[e0:v0].sum(), first[v0:].sum()])
    pi = pi / pi.sum()

    # A: The capital of <E> is <V>
    pe = sm(model(torch.tensor([[M_BOS, M_THE, M_CAP, M_OF]]))[0, -1])[e0:v0]
    x = torch.empty((E, 6), dtype=torch.long)
    x[:, :4] = torch.tensor([M_BOS, M_THE, M_CAP, M_OF])
    x[:, 4], x[:, 5] = torch.arange(e0, v0), M_IS
    pv = sm(model(x)[:, 5])[:, v0:]
    qA = (pe / pe.sum())[:, None] * (pv / pv.sum(1, keepdim=True))

    # B: <E> 's capital is <V>
    pe = first[e0:v0]
    x = torch.empty((E, 5), dtype=torch.long)
    x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4] = M_BOS, torch.arange(e0, v0), M_S, M_CAP, M_IS
    pv = sm(model(x)[:, 4])[:, v0:]
    qB = (pe / pe.sum())[:, None] * (pv / pv.sum(1, keepdim=True))

    # C: <V> is the capital of <E>   (computed as a V x E grid, then transposed)
    pv0 = first[v0:]
    x = torch.empty((V, 6), dtype=torch.long)
    x[:, 0], x[:, 1] = M_BOS, torch.arange(v0, v0 + V)
    x[:, 2:] = torch.tensor([M_IS, M_the, M_CAP, M_OF])
    pe = sm(model(x)[:, 5])[:, e0:v0]
    qC = ((pv0 / pv0.sum())[:, None] * (pe / pe.sum(1, keepdim=True))).T

    q = pi[0] * qA + pi[1] * qB + pi[2] * qC
    to = lambda a: a.double().numpy()  # noqa: E731
    q = to(q)
    return q / q.sum(), {"pi": pi.double().numpy(), "A": to(qA), "B": to(qB), "C": to(qC)}


def parse_multi(world: FactWorld, toks: list[str]):
    """String-match a generated token list (after <bos>) to (e, v, template)."""
    if not hasattr(world, "_eidx"):
        world.parse("x")  # builds the name -> index maps
    ei, vi = world._eidx, world._vidx
    t = toks
    if t[:3] == ["The", "capital", "of"] and len(t) >= 7 and t[4] == "is" and t[6] == ".":
        e, v, k = ei.get(t[3]), vi.get(t[5]), 0
    elif len(t) >= 6 and t[1:4] == ["'s", "capital", "is"] and t[5] == ".":
        e, v, k = ei.get(t[0]), vi.get(t[4]), 1
    elif len(t) >= 7 and t[1:5] == ["is", "the", "capital", "of"] and t[6] == ".":
        e, v, k = ei.get(t[5]), vi.get(t[0]), 2
    else:
        return None
    return None if e is None or v is None else (e, v, k)


@torch.no_grad()
def multi_sample_text(model, vocab: MultiVocab, world: FactWorld, k=2000, temperature=1.0, seed=0):
    """Free-running generation for the paraphrase model (sanity check of the exact grid)."""
    torch.manual_seed(seed)
    inv = MFIXED + world.entities + world.values
    x = torch.full((k, 1), M_BOS, dtype=torch.long)
    for _ in range(8):
        p = F.softmax(model(x)[:, -1] / temperature, dim=-1)
        x = torch.cat([x, torch.multinomial(p, 1)], dim=1)
    return [parse_multi(world, [inv[t] for t in row[1:]]) for row in x.tolist()]
