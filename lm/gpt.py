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
          seed=0, callback=None, every=None):
    """Plain next-token cross-entropy on every position (MLE)."""
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
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1))
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
