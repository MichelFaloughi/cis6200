"""Figures for W3.  python lm/plots.py   ->  lm/figures/*.png"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIG = os.path.join(HERE, "out"), os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# validated default palette (dataviz reference instance), light mode
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "lines.linewidth": 2, "legend.frameon": False, "font.family": "DejaVu Sans",
})
RULE_C = {"mle": S1, "gt_uniform": S2, "gt_backoff": S3}
RULE_L = {"mle": "MLE (no breadth)", "gt_uniform": "Good-Turing, uniform guess",
          "gt_backoff": "Good-Turing, back-off guess"}


def band(ax, d, x, y, c, label=None, ls="-"):
    g = d.groupby(x)[y]
    m, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
    ax.fill_between(m.index, lo, hi, color=c, alpha=0.15, lw=0)
    ax.plot(m.index, m.values, color=c, label=label, ls=ls)
    return m


def end_label(ax, m, text, c, dy=0.0):
    ax.annotate(text, (m.index[-1], m.values[-1] + dy), xytext=(4, 0), textcoords="offset points",
                color=INK2, fontsize=8.5, va="center")


def bigram_figs():
    df = pd.read_csv(os.path.join(OUT, "bigram_loop.csv"))
    base = df[(df["filter"] == "none") & (df.lam == 0)]

    # Fig 1: the dichotomy
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for r in ["mle", "gt_uniform", "gt_backoff"]:
        d = base[base.rule == r]
        band(ax[0], d, "t", "h", RULE_C[r], RULE_L[r])
        band(ax[1], d, "t", "acc_p", RULE_C[r], RULE_L[r])
        band(ax[2], d, "t", "false_topk", RULE_C[r], RULE_L[r])
    ax[0].axhline(1 - 1 / 49, color=INK2, lw=1, ls=":")
    ax[0].text(0.5, 1 - 1 / 49 - 0.06, "1 − ρ", color=INK2, fontsize=8.5)
    ax[0].set(title="Hallucination rate  q_t(H)", xlabel="generation t", ylim=(-0.02, 1.02))
    ax[1].set(title="Accuracy (model's answer = truth)", xlabel="generation t", ylim=(0, 1))
    ax[2].set(title="Share of top-100 facts that are false", xlabel="generation t", ylim=(-0.02, 1.02))
    ax[0].legend(loc="center right", fontsize=8.5)
    fig.suptitle("Bigram LM retrained on its own text (λ = 0, 10 seeds, band = 10–90%)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bigram_dichotomy.png"), dpi=150)
    plt.close(fig)

    # Fig 2: breadth is relative to the previous generation
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for a, r in zip(ax, ["gt_backoff", "mle"]):
        d = base[base.rule == r]
        band(a, d, "t", "monofact", S1, "monofact rate N₁/n (what the model can see)")
        band(a, d, "t", "missing_mass_src", S2, "true missing mass w.r.t. q_{t−1}", ls="--")
        band(a, d, "t", "missing_mass_p", S3, "true missing mass w.r.t. real p")
        a.set(title=RULE_L[r], xlabel="generation t", ylim=(0, 1))
        a.legend(fontsize=8, loc="center right" if r == "mle" else "lower right")
    fig.suptitle("Good–Turing estimates missing mass of the distribution the data came from (q_{t−1}), not of p",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bigram_breadth_local.png"), dpi=150)
    plt.close(fig)

    # Fig 3: interventions frontier at t = 30
    end = df[(df.t == 30) & (df.rule == "gt_backoff")].groupby(["filter", "lam"])[["h", "acc_p"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    FC = {"none": S1, "drop_singletons": S2, "verify_0.5": S3}
    FL = {"none": "no filter", "drop_singletons": "drop singletons", "verify_0.5": "fact-checker (50% recall)"}
    for f, c in FC.items():
        d = end[end["filter"] == f].sort_values("lam")
        ax.plot(d.acc_p, d.h, color=c, marker="o", ms=8, mec=SURF, mew=2, label=FL[f])
        for _, r in d.iterrows():
            ax.annotate(f"λ={r.lam:g}", (r.acc_p, r.h), xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color=INK2)
    mle = df[(df.t == 30) & (df.rule == "mle") & (df["filter"] == "none")].groupby("lam")[["h", "acc_p"]].mean()
    ax.plot(mle.acc_p, mle.h, color=INK2, marker="s", ms=6, lw=1, ls=":", label="MLE (no breadth), λ = 0 … 0.5")
    ax.set(xlabel="accuracy after 30 generations  (higher is better)",
           ylabel="hallucination after 30 generations", xlim=(0, 0.95), ylim=(-0.03, 1.03),
           title="Interventions (Good-Turing back-off model)")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bigram_interventions.png"), dpi=150)
    plt.close(fig)


def gpt_figs():
    b = pd.read_csv(os.path.join(OUT, "gpt_breadth.csv"))
    z = b[b.exp == "zipf"]

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    # (a) training length
    d = z[(z.zipf == 1.0) & (z["T"] == 1.0)]
    m1 = band(ax[0], d, "epoch", "h", S1)
    m2 = band(ax[0], d, "epoch", "q_unseen", S2, ls="--")
    mm = d.groupby("epoch").missing_mass_src.mean()
    ax[0].plot(mm.index, mm.values, color=S3)
    end_label(ax[0], m2, "hallucination ≈ mass on unseen", S2, 0.03)
    end_label(ax[0], mm, "true missing mass", S3)
    ax[0].set(title="(a) GPT drops breadth as it trains", xlabel="training epoch", xlim=(0, 290), ylim=(0, 0.75))
    # (b) temperature
    d = z[(z.zipf == 1.0) & (z.epoch == 200)]
    m1 = band(ax[1], d, "T", "h", S1)
    m2 = band(ax[1], d, "T", "q_unseen", S2, ls="--")
    mm = d.groupby("T").missing_mass_src.mean()
    ax[1].plot(mm.index, mm.values, color=S3)
    end_label(ax[1], m2, "hallucination ≈\nmass on unseen", S2, 0.0)
    end_label(ax[1], mm, "true missing mass", S3)
    ax[1].set(title="(b) Temperature brings breadth back", xlabel="decoding temperature T", xlim=(0.4, 3.9), ylim=(0, 1))
    # (c) monofact rate
    for ep, c, lab in [(20, S1, "epoch 20 (under-trained)"), (200, S2, "epoch 200 (converged)")]:
        d = z[(z.epoch == ep) & (z["T"] == 1.0)].groupby("zipf")[["monofact", "h"]].mean()
        ax[2].plot(d.monofact, d.h, color=c, marker="o", ms=8, mec=SURF, mew=2, label=lab)
    xs = np.linspace(0, z.monofact.max() * 1.05, 10)
    ax[2].plot(xs, (1 - 1 / 49) * xs, color=INK2, lw=1, ls=":", label="(1−ρ)·monofact (KV bound if calibrated)")
    ax[2].set(title="(c) Hallucination vs monofact rate", xlabel="monofact rate (varied via Zipf exponent)",
              ylabel="hallucination", ylim=(-0.02, 1.0))
    ax[2].legend(fontsize=8, loc="upper left")
    fig.suptitle("Single generation: tiny GPT (236k params) on 2,000 factoid sentences",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gpt_breadth.png"), dpi=150)
    plt.close(fig)

    # loop
    parts = [pd.read_csv(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.startswith("gpt_loop_")]
    L = pd.concat(parts)
    SC = {"converged_T1": S1, "early_stop_T1": S4, "converged_T1.5": S2, "converged_T1.5_lam0.3": S3}
    SL = {"converged_T1": "trained to fit, T = 1", "early_stop_T1": "early-stopped, T = 1",
          "converged_T1.5": "trained to fit, T = 1.5", "converged_T1.5_lam0.3": "T = 1.5 + 30% fresh data"}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for s, c in SC.items():
        d = L[L.setting == s]
        if d.empty:
            continue
        band(ax[0], d, "t", "h", c, SL[s])
        band(ax[1], d, "t", "acc_p", c, SL[s])
        band(ax[2], d, "t", "monofact", c, SL[s])
    ax[0].set(title="Hallucination rate of the text it writes", xlabel="generation t", ylim=(-0.02, 1))
    ax[1].set(title="Accuracy", xlabel="generation t", ylim=(0, 1))
    ax[2].set(title="Monofact rate of its training data", xlabel="generation t", ylim=(0, 0.6))
    ax[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Tiny GPT retrained from scratch on its own samples (3 seeds)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gpt_loop.png"), dpi=150)
    plt.close(fig)


def hero_fig():
    """Two-panel summary for the main slide."""
    df = pd.read_csv(os.path.join(OUT, "bigram_loop.csv"))
    base = df[(df["filter"] == "none") & (df.lam == 0)]
    L = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.startswith("gpt_loop_")])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for r, lab in [("gt_backoff", "Good–Turing (breadth)"), ("mle", "MLE (no breadth)")]:
        m = band(ax[0], base[base.rule == r], "t", "h", RULE_C[r])
        ax[0].annotate(lab, (m.index[12], m.values[12]), xytext=(0, 8 if r == "mle" else -16),
                       textcoords="offset points", fontsize=9, color=INK2)
    ax[0].axhline(1 - 1 / 49, color=INK2, lw=1, ls=":")
    ax[0].set(title="Bigram LM, 30 generations", xlabel="generation", ylabel="hallucination rate", ylim=(-0.03, 1.03))
    SC = [("converged_T1.5", S2, "sample at T = 1.5"), ("converged_T1.5_lam0.3", S3, "T = 1.5 + 30% fresh"),
          ("early_stop_T1", S4, "early-stopped"), ("converged_T1", S1, "trained to fit, T = 1")]
    for s, c, lab in SC:
        d = L[L.setting == s]
        m = band(ax[1], d, "t", "h", c)
        ax[1].annotate(lab, (m.index[-1], m.values[-1]), xytext=(4, 0), textcoords="offset points",
                       fontsize=8.5, color=INK2, va="center")
    ax[1].set(title="Tiny GPT, 6 generations", xlabel="generation", ylim=(-0.03, 1.03), xlim=(0, 8.6),
              xticks=range(6))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "slide_hero.png"), dpi=200)
    plt.close(fig)


def recursion_fig():
    """Inherited + new hallucination:  h_{t} ≈ (1 − m_t)·ĥ_sample,t + (1 − ρ)·m_t."""
    df = pd.read_csv(os.path.join(OUT, "bigram_loop.csv"))
    rho = 1 / 49
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for lam, c in [(0.0, S2), (0.3, S1)]:
        d = df[(df.rule == "gt_uniform") & (df["filter"] == "none") & (df.lam == lam)].copy()
        d["pred"] = (1 - d.monofact) * d.h_sample + (1 - rho) * d.monofact
        m = band(ax, d, "t", "h", c)
        p = d.groupby("t").pred.mean()
        ax.plot(p.index, p.values, color=c, ls="--", lw=1.5)
        ax.annotate(f"λ = {lam:g}", (m.index[-1], m.values[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="center")
    ax.plot([], [], color=INK2, label="measured")
    ax.plot([], [], color=INK2, ls="--", lw=1.5, label="(1 − m)·ĥ_sample + (1 − ρ)·m")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set(title="One-step recursion vs measured (GT-uniform bigram)", xlabel="generation t",
           ylabel="hallucination rate", ylim=(0, 1.02), xlim=(0, 33))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bigram_recursion.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    bigram_figs()
    recursion_fig()
    if os.path.exists(os.path.join(OUT, "gpt_breadth.csv")):
        gpt_figs()
        hero_fig()
    print("figures in", FIG)
