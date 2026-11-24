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


# ---------------------------------------------------------------------------
# Phase 2 (after midterm): run_gpt_v2.py -> summary/gpt_v2_all.csv
# ---------------------------------------------------------------------------
V2 = os.path.join(HERE, "summary", "gpt_v2_all.csv")
REG = [("conv_T1", S1, "trained to fit, T = 1"),
       ("early_T1", S4, "early-stopped, T = 1"),
       ("conv_T1.5", S2, "trained to fit, T = 1.5")]


def _sub(df, s):
    return df[df.setting == s]


def v2_long_loop(df):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for s, c, lab in REG:
        d = _sub(df, s)
        band(ax[0], d, "t", "h", c, lab)
        band(ax[1], d, "t", "acc_p", c, lab)
        band(ax[2], d, "t", "coverage", c, lab)
    ax[0].axhline(1 - 1 / 49, color=INK2, lw=1, ls=":")
    ax[0].text(0.2, 1 - 1 / 49 - 0.06, "1 − ρ (all-false limit)", color=INK2, fontsize=8.5)
    ax[0].set(title="Hallucination rate of the text it writes", xlabel="generation t", ylim=(-0.02, 1.02))
    ax[1].set(title="Accuracy (most likely answer is true)", xlabel="generation t", ylim=(0, 1))
    ax[2].set(title="Share of true facts present in its data", xlabel="generation t", ylim=(0, 0.55))
    ax[0].legend(fontsize=8.5, loc="center right")
    n = _sub(df, "conv_T1").seed.nunique()
    fig.suptitle(f"Tiny GPT retrained from scratch on its own text, 12 generations ({n} seeds; band = 10–90%)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gpt_long_loop.png"), dpi=150)
    plt.close(fig)


def v2_accumulate(df):
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True)
    for j, (s, c, lab) in enumerate(REG):
        for y, i in [("h", 0), ("acc_p", 1)]:
            band(ax[i, j], _sub(df, s), "t", y, c, "replace (train on newest batch only)")
            band(ax[i, j], _sub(df, s + "_acc"), "t", y, INK2, "accumulate, training grows with data", ls="--")
            band(ax[i, j], _sub(df, s + "_acc_fc"), "t", y, S3, "accumulate, fixed training compute", ls=":")
            ax[i, j].set(ylim=(-0.02, 1.02))
        ax[0, j].set_title(lab)
        ax[1, j].set_xlabel("generation t")
    ax[0, 0].set_ylabel("hallucination rate")
    ax[1, 0].set_ylabel("accuracy")
    ax[0, 0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Replacing vs accumulating data (accumulate runs: 3 seeds each)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gpt_accumulate.png"), dpi=150)
    plt.close(fig)


INTERV = [("", "none", S2), ("_lam0.3", "30% fresh real data", S1), ("_verify0.5", "fact-checker (catches 50%)", S3),
          ("_drop1", "drop facts seen once", S4), ("_acc", "accumulate (training grows)", INK2)]
SHORT = {"": "none", "_lam0.3": "30% fresh", "_verify0.5": "fact-check", "_drop1": "drop once-seen",
         "_acc": "accumulate"}


def v2_interventions(df):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for k, base in enumerate(["conv_T1.5", "early_T1"]):
        for suf, lab, c in INTERV:
            d = _sub(df, base + suf)
            if not d.empty:
                band(ax[k], d, "t", "h", c, lab)
        ax[k].set(title=f"Hallucination, {'trained to fit, T = 1.5' if k == 0 else 'early-stopped, T = 1'}",
                  xlabel="generation t", ylim=(-0.02, 1.02))
    ax[0].legend(fontsize=8, loc="upper left")
    # trade-off at the last generation
    last = df[df.t == df.t.max()]
    for base, mk in [("conv_T1.5", "o"), ("early_T1", "s")]:
        for suf, lab, c in INTERV:
            d = _sub(last, base + suf)
            if d.empty:
                continue
            ax[2].errorbar(d.h.mean(), d.acc_p.mean(), xerr=d.h.std(), yerr=d.acc_p.std(), fmt=mk, color=c,
                           ms=7, capsize=2, lw=1)
            dy = -12 if (base, suf) == ("early_T1", "_verify0.5") else 4
            ax[2].annotate(SHORT[suf], (d.h.mean(), d.acc_p.mean()), xytext=(6, dy), textcoords="offset points",
                           fontsize=7.5, color=INK2)
    ax[2].plot([], [], "o", color=INK2, label="trained to fit, T = 1.5")
    ax[2].plot([], [], "s", color=INK2, label="early-stopped, T = 1")
    ax[2].legend(fontsize=8, loc="lower left")
    ax[2].set(title="Where each fix ends up (generation 11)", xlabel="hallucination rate  (lower is better)",
              ylabel="accuracy  (higher is better)", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gpt_interventions.png"), dpi=150)
    plt.close(fig)


def v2_distribution(df):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for k, base in enumerate(["conv_T1.5", "early_T1"]):
        for suf, lab, c in [("_zipf0.7", "flatter tail (Zipf 0.7)", S1), ("", "Zipf 1.0 (default)", S2),
                            ("_zipf1.3", "steeper tail (Zipf 1.3)", S3), ("_para", "3 phrasings per fact", S4)]:
            band(ax[k], _sub(df, base + suf), "t", "h", c, lab)
        ax[k].set(title=f"{'Trained to fit, T = 1.5' if k == 0 else 'Early-stopped, T = 1'}",
                  xlabel="generation t", ylabel="hallucination rate" if k == 0 else None, ylim=(-0.02, 1.02))
    ax[0].legend(fontsize=8, loc="lower right")
    d = _sub(df, "conv_T1.5_para")
    for nm, c, lab in [("A", S1, "“The capital of E is V”"), ("B", S3, "“E 's capital is V”"),
                       ("C", S2, "“V is the capital of E” (reversed)")]:
        band(ax[2], d, "t", f"h_{nm}", c, lab)
    ax[2].set(title="3-phrasing run (fit, T = 1.5), by phrasing", xlabel="generation t",
              ylim=(-0.02, 1.02))
    ax[2].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gpt_distribution.png"), dpi=150)
    plt.close(fig)


def poster_figs(df):
    """Two poster figures: big type, direct labels, minimal text."""
    rc = {"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 13,
          "ytick.labelsize": 13, "lines.linewidth": 3.2}
    with plt.rc_context(rc):
        # 1. the problem: hallucination compounds
        fig, ax = plt.subplots(figsize=(8, 5))
        for s, c, lab in [("conv_T1.5", S2, "sampled with extra\nrandomness (T = 1.5)"),
                          ("early_T1", S4, "under-trained model"),
                          ("conv_T1", S1, "fully trained, plain sampling")]:
            m = band(ax, _sub(df, s), "t", "h", c)
            xy, kw = ((1.6, 0.80), {"ha": "left"}) if s == "conv_T1.5" else \
                ((m.index[-1], m.values[-1]), dict(xytext=(-8, -26 if s != "conv_T1" else 14), textcoords="offset points"))
            ax.annotate(lab, xy, **{"ha": "right", **kw}, fontsize=13, color=c, fontweight="bold")
        ax.axhline(1 - 1 / 49, color=INK2, lw=1.2, ls=":")
        ax.text(0, 1 - 1 / 49 + 0.02, "everything it says is false", color=INK2, fontsize=12)
        ax.set(xlabel="generation (each trained only on the previous one's text)",
               ylabel="share of its facts that are false", ylim=(-0.02, 1.08), xticks=range(0, 12, 2))
        ax.set_title("Trained on its own output, a tiny GPT\nhallucinates more every round", loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "poster_compounding.png"), dpi=300)
        plt.close(fig)

        # 2. the fixes: every fix trades something
        last = df[df.t == df.t.max()]
        pts = [("conv_T1.5", "no fix", S2), ("conv_T1.5_lam0.3", "30% fresh real data", S1),
               ("conv_T1.5_verify0.5", "fact-checker (catches 50%)", S3),
               ("conv_T1.5_drop1", "drop facts seen once", S4),
               ("conv_T1.5_acc_fc", "keep old data,\nsame compute", "#7a52c7"),
               ("conv_T1.5_acc", "keep old data,\nmore compute", INK)]
        off = {"no fix": (-10, 10, "right"), "30% fresh real data": (10, -26, "left"),
               "fact-checker (catches 50%)": (10, -18, "left"), "drop facts seen once": (10, 8, "left"),
               "keep old data,\nsame compute": (0, 16, "center"), "keep old data,\nmore compute": (0, 16, "center")}
        fig, ax = plt.subplots(figsize=(8, 5))
        for s, lab, c in pts:
            d = _sub(last, s)
            ax.errorbar(d.h.mean(), d.acc_p.mean(), xerr=d.h.std(), yerr=d.acc_p.std(), fmt="o", color=c,
                        ms=13, capsize=3, lw=1.5)
            dx, dy, ha = off[lab]
            ax.annotate(lab, (d.h.mean(), d.acc_p.mean()), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, fontsize=12.5, color=c, fontweight="bold")
        ax.set(xlabel="share of its facts that are false  (lower is better)",
               ylabel="accuracy  (higher is better)",
               xlim=(0, 1), ylim=(0, 1))
        ax.set_title("After 12 generations, every fix has a cost", loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "poster_fixes.png"), dpi=300)
        plt.close(fig)


def v2_figs():
    df = pd.read_csv(V2)
    v2_long_loop(df)
    v2_accumulate(df)
    v2_interventions(df)
    v2_distribution(df)
    poster_figs(df)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["v2"]:          # python lm/plots.py v2  (needs only summary/gpt_v2_all.csv)
        v2_figs()
        raise SystemExit
    bigram_figs()
    recursion_fig()
    if os.path.exists(os.path.join(OUT, "gpt_breadth.csv")):
        gpt_figs()
        hero_fig()
    if os.path.exists(V2):
        v2_figs()
    print("figures in", FIG)
