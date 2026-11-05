# W3: small language models (Madhav)

Do the hallucination dynamics predicted by W1/W2 show up in an actual (small) language model,
trained on text, sampled, and retrained?

## Files

| file | what it is |
|---|---|
| `factworld.py` | templated factoid corpus "The capital of E is V ." with synthetic names; knows L* exactly |
| `metrics.py` | exact per-generation metrics (hallucination split by source, monofact, breadth check vs q_{t−1} and vs p, accuracy, coverage, miscalibration, false-fact persistence, top-k false share) |
| `bigram.py` | smoothed bigram LM over the two content slots: `mle`, `gt_uniform`, `gt_backoff` |
| `run_bigram_loop.py` | recursive loop: 3 rules × λ ∈ {0, .1, .3, .5} × filters {none, drop singletons, 50% fact-checker}, 30 generations, 10 seeds |
| `gpt.py` | tiny word-level GPT (2 layers, d = 64, 236k params); exact q over U read off the logits |
| `run_gpt.py breadth` | single generation: hallucination vs epochs, Zipf exponent (monofact rate), model size, decoding temperature |
| `run_gpt.py loop` | recursive retraining from scratch, 6 generations, 3 seeds, four settings |
| `plots.py` | all figures → `figures/` |
| `ASSUMPTIONS.md` | assumptions table (what we assumed vs. real pipelines) |

## Reproduce (CPU, ~10 min total)

```bash
pip install numpy pandas matplotlib torch
python lm/run_bigram_loop.py                       # ~35 s
python lm/run_gpt.py breadth                       # ~6 min
python lm/run_gpt.py loop converged_T1,early_stop_T1 3
python lm/run_gpt.py loop converged_T1.5,converged_T1.5_lam0.3 3
python lm/plots.py
```

World: 1,000 entities × 50 cities (ρ = 1/49 ≈ 0.02), source Zipf(1.0) over entities, n = 2,000 sentences per generation.

## Findings (mid-semester, 2026-11-05)

1. **The dichotomy shows up in a real LM.** Bigram with Good–Turing breadth: hallucination 0.25 → 0.45 → 0.59 → 0.80 (t = 5) → 0.98 (t = 30), right at the predicted fixed point 1 − ρ. MLE bigram: hallucination stays 0, but accuracy falls 0.86 → 0.53 and only 7% of truths survive in the data (Wright–Fisher drift).
2. **Breadth is relative to the previous generation (Surbhi's point), and this can be measured.** Under GT the monofact rate matches the missing mass w.r.t. q_{t−1} to within 0.01 at every generation, while missing mass w.r.t. the real p is very different (0.23 vs 0.30 at t = 1). Under MLE the monofact rate → 0 even as 47% of real probability mass is lost: the model cannot see its own collapse.
3. **False facts do not stay singletons, and the model collapses onto them.** Up to 18% of the false mass in a training sample sits on false facts seen ≥ 2 times (back-off smoothing makes it worse than uniform: 18% vs 8%). By t = 30, 98% of the model's 100 most probable facts are false. Collapse and hallucination co-exist.
4. **Interventions trade hallucination for coverage.** Fresh data λ = 0.5 caps hallucination at 0.48. Dropping singletons gets hallucination to ≈0.02, but it also deletes the monofacts Good–Turing needs, so breadth goes too and accuracy ends at MLE's level (0.49). With a filter, adding fresh data *raises* hallucination (it brings back singletons and therefore breadth). See `figures/bigram_interventions.png`.
5. **A GPT trained by MLE to convergence has (almost) no breadth.** It memorises all 2,000 sentences by epoch ~40; mass on unseen facts drops from 0.98 (epoch 5) to 0.002 while the true missing mass is 0.14, so it acts like the MLE rule. Hallucination ≈ mass on unseen facts at every point, i.e. almost every guess is wrong. Breadth comes back with under-training (h = 0.34 at epoch 20), tiny models (d = 4, 9k params: h = 0.76), or decoding temperature (T = 2 puts 0.17 on unseen facts, about the missing mass). The monofact rate was *not* a good predictor of single-generation GPT hallucination in our sweep (Zipf 0.4 and 0.7 have the same monofact rate but h = 0.92 vs 0.67 at epoch 20).
6. **GPT loop (6 generations, 3 seeds).** T = 1: hallucination creeps 0.005 → 0.024, accuracy drifts 0.87 → 0.71 and coverage of truths in the data 0.49 → 0.20 (collapse), yet 52% of false sample mass is already on repeated false facts. T = 1.5 (only partial breadth, 0.04 vs 0.14): hallucination compounds 0.08 → 0.21 → 0.35 → 0.47 → 0.59 → 0.68. Adding 30% fresh data slows it to 0.40 but does not stop it within 6 generations. Early-stopped models start at 0.34 and reach 0.63, with 83% of their top-100 facts false (collapse onto false facts). See `figures/gpt_loop.png`.
7. **Recursion check for W1.** "inherited + new" decomposition h_{t} ≈ (1 − m_t)·ĥ_sample,t + (1 − ρ)·m_t (m_t = monofact rate of generation t's data, ĥ_sample = false fraction of that data) tracks the measured bigram curves with mean abs. error 0.03 (λ = 0) / 0.08 (λ = 0.3) and gets the fixed point; it overshoots mid-way by up to ~0.1. See `figures/bigram_recursion.png`.

## Caveats / what did not work

- The GT bigram over-allocates mass to unseen facts at generation 0 (0.25 vs 0.13 needed) because both slots are smoothed separately; breadth is approximate, not exact.
- The Zipf sweep moves the monofact rate only over 0.04–0.15 with n = 2,000 and 1,000 entities, too narrow to replicate Miao & Kearns' monofact → hallucination curve.
- GPT loops are short (6 generations, 3 seeds) because each generation retrains from scratch.
