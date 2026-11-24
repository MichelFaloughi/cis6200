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
| `run_gpt_v2.py` | **phase 2**: one (setting, seed) job per call; 12 generations; replace vs accumulate (+ fixed-compute control), fresh data / filters inside the GPT loop, Zipf 0.7/1.3, three phrasings per fact |
| `summarize_v2.py` | merges phase-2 runs → `summary/gpt_v2_all.csv` (every generation × seed, committed) and `summary/gpt_v2_table.md` |
| `plots.py` | all figures → `figures/` (`python lm/plots.py v2` rebuilds only the phase-2 figures from the committed CSV) |
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

Phase 2 (72 jobs, ≈ 50 min on a 12-core laptop with 6 jobs in parallel):

```bash
sh lm/run_all_v2.sh                                     # everything, then summary + figures
python lm/run_gpt_v2.py --list                          # setting names
python lm/run_gpt_v2.py conv_T1.5_acc 0                 # one job -> lm/out/v2/conv_T1.5_acc_s0.csv
python lm/plots.py v2                                   # figures only, from the committed summary CSV
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

## Findings, phase 2 (2026-11-24): longer loops, accumulation, fixes, other data

All numbers: tiny GPT retrained from scratch every generation, 12 generations, mean over seeds
(5 seeds for the three base settings, 3 for everything else; seed-to-seed sd is small, see
`summary/gpt_v2_table.md`). "Accuracy" = the model's most likely capital is the true one,
weighted by how often the entity is asked about. Figures: `gpt_long_loop.png`, `gpt_accumulate.png`,
`gpt_interventions.png`, `gpt_distribution.png`; poster versions `poster_compounding.png`, `poster_fixes.png`.

8. **The midterm trends continue and do not level off within 12 generations.** Hallucination at t = 0 → 5 → 11:
   trained-to-fit, T = 1: 0.005 → 0.027 → 0.053 (slow, but 10× up; accuracy 0.87 → 0.64; truths still in the data 50% → 14%).
   Early-stopped: 0.36 → 0.65 → 0.75 (accuracy 0.73 → 0.30). T = 1.5: 0.08 → 0.66 → 0.92, close to the all-false limit
   1 − ρ = 0.98, accuracy 0.08. Collapse onto false facts: 83% (T = 1.5) and 91% (early-stopped) of the model's 100 most
   likely facts are false at t = 11; 39–78% of the false mass in the training data sits on false facts seen ≥ 2 times.
   Reproducibility: with the same seeds, generation 0 is identical to the midterm runs; later generations differ by
   ≤ 0.01 in hallucination per run (floating-point differences from running with 2 instead of 3 CPU threads change
   which sentences are sampled), and seed-averaged hallucination agrees within 0.006 over generations 0–5.
9. **Accumulating data (keep all old data, add the new synthetic batch) always protects accuracy, but whether it stops
   hallucination depends on compute.** With training that grows with the data (same number of passes over every
   sentence): T = 1 stays at 0.003 hallucination and 0.87 accuracy (collapse fully prevented); early-stopped
   *improves* 0.34 → 0.23; T = 1.5 still rises 0.08 → 0.34, but the per-generation increase shrinks like c/(t+1)
   with c ≈ h − (false share of the data) ≈ 0.12 (measured increments 0.021 at t = 5 and 0.0105 at t = 11 vs predicted
   0.019 and 0.010), i.e. **logarithmic instead of runaway growth**. Control with fixed training compute (same number
   of gradient steps as replace): accuracy is still protected (0.85), but hallucination climbs to 0.75 (T = 1.5) and
   0.62 (early-stopped), only modestly below replace (0.92, 0.75). So part of the "accumulation fixes collapse" effect
   is really "the model gets trained more".
10. **Every fix trades something (T = 1.5 model, t = 11; hallucination / accuracy).** None: 0.92 / 0.08.
    30% fresh real data: 0.46 / 0.80 (levels off around 0.46). Fact-checker catching half the false sentences:
    0.38 / 0.63. Dropping facts seen once: 0.12 / 0.69 (lowest hallucination, but it removes the rare true facts too,
    so 30% of truths remain vs 43% with fresh data). Accumulate: 0.34 / 0.86 (more compute) or 0.75 / 0.85 (same
    compute). For the early-stopped model the two filters get hallucination to 0.04, **but only because the model
    shrinks to ≈ 18 facts**: it knows the right capital for just 4.6% of entities (unweighted), vs 34% with accumulation.
    It says little, and what it says is true. This matches the bigram result (finding 4) in a real network.
11. **Different data: tail shape changes the speed, phrasing variety makes it worse.**
    - Zipf exponent (how long-tailed the source is), t = 11: for the T = 1.5 model all reach 0.90–0.94 (the
      destination does not depend on the tail). For the early-stopped model it matters a lot: flatter tail
      (Zipf 0.7, every entity is rare) 0.95; default 0.75; steeper (Zipf 1.3, a few entities dominate) 0.43.
    - Three phrasings per fact ("The capital of E is V", "E 's capital is V", "V is the capital of E"), each sentence
      in one random phrasing: more hallucination from generation 0 and faster compounding
      (trained-to-fit T = 1: 0.019 → 0.17 vs 0.005 → 0.053 with one phrasing; T = 1.5: 0.20 → 0.97). The reversed phrasing
      hallucinates most at first (T = 1.5, t = 0: 0.33 vs 0.13; the "reversal" gap), and the gap disappears once
      the model's own text dominates. Takeaway: varied real-world phrasing makes a model less certain, which acts
      like extra breadth and speeds up compounding.
12. **Checks.** The exact readout of q agrees with 2,000 actually generated sentences at the first and last
    generation of every job (144 checks, mean |difference| 0.0095, max 0.042, the max on early-stopped models whose
    sentences are 12% malformed). The paraphrase readout was separately checked on 20,000 sampled sentences, per
    phrasing (≤ 0.02 apart).

## Caveats / what did not work

- The GT bigram over-allocates mass to unseen facts at generation 0 (0.25 vs 0.13 needed) because both slots are smoothed separately; breadth is approximate, not exact.
- The Zipf sweep moves the monofact rate only over 0.04–0.15 with n = 2,000 and 1,000 entities, too narrow to replicate Miao & Kearns' monofact → hallucination curve.
- ~~GPT loops are short (6 generations, 3 seeds)~~ fixed in phase 2 (12 generations, 3–5 seeds). Still short of the bigram's 30.
- "Share of top-100 facts that are false" is meaningless once a model uses fewer than ~100 facts (effective support
  ≈ 18 for the filtered early-stopped runs); quote it only when `eff_support` ≫ 100.
- Popularity-weighted accuracy (`acc_p`) flatters collapsed models; check `acc_u` (every entity equal) as well.
- With filters, the breadth reference `missing_mass_src` uses the unfiltered q (approximate), as in the bigram loop.
- Paraphrase runs: the next generation's phrasing is re-drawn uniformly at random rather than taken from the model's
  own phrasing preference (which drifts at T = 1.5, toward "E 's capital is V").
- Accumulate runs grow to 24,000 sentences; "trained-to-fit" there means 100 passes over all of it (≈ 12× the compute
  of a replace run at t = 11). The fixed-compute control brackets the effect from the other side (fewer passes per
  sentence, so somewhat under-trained).
