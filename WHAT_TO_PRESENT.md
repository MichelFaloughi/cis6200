# What to present (running log)

Target: Thursday 2026-11-05 check-in. Keep this short; move stale items to the bottom.

## The one-slide story

Breadth is the switch. With breadth (Good-Turing rule), hallucination mass h_t compounds toward 1−ρ. Without breadth (empirical rule), h_t ≡ 0 but coverage of rare truths drifts to zero (Wright-Fisher). Fresh data λ slows the first but does not stop it at the fractions tried so far.

## Evidence so far

### 2026-11-02, smoke run (toy scale: |L*|=200, |H|=2000, ρ=0.1, n=400, Zipf 1.1, 3 seeds)

| rule | λ | h_0 | h_5 | h_10 | c_10 |
|---|---|---|---|---|---|
| empirical | 0 | 0.00 | 0.00 | 0.00 | 0.15 |
| good_turing | 0 | 0.14 | 0.87 | 0.91 | 1.00 |
| good_turing | 0.3 | 0.14 | 0.58 | 0.73 | 1.00 |

- Dichotomy appears immediately.
- Good-Turing coverage stays at 1.0 because the uniform mass on U \ S_t covers every truth. That is a quirk of the rule, not a finding; flag it.
- m_t (monofact rate) rises with h_t under good_turing, consistent with the mechanism in the proposal.

## Open questions to raise Thursday

1. Breadth must be defined relative to q_t, not p (Surbhi). Our rule already samples from q_t; the definition in shared/model.md needs to match. Confirm this is what she meant.
2. False facts do not stay singletons under good_turing. Measure: fraction of sample mass on false facts with count ≥ 2, per generation.
3. Three regimes, not two: can the empirical rule collapse onto a false fact if the initial sample contains one? Under our setup generation 0 is from p so it cannot; with λ<1 and a nonzero initial h it can. Worth one experiment.
4. What λ keeps h_t bounded? Phase diagram over (λ, n) from the default config.

## Figures to bring

- [ ] results/default/trajectories.png (h, c, m vs t, three rules, λ=0)
- [ ] results/default/phase_good_turing.png (terminal h over λ × n)
- [ ] same phase diagram with filter=drop_singletons for comparison

## Done / stale

- 2026-11-02: repo scaffolded, simulator baseline runs.
