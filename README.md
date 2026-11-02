# Inherited Hallucinations: Breadth versus Collapse under Recursive Training

CIS 6200 (Emerging Theory of LLMs) group project.
Mohit Shah, Michel Faloughi, Madhav Sharma.

## Question

When generation t+1 is trained on samples from generation t, does hallucination mass h_t grow, shrink, or stabilize, and which property of the generator decides it? Our hypothesis: breadth (the property that forces hallucination in one generation, Kalai & Vempala 2024) is the switch between compounding falsehood and silent collapse. Fresh real data (fraction λ) interpolates.

## Layout

```
theory/    W1 (Mohit): recursion derivation, lower bound, Wright-Fisher proposition
sim/       W2 (Michel): NumPy simulator, parameter sweeps, phase diagrams
lm/        W3 (Madhav): templated factoid corpus, bigram and small-GPT retraining loops
shared/    single source of truth for the model definition and metrics
```

`shared/model.md` is the only file every workstream depends on. Change it there, not in code comments.

## Quick start (sim)

```bash
pip install -r sim/requirements.txt
python sim/run.py --config sim/configs/default.yaml
python sim/plots.py results/default
```

## Status

- [ ] Model definition revised per proposal feedback (breadth w.r.t. q_t, mixed true/false sample, no singleton assumption)
- [ ] W1 one-generation recursion for E[h_{t+1} | X_t]
- [ ] W2 simulator: trajectories of (h_t, c_t, m_t) per rule
- [ ] W2 phase diagram over (λ, n)
- [ ] W3 bigram loop
- [ ] W3 GPT loop
- [ ] Related-work pass (see `shared/related_work.md`)

## References

See the proposal and `shared/related_work.md`.
