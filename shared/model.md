# Model definition (single source of truth)

Status: **draft, pending revision after proposal feedback (2026-10-31).** Items marked TODO are the changes Surbhi asked for.

## Setting

- Finite universe U of factoids.
- Truth set L* ⊆ U, hallucination set H = U \ L*, with |L*| ≤ ρ|H|.
- Long-tailed source p on L* (Zipf with exponent s).
- Generation 0 trains on n i.i.d. draws from p.
- Generation t+1 trains on n draws, of which a fraction λ ∈ [0,1] are fresh draws from p and the rest are draws from q_t.
- Optional filter F applied to the sample before training (TODO: define; candidates are "drop singletons" and "drop items absent from an oracle subset").

## Generator rules

Given a sample X_t with empirical distribution p̂_t, support S_t, and singleton count N_1:

- **empirical**: q_t = p̂_t. No breadth.
- **good_turing**: m_t = N_1 / n. q_t puts mass m_t uniformly on U \ S_t and mass (1 − m_t) on p̂_t.
- **add_alpha**: q_t ∝ counts + α on all of U.

## Breadth

TODO. The proposal wrote breadth as q(S̄) = p(S̄) with p the true source. From generation 1 on, the sample is drawn from q_t (mixed with p), so the condition must be stated relative to the sampling distribution: q_{t+1}(S̄_t) ≈ q_t^λ(S̄_t), where q_t^λ = λp + (1−λ)q_t.

## Tracked quantities

- h_t = q_t(H): hallucination mass.
- c_t = |supp(q_t) ∩ L*| / |L*|: coverage of truths.
- m_t = N_1 / n: monofact rate (missing-mass estimate).
- miscalibration: TV(q_t restricted to bins, true bin mass) per Miao & Kearns.

All exact; the simulator knows L*.

## Known gaps (from feedback)

1. The Kalai-Vempala bound assumes the training sample lies in L*. Once hallucinations enter the sample it is mixed. W1 must extend the argument.
2. An inherited false fact need not stay a singleton: under good_turing it has mass (1−m_t)/n and can be drawn several times. The heuristic m_{t+1} ≳ e^{-1} h_t ignores this.
3. Mode collapse and rising h_t can coexist: the support can collapse onto false facts. The phase diagram should label three regimes, not two.
