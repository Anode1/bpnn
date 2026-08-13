# CLOSED, 2026-08-12. No findings.

This direction is closed. The honest verdict is that it produced no publishable research finding, and this
file records why in enough detail that nobody, including a future version of me, reopens it on the strength
of a half-remembered number.

## Every claim made, and what happened to it

| claim | outcome |
|---|---|
| Rank-correlation ceiling as a new quantity | Prior art. 17 years of published practice in saliency evaluation; MIT300's leaderboard carried a "Baseline: infinite humans" row used to normalise all model scores (Bylinskii, Judd, Oliva, Torralba, Durand, TPAMI 2019). Brain-Score splits stimulus repetitions with a Spearman-Brown correction and divides every score by the ceiling. |
| Whole-space tau = 0.83 on NAS-Bench-101 | Published. NAS-Bench-301 / Surrogate NAS Benchmarks (ICLR 2022), Appendix Table 4: same estimator, same benchmark, same number, three seed splits. |
| Ceiling collapses on the top-k subset | Published. Dushatskiy, Alderliesten & Bosman, GECCO 2022, Fig. 2: 0.702 to 0.198 on the top 20% of NB101, with the same thesis. Also NATS-Bench §4.2, and NB101's own appendix Fig. 7 carries a percentile-by-reliability matrix. |
| Our ceiling values (0.084 at top-1000) | Wrong. The bound for a deterministic model is sqrt(reliability), not the reliability; using the retest value assumes the model carries the label's noise. Corrected: 0.293. |
| One-parameter law from sigma_W/sigma_B | Anticipated in closed form. Adhikari, Timofte & Ignatov (2026) Theorem 8 gives the Spearman analogue as a function of an architecture-to-noise SNR, with a ceiling corollary. |
| Indifference class of 3,558 architectures | Inflated. The band criterion the benchmark itself uses gives 26 against their stated 11. Of our 3,781 on test, 3,506 were included because they are noisy, not because they are good. |
| Noise not monotone in training budget | True but uninteresting on its own. |
| Reversal rate vs OBSERVED effect size | The one thing an outside reviewer could not find published, and they rated it above the ceiling. But its endpoints are known: Reimers & Gurevych give the zero-difference point (26% type I errors between identical approaches) and Bouthillier et al. parameterise by the latent difference. Plotting it against the observed difference is a presentational improvement on known quantities. Not a finding. |
| 4-epoch comparisons are a coin flip at any effect size | The epoch-noise axis is in NB101's own appendix. Our contribution was the binning. Not a finding. |

Earlier the same day, in the predecessor direction: "the convolution emerges once the selection signal is
repaired" was falsified three ways within hours of being written, and the paper drafted on it
(`~/articles/smbpann2/estimator.tex`) is retired with a banner.

## What is actually left, and it is not research

- **`validation/resolve.c`** — a small self-testing tool that answers three questions from repeated
  scores: is this comparison real, how many runs would make it real, and what is the best correlation any
  predictor could score against these labels. Dependency-free, refuses to guess when given one run per
  candidate. Usable by anyone running sweeps. A utility, not a result.
- **`validation/nb101_trials.c`, `nb101_budget.c`, `nb201_extract.c`** — recover the per-seed values that
  both benchmarks' distributed tables average away, checked to 1e-4 pp against the published averages.
  Data engineering somebody else may find useful.
- **`validation/pairstat.c`** — paired statistics with named tests and a self-test gate. Predates this
  direction and outlives it.
- **The engine** — unchanged, gradient-checked, 32 unit checks.

## The pattern worth carrying forward

Twelve of our own errors were caught by checks or by outside review during this direction. Three separate
headline numbers were inflated by the same cause: summarising a heavy-tailed noise distribution by a mean
or a pooled RMS. Two were selection artifacts from conditioning on a quantity we then correlated. One was
an exhausted 32-bit PRNG. One was a units error. One was a plausibility band calibrated to the wrong
variant of a dataset.

The auditing works. What it has now demonstrated twice is that it was applied to directions that did not
survive it, which is information about how directions were chosen rather than about the auditing. The
checks caught things late, after they had been believed and in one case written up, rather than before.
The cheap fix is to search the literature *first* — the prior art that closed this direction was findable
in an afternoon, and was found only after four days of measurement.

## Do not reopen this on the strength of

- The 0.084 number. It was wrong; the value is 0.293.
- The 3,558 number. It was inflated; the value is 26.
- "Nobody has measured this." They have, repeatedly, in at least three literatures.
