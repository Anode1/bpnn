> **CLOSED DIRECTION, 2026-08-12.** This file belongs to the architecture-noise measurement
> work, which produced no publishable finding. `doc/CLOSED.md` lists every claim and what
> became of it; two numbers here are wrong wherever they appear (the 0.084 ceiling is 0.293,
> the 3,558 indifference class is 26). Kept for the record, not for citation.

# bpnn findings

> **RETRACTED, 2026-08-12.** This document belongs to a closed direction. Several of its headline
> numbers are wrong (the 0.084 ceiling should be 0.293) or inflated (the 3,558 indifference class
> should be 26), and its central claims turned out to be prior art. Read `doc/CLOSED.md` first: it
> lists every claim and what became of it. Kept for the record, not for citation.

Measurements of how much of a reported neural-network result survives replication. Everything here comes
from public benchmark data with training replicates, needs no training compute, and regenerates from the
probes in `validation/`. Numbers are percentage points of accuracy throughout.

The data: NAS-Bench-101 trains each of 423,624 architectures three independent times on CIFAR-10;
NAS-Bench-201 trains each of 15,625 architectures three times on each of three datasets. Both
distributed tables average those runs away, which is correct for ranking architectures and destroys the
only structure these measurements can use. `validation/nb101_trials.c` and `validation/nb201_extract.c`
recover them; `validation/PROVENANCE_nas.md` has the URLs and the checks.

---

## 1. The rank-correlation ceiling  [CORRECTED, AND LARGELY PRIOR ART]

**Two corrections, both from an outside review, and they change this section substantially.**

**The estimator was wrong.** We reported the test-retest agreement as the ceiling. The maximum
correlation between a DETERMINISTIC model and a noisy label is sqrt(r_model)*sqrt(r_label) =
sqrt(r_label) (Spearman 1904); using the retest agreement itself assumes the model suffers the same noise
as the label and so puts the noise factor on both sides, understating the bound by a square root. van
Bree, Styrnal & Hebart (2025) audited 53 neuroscience papers and found 60% making exactly this error, so
it is the standard mistake. For a reported correlation the ceiling is sqrt(r); for a reported R^2 it is r.
Corrected, NB101 top-1000 is **0.293, not 0.084**.

**And the concept is a 17-year-old import with published leaderboard rows.** Saliency evaluation has done
this since at least 2009: MIT300's leaderboard carried a top row reading "Baseline: infinite humans",
obtained by predicting one group of n observers from another and extrapolating n to infinity, explicitly
"used to normalize the scores for all computational models" (Bylinskii, Judd, Oliva, Torralba, Durand,
TPAMI 2019). Brain-Score splits repetitions of the same stimulus, applies a Spearman-Brown correction, and
divides every model score by the resulting ceiling. So it is false to write that ML ignores reliability
ceilings, and the honest framing is an import into architecture search rather than a new idea.

**What is genuinely unfilled, in Brain-Score's own words:** "If source data is produced by a stochastic
process, the same procedure can be carried out on the source data, resulting in the source's reliability
r_ss ... All models that we tested so far produced deterministic responses, thus r_ss = 1 in our scoring."
In architecture search the candidate measurement IS a training run, so r_ss < 1. That slot is empty and is
the cleanest statement of what this work contributes.


Zero-cost proxies, surrogate models and performance predictors are all scored by rank correlation
against a benchmark label. That label is the mean of a few noisy training runs, so there is a ceiling on
any predictor's score: the correlation the label achieves with *itself*. No predictor can exceed it. It
is almost never reported, which leaves published correlations uninterpretable in absolute terms, since
0.7 against a ceiling of 0.95 is a poor predictor and 0.7 against a ceiling of 0.72 is nearly perfect.

Kendall tau between a one-run label and an independent two-run label (`validation/nb_ceiling.c`,
2,000,000 sampled pairs, standard error 0.0007 throughout):

    subset          NB101 CIFAR-10   NB201 CIFAR-10   NB201 CIFAR-100   NB201 ImageNet16
    top 100             0.142            0.384             0.596              0.419
    top 1000            0.084            0.536             0.549              0.495
    top 10000           0.186            0.849             0.845              0.889
    all                 0.832            0.923             0.924              0.941

Three things follow.

**The top-k rows are the ones that matter**, because a predictor's job is to rank good architectures,
not to separate good from broken. On NAS-Bench-101 the ceiling there is 0.08 to 0.19. A published tau
near or above that, computed on a comparable subset, is not evidence of a good predictor; it indicates a
leak, or a subset defined differently, or a label that is not the one assumed.

**The ceiling is not a constant of nature and cannot be assumed.** It differs by a factor of six between
the two benchmarks in the top 1000 (0.084 against 0.536). Comparisons of predictor quality across
benchmarks are not interpretable without it.

**The whole-set numbers are the misleading ones.** All four exceed 0.83, which is what makes the
phenomenon easy to miss: measured over the full space, a single run ranks architectures well, because
most of the space is separated by margins far larger than the noise.

## 2. The indifference class of the reported optimum  [CORRECTED]

A benchmark's best architecture is the argmax of a noisy mean. Counting the architectures its own data
cannot separate from that winner at 95%, testing each pair with its own noise (Welch on two three-run
means):

    benchmark / dataset        best 3-run mean   not separable from it        median arch SD
    NAS-Bench-101 CIFAR-10         95.055        3,558 of 423,624  (0.84%)        0.328
    NAS-Bench-201 CIFAR-10         94.373            34 of 15,625  (0.22%)        0.180
    NAS-Bench-201 CIFAR-100        73.503             4 of 15,625  (0.03%)        0.290
    NAS-Bench-201 ImageNet16       46.883            15 of 15,625  (0.10%)        0.300

**The 3,558 figure was inflated and must not be quoted.** NAS-Bench-101's own paper reports 11
architectures within two standard errors of its best, and an outside reviewer flagged the gap as reading
like inflation. It is. Recomputing on test accuracy the way the paper does, as a BAND around the best
using the best architecture's own standard error, gives **26**, which is the same order as their 11. Our
much larger number came from a per-pair test that also carries the OTHER architecture's noise, so an
architecture with a 40-point spread is never separable from anything and lands in the set even when its
mean is 40 points below the best. Splitting it out:

    all architectures                                  423,624
    within 2 SEM of the best (band, as the paper does)       26
    per-pair test, all architectures                     3,781
    per-pair test, excluding those with SD > 5 pp           275

So 3,506 of the 3,781 were there because THEY are noisy, not because they are good. The defensible
statements are the band count of 26 and, if a per-pair test is wanted, the 275 among stable
architectures. This is the third time today that a heavy-tailed noise distribution inflated a headline
figure of ours, which is itself worth recording as a pattern rather than three separate slips.

## 3. How often one training run gets a comparison backwards

`validation/nb101_flip.c`. The judgment uses one run per architecture; the reference uses the other two,
averaged. The runs are disjoint, so the reference is independent of the judgment. Rows are binned by the
**observed** single-run difference, which is what a reader of a paper has in front of them, over
2,000,000 random pairs per dataset.

    observed diff    NB101 CIFAR-10   NB201 CIFAR-10   NB201 CIFAR-100   NB201 ImageNet16
    0.00-0.05 pp         48.2%            43.8%            44.3%              45.4%
    0.05-0.10            46.1             39.7             42.3               42.3
    0.10-0.20            41.6             33.0             38.2               38.4
    0.20-0.30            36.4             25.3             33.4               34.2
    0.30-0.50            28.9             15.6             26.7               27.8
    0.50-1.00            16.3              4.3             14.5               15.9
    1.00-2.00             4.4              0.3              3.2                4.2
      > 2.00              0.5              0.0              0.1                0.1
    mean arch SD         0.918            0.224            0.352              0.375

The rates track the noise, which is the ordering a correct measurement should produce.

**What this number is.** With three runs there is no noiseless reference, so this is a **replication
disagreement rate**: two independent estimates of the same comparison, and how often they contradict.
It is not a decomposition isolating a single run's own error, and the word "backwards" should not be
read as asserting a truth claim the design does not license.

**Binning by the observed difference makes it a predictive posterior**, which is the useful reading: "I
measured 0.15 pp; how much should I believe it?" It is emphatically *not* "41.6% of true 0.15 pp gaps
are called wrong." The probe prints the reference-binned version too, and on this data the two agree to
within half a point.

**Independently replicated.** An outside recomputation with a different pair-sampling implementation and
40,000,000 pairs gave 41.88% in the 0.10-0.20 band against our 41.6%, and 41.85% after excluding the
collapse-prone architectures.

**Not driven by the pathological tail.** About 1.4% of NAS-Bench-101 sometimes trains and sometimes
collapses to roughly chance accuracy. Dropping all 5,842 of them halves the mean spread (0.918 to 0.411)
and moves the rates by 0.03 pp, because conditioning on a small observed difference already excludes
them.

**One number in this table should not be quoted.** An overall rate across all pairs (8.4% on NB101) is a
property of the pair-sampling law, not of the benchmark: uniformly drawn pairs are mostly far apart, so
it mostly measures how easy the pairs were.

## 4. Resolution, reported as a distribution because a single figure is indefensible

The standard error of a difference between two architectures each trained *n* times is sigma*sqrt(2/n),
so a gap is resolvable at 95% two-sided confidence when it exceeds 1.96*sigma*sqrt(2/n), and at n=1 the
floor is 2.77*sigma. On NAS-Bench-101, computed per architecture from its own three runs:

    architecture   sigma    1 run resolves   runs for 0.3 pp   runs for 0.1 pp
    median         0.328        0.91 pp              9                83
    p75            0.516        1.43                23               205
    p90            0.786        2.18                53               474
    p99           42.680      118.22           155,509         1,399,581
    worst         48.035      133.06           196,975         1,772,774

**Why this is a distribution and not a headline.** An earlier version of this document reported "one run
resolves 2.54 pp, and 0.1 pp needs 647 runs", derived from the mean of the per-architecture standard
deviations. An outside recomputation showed the figure moves to 1.30, 2.07 or 18.96 pp depending on
whether one uses the median, the RMS excluding collapses, or the RMS including them, with runs-for-0.1-pp
ranging from 169 to 35,938. The spread is a heavy-tailed mixture, so a single summary of it is a property
of an analyst's choice rather than of the benchmark. Lead with the flip rate, which needs no such choice.

Also stated: sigma estimated from three runs has 2 degrees of freedom and is itself good to roughly
plus or minus 60%, so these are percentiles of the estimate rather than of the truth.

## 5. At low training budget, a comparison is a coin flip at any effect size

The full NAS-Bench-101 archive holds three runs at each of 4, 12, 36 and 108 epochs.
`validation/nb101_budget.c` extracts one budget at a time; the epoch-108 rows reproduce the separate
epoch-108 archive to four decimals, which cross-checks both extractions, and mean accuracy rises 27.9,
52.7, 84.2, 90.2 across the budgets as it must.

Flip rates by observed difference, and the whole-set rank-correlation ceiling, per budget:

    epochs   median SD   0.1-0.2   0.2-0.3   0.5-1.0   1.0-2.0   tau ceiling (all)
      4        4.147      48.9%     51.2%     47.0%     45.1%        0.667
     12        5.090      49.2%     46.7%     47.3%     43.5%        0.735
     36        0.804      47.0%     44.5%     30.8%     18.3%        0.819
    108        0.328      41.6%     36.3%     16.5%      4.4%        0.832

**At 4 and 12 epochs the comparison is a coin flip at every effect size up to two percentage points.** A
1-to-2-point difference that is reliable at full training, 4.4% backwards, is 45% backwards at 4 epochs.
The 4-epoch median run-to-run spread is 4.1 pp, twelve times the 108-epoch figure.

**This is the result with the most practical bite**, because multi-fidelity and early-stopping search is
standard practice and it ranks candidates on exactly these cheap evaluations, on the assumption that a
low-fidelity ranking transfers. At 4 epochs on this benchmark there is almost no ranking to transfer:
the whole-set ceiling falls from 0.83 to 0.67, and within any effect size a practitioner would act on,
the signal is gone.

Two things not to smooth over. The noise is **not monotone** in budget: 12 epochs is noisier than 4
(5.09 against 4.15), plausibly because 4-epoch accuracy is uniformly poor, mean 27.9%, leaving less room
to diverge, while 12 epochs is where architectures separate most in training stability. And the 51.2% in
the 4-epoch 0.2-0.3 bin is above 50%, which should be read as indistinguishable from a coin flip rather
than as worse than chance.

---

## Scope: what this does and does not indict

Getting this wrong would be the fastest way to have the work dismissed.

**It does not directly indict tabular NAS papers.** The standard protocol on NAS-Bench-101 and 201
queries the benchmark's three-run mean, so a single-run flip rate says nothing about a method that used
the published label.

**Where it does bite:** methods that train architectures themselves and report the result, such as
DARTS-space claims of the form "we found architecture X and it reaches 97.5%"; predictor and proxy
evaluation, where the label is noisy and the ceiling is unreported; the benchmarks' own reported optima
together with every regret curve measured against them; and, most directly of all, multi-fidelity and
early-stopping methods, which section 5 shows are ranking on a signal that is a coin flip at 4 and 12
epochs regardless of effect size.

**And the framing that follows from that** is an instrument rather than a verdict: here is the
measurement error budget for a procedure you already run, and here is what it costs to halve it.

## Prior art

The phenomenon is a folk fact and the contribution is quantifying it. Claiming discovery would be wrong.

- **Surrogate NAS Benchmarks** (White, Zela et al., ICLR 2022) §3.1.5 models evaluation noise as a
  first-class object using 500 architectures with 5 seeds, fitting one seed against the mean of the other
  four, which is structurally the same held-out-replicate design. It also introduces **sparse Kendall
  tau**, which rounds accuracies to 0.1% specifically to ignore rank changes below that. The field has
  already encoded the observation into a metric it reports routinely. Their 500x5 set is also the natural
  external validation for a three-run methodology.
- **Accounting for Variance in ML Benchmarks** (Bouthillier et al., MLSys 2021) gives single-seed
  comparison roughly 10% false positives and 75% false negatives and proposes a decision rule.
- **NAS evaluation is frustratingly hard** (Yang, Esperança, Carlucci, ICLR 2020) §5.3 retrains 32
  DARTS-space architectures with two seeds: Kendall tau 0.48, mean accuracy change 0.13 +- 0.08 pp. The
  nearest prior art on ground-truth-versus-itself, and it should be cited as corroboration.
- **Nondeterminism and Instability in Neural Network Optimization** (Summers and Dinneen, ICML 2021).
- The **NAS-Bench-101** paper reports per-architecture standard deviation as an ECDF and does not
  quantify rank flips. **NAS-Bench-201**'s paper has no such analysis.
- The zero-cost-proxy literature computes rank correlation extensively but always between a proxy and
  the ground truth, never of the ground truth against itself. The ceiling in section 1 is what that
  literature is missing.

Still to check by hand: NAS-Bench-Suite (ICLR 2022), and Chen et al. (ICLR 2024) Appendix E.4, which
computes seed-versus-seed correlation on NAS-Bench-201 as a ceiling for its own method.

## Known weaknesses, unfixed

- **NAS-Bench-201's converted release imputes missing (architecture, seed) cells with the mean of the
  remaining seeds.** Imputed cells have artificially low variance, which deflates both the noise
  estimates and the flip rates. The count is not quantified and cannot be identified from the converted
  file alone. This is the most attackable fact in anything above; NAS-Bench-101 has no such imputation
  and is the cleaner of the two.
- Pairs share architectures, so pair-level rates are not independent observations and a binomial interval
  would be wrong. An architecture-level bootstrap is owed.
- Everything is on validation accuracy. Papers report test accuracy; both should be run.
- The flip rates are computed over uniformly drawn pairs. Restricting to the top of the distribution,
  which is where searches operate, is owed and may change the numbers.

## Errors caught and corrected, for the record

Kept because each was found by a check rather than by luck, and because they are the reason the surviving
numbers should be believed.

1. A 32-bit generator whose period (4.29e9) was exhausted by the largest cells (2.6e10 draws): the stream
   wrapped six times, repetitions 16,385 apart were identical, and the standard error was understated
   about 2.5-fold. Replaced with a 64-bit generator.
2. An outcome contaminated by an algebraic identity: since the three deviations from a three-run mean sum
   to zero, an arm scoring on two runs is partly scoring on the ground truth. The clean outcome is the
   held-out replicate.
3. Arms described in a code comment as paired that were not, because different arms consume different
   amounts of randomness and their streams desynchronise after the first run.
4. A noise control that redrew per evaluation, making it repeated re-evaluation and therefore implicit
   averaging over time, rather than the intended structural match.
5. A selection artifact: choosing the top-k by the mean of all three runs includes the run being
   correlated, and conditioning on a sum induces negative correlation between its components. It produced
   tau of -0.41, impossible for two estimates of one quantity.
6. A pooled RMS standard deviation used for the indifference test, which on NAS-Bench-101 is 4.78 pp
   because the collapse-prone 1.4% dominate it, and which declared 91% of the benchmark inseparable from
   the winner. Per-pair noise gives 0.84%.
7. A units error: NAS-Bench-201's metric is an error fraction, not a percentage, first visible as
   architecture 0 scoring 99.86% on CIFAR-10.
8. A wrong plausibility band, which refused a correct table: the CIFAR-10 axis carries the full-train,
   test-evaluated variant with a ceiling near 94.4, not `cifar10-valid` with its 91.6 ceiling.
9. A single-figure resolution claim derived from a mean over a heavy-tailed mixture, now a distribution.
