# bpnn findings

## 1. A single training run gets architecture ordering wrong about 40% of the time, on both
   benchmarks and all three datasets

`validation/nb101_flip.c`, NAS-Bench-101, 423,624 architectures each trained three independent times,
3,000,000 random pairs. The judgment uses ONE run per architecture; the reference uses the OTHER TWO,
averaged. The runs are disjoint, so the reference is independent of the judgment, and rows are binned by
the OBSERVED single-run difference, which is what a reader of a paper actually has in front of them.

    observed difference    pairs      disagree    ref unresolvable
    0.00-0.05 pp          34,518        48.2%          36.1%
    0.05-0.10             41,404        46.1%          35.5%
    0.10-0.20             84,025        41.7%          35.3%
    0.20-0.30             83,404        36.4%          34.8%
    0.30-0.50            165,997        28.9%          32.4%
    0.50-1.00            399,378        16.3%          25.0%
    1.00-2.00            693,789         4.4%          10.6%
      > 2.00           1,491,889         0.5%           2.5%
    all                2,994,404         8.4%

Mean within-architecture SD of validation accuracy: 0.918 pp.

**What the number is, stated carefully.** With three runs there is no noiseless reference, so this does
not isolate the single run's error from the reference's own. It measures the **replication disagreement
rate**: two independent estimates of the same comparison, and how often they contradict each other. That
is the quantity a reader wants regardless, because it answers whether an independent repetition of a
published comparison would come out the same way.

**Not the pathological tail.** About 1.4% of the benchmark's architectures sometimes train and sometimes
collapse to roughly chance accuracy. Dropping all 5,842 of them (`MINV=50`) halves the mean noise, from
0.918 to 0.411 pp, and leaves the disagreement rates unchanged: 41.8% against 41.7% in the 0.10-0.20
band. The effect is ordinary training variance, not the outliers.

**A design mistake worth recording, because the first version made it.** Binning by the REFERENCE gap
conditions on a noisy quantity, and selecting on it being small picks pairs the reference itself
misjudged. Binning by the observed judgment gap has no such problem and answers the practitioner's
question directly. Both are reported by the probe. On this data they agree to within half a percentage
point, so the concern was theoretically right and empirically minor here; that is worth knowing rather
than assuming either way.

**Novelty: unestablished.** Seed variance in NAS benchmarks is a studied topic and rank correlation is
standard in the performance-predictor literature, though usually between a cheap proxy and ground truth
rather than of the ground truth with itself. A literature check is in progress. This result should not
be written up as new until that returns.

### It replicates across benchmarks and datasets

`validation/nb201_extract.c` recovers NAS-Bench-201's three seeds for each of 15,625 architectures on
each of three datasets. Same probe, 2,000,000 pairs each:

    benchmark / dataset            run-to-run SD   0.10-0.20 pp   0.20-0.30 pp   0.50-1.00 pp
    NAS-Bench-101  CIFAR-10           0.918 pp         41.6%          36.4%          16.3%
    NAS-Bench-201  CIFAR-10           0.224            33.0           25.3            4.3
    NAS-Bench-201  CIFAR-100          0.352            38.2           33.4           14.5
    NAS-Bench-201  ImageNet16-120     0.375            38.4           34.2           15.9

The rates track the noise, which is the ordering a correct measurement should produce: NAS-Bench-201 on
CIFAR-10 has the smallest run-to-run spread and the lowest flip rate.

### The stronger form of the same number: what one run can resolve

The standard error of a difference between two architectures each trained *n* times is
sigma*sqrt(2/n), so a gap is resolvable at 95% two-sided confidence when it exceeds
1.96*sigma*sqrt(2/n). At n=1 that floor is 2.77*sigma, and inverting gives the runs required for a
target resolution:

    benchmark / dataset            one run resolves    runs for 0.1 pp   for 0.3 pp   for 1.0 pp
    NAS-Bench-101  CIFAR-10          2.54 pp                647              72            6
    NAS-Bench-201  CIFAR-10          0.62                    38               4            1
    NAS-Bench-201  CIFAR-100         0.97                    95              11            1
    NAS-Bench-201  ImageNet16-120    1.04                   108              12            1

**This is the practically useful statement.** Architecture-search papers routinely report CIFAR-10
improvements of 0.1 to 0.3 percentage points from a single training run per architecture. On
NAS-Bench-101 that range needs between 72 and 647 runs each to establish at 95% confidence, and a
single run cannot establish anything below 2.5 points.

Two caveats stated rather than buried. The arithmetic is Gaussian and uses the mean within-architecture
sigma, while the per-architecture spread is heavy-tailed on NAS-Bench-101, so for an architecture from
the upper tail the requirement is worse and these figures are optimistic. And NAS-Bench-201's converted
release imputes a small number of missing (architecture, seed) cells with the mean of the remaining
seeds, which lowers apparent variance there; NAS-Bench-101 has no such imputation and is the cleaner of
the two.

### Prior art, and where this sits

Not novel as a phenomenon, and the nearest prior work should be cited as corroboration rather than
competition. Yang, Esperança and Carlucci, *NAS evaluation is frustratingly hard* (ICLR 2020), §5.3,
retrain 32 DARTS-space architectures with two seeds and report Kendall tau 0.48 with a mean accuracy
change of 0.13 +- 0.08 pp. That establishes the qualitative claim on a different search space at small
scale. The NAS-Bench-101 paper reports per-architecture standard deviation as an ECDF and does not
quantify rank flips; NAS-Bench-201's paper has no such analysis; and the zero-cost-proxy literature
computes rank correlation extensively but always between a proxy and the ground truth, never of the
ground truth against itself.

Still to verify by hand before any write-up: *Surrogate NAS Benchmarks* (White et al.) §3.1.5 and
Table 3, NAS-Bench-Suite (ICLR 2022), and Chen et al. (ICLR 2024) Appendix E.4, which does compute
seed-versus-seed correlation on NAS-Bench-201 as a ceiling for its own method.
