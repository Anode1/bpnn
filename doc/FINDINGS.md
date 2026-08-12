# bpnn findings

## 1. A single training run gets the ordering of two architectures wrong about 40% of the time
   in the band architecture-search papers contest

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
