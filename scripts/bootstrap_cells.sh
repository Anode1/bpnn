#!/bin/sh
# bootstrap_cells.sh -- sampling intervals for every cell, by resampling the exchangeable unit.
#
# WHY. The 104 cells are badly non-independent: they share architectures, the subsets are nested
# (top-1000 sits inside top-10000), and the training budgets are repeated measurements of the same
# architectures. A binomial interval on a rate computed from millions of pairs is therefore meaningless,
# because the pairs are not the sampling unit. The architecture is.
#
# So each replicate draws a subsample of architectures WITHOUT replacement (with replacement would
# duplicate architectures and manufacture ties, which a rank statistic notices), then recomputes the whole
# pipeline including the top-k selection inside the replicate, so that selection variability is part of
# the interval rather than excluded from it. The SD over subsamples of size m is scaled by sqrt(m/n).
#
# The point of the exercise is a comparison, not a p-value: if the systematic residual is many multiples
# of this sampling SE then the one-parameter model is falsified as an exact statement, and the honest claim
# becomes an out-of-sample prediction error instead of a confidence interval.
cd "$(dirname "$0")/.." || exit 1
R=${R:-1500}
P=${P:-400000}
OUT=bootstrap_cells.txt
: > $OUT

one() {                                  # table, label, minv, top-k
  printf '### %s topk=%s minv=%s\n' "$2" "$4" "$3" >> $OUT
  MINV=$3 BOOT=$R BOOTK=$4 PAIRS=$P ./nb_ceiling "$1" 2>&1 \
    | sed -n '/SUBSAMPLE INTERVALS/,/honest claim/p' >> $OUT
}

for K in 1000 10000; do
  one validation/nb101_b108.txt nb101_b108_clean 50 $K &
  one validation/nb101_b108.txt nb101_b108       0  $K &
  one validation/nb101_b36.txt  nb101_b36        0  $K &
  one validation/nb101_b4.txt   nb101_b4         0  $K &
  wait
  one validation/e_t0_e199.txt  nb201_c10        0  $K &
  one validation/e_t1_e199.txt  nb201_c100       0  $K &
  one validation/e_t2_e199.txt  nb201_in16       0  $K &
  one validation/e_t0_e49.txt   nb201_c10_e50    0  $K &
  wait
done

echo "=== residual against its own sampling SE, per cell"
awk '/^### /{lab=$2" topk="$3}
     /mean residual/{ for(i=1;i<=NF;i++) if($i=="SE") se=$(i+1);
                      r=$NF+0; a=(r<0?-r:r);
                      printf "  %-28s resid %+.4f  SE %.4f  ratio %.1f\n", lab, r, se, (se>0?a/se:0) }' $OUT
