#!/bin/sh
# recompute_cells.sh -- redo every cell with the corrected, selection-orthogonal design.
#
# WHY EVERY CELL MUST BE REDONE. The earlier 92-cell table selected each top-k subset by the same 2-run
# reference it then correlated against. That leaves the judgment run's variance unrestricted, which was the
# argument for it, but it restricts Cov(A,B|S) and Var(B|S), which is Pearson-Thorndike Case-2 range
# restriction. An independent statistician reproduced the entire residual pattern from that design alone,
# in a Gaussian simulation with no ties, no heteroscedasticity and normal quality. So the old residuals
# measured my design and not the data, and none of them can be reported.
#
# The corrected design selects on a run that NEITHER label uses, so both labels are orthogonal to the
# selection and the within-subset Pearson correlation of the two labels IS the ICC directly.
#
# Coverage: NAS-Bench-101 at 4 training budgets, NAS-Bench-201 on 3 datasets at 6 epochs each, four
# subset sizes per table, with and without the collapse-prone architectures on NB101.
cd "$(dirname "$0")/.." || exit 1
OUT=cells_corrected.txt
: > $OUT

run() { TAG=$1 CSV=1 PAIRS=${PAIRS:-600000} MINV=$3 ./nb_ceiling "$2" >> $OUT 2>&1; }

for B in 4 12 36 108; do
  run nb101_b$B          validation/nb101_b$B.txt 0 &
done
wait
for B in 4 12 36 108; do
  run nb101_b${B}_clean  validation/nb101_b$B.txt 50 &
done
wait
for T in 0 1 2; do
  for E in 9 24 49 99 149 199; do
    run nb201_t${T}_e$((E+1)) validation/e_t${T}_e${E}.txt 0 &
  done
  wait
done

echo "cells written: $(grep -c '^CELL' $OUT)"
awk '$1=="CELL"{n++; r=$7; s+=r; ss+=r*r; a=(r<0?-r:r); if(a>mx){mx=a; who=$2" n="$3}}
     END{printf "residual over %d cells: mean %+.4f  RMS %.4f  worst |r| %.4f at %s\n",
                n, s/n, sqrt(ss/n), mx, who}' $OUT
echo
echo "by noise-to-signal ratio:"
awk '$1=="CELL"{r=$4; e=$7;
       b=(r<0.3?"a <0.3":(r<0.6?"b 0.3-0.6":(r<1.0?"c 0.6-1.0":(r<2.0?"d 1.0-2.0":"e >2.0"))));
       n[b]++; s[b]+=e; ss[b]+=e*e}
     END{for(k in n) printf "  ratio %-10s n=%-3d mean %+.4f  RMS %.4f\n", k, n[k], s[k]/n[k], sqrt(ss[k]/n[k])}' $OUT | sort
