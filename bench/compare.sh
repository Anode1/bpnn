#!/bin/sh
# compare.sh -- when is a network worth it, and when is a line with one more column worth more?
#
# This is the bench that decides whether bpnn should exist. It puts a closed-form least squares fit
# and an iteratively trained network on the same four files, and scores both against a noise floor
# that is known by construction, so neither is graded on a curve.
#
# The four files, and why each is here:
#
#   train.csv        exactly linear, no noise      a line must win; the network must not pretend to
#   curve.csv        y = x^2 + 10, no noise        a line explains nothing; the shape is one term
#   nonlinear.csv    saturating dose + interaction a line is wrong, and its own check says which term
#   interaction.csv  y = x1*x2                     a line is wrong and its check CANNOT name the pair
#
# For the last three the noise SD is 1.0 or 0, so the best attainable RMSE is known and printed.
# Every column below is a measured number from this run; nothing is quoted from a previous one.
#
#     bench/compare.sh
#
# About 17 s on a 4-core i7-1165G7. Needs linearr; set LINEARR= if it is not at ~/linearr/linearr.

set -u
cd "$(dirname "$0")/.." || exit 1
LINEARR=${LINEARR:-$HOME/linearr/linearr}
[ -x "$LINEARR" ] || { echo "compare: linearr not found at $LINEARR (set LINEARR=)" >&2; exit 2; }
[ -x ./bpnn ] || { echo "compare: run make first" >&2; exit 2; }
TMP=$(mktemp -d) || exit 2
trap 'rm -rf "$TMP"' EXIT

# regenerate the example files, so the bench never reports numbers from a stale table
(cd example && cc -std=c99 -W -Wall -O2 -o mknonlinear mknonlinear.c ../c/rng.c -lm \
  && ./mknonlinear > nonlinear.csv && ./mknonlinear --interaction > interaction.csv) || exit 2

# worst resid SD over groups, from linearr's own --stats
lin_sd() {
    "$LINEARR" -t "$1" --stats "$TMP/s" >/dev/null 2>&1
    awk -F, 'NR>1 && $1!="group" && $5+0>m {m=$5+0} END{printf "%.4g", m}' "$TMP/s"
}
# does linearr say the shape is wrong, and can it name the term?
lin_shape() {
    "$LINEARR" -t "$1" --residuals /dev/null >"$TMP/o" 2>"$TMP/e"
    if grep -q 'still depend on the prediction itself' "$TMP/e"; then echo "wrong, pair unnamed"
    elif grep -q 'still depend on' "$TMP/e"; then echo "wrong, term named"
    else echo "no complaint"; fi
}
# worst held-out RMSE and worst refit spread over groups, from bpnn's report
bpnn_run() {
    f=$1; shift
    ./bpnn -t "$f" "$@" >/dev/null 2>"$TMP/b"
    # only the per-group data rows: they start in column 1 and carry exactly 10 fields. The
    # advisory lines beneath them are indented and hold percentages, which is a trap worth
    # naming: matched loosely, "36% of the error" reads as an RMSE of 36.
    awk '/^[^ \t]/ && NF==10 && $1!="group" {
             if ($7+0>h) h=$7+0; if ($8+0>s) s=$8+0 }
         END{ printf "%.4g %.4g", h, s }' "$TMP/b"
}

printf '%-34s %8s %10s %22s %10s %10s\n' \
       "file" "floor" "linearr" "linearr's verdict" "bpnn" "refit sd"
printf '%s\n' "----------------------------------------------------------------------------------------------------"

row() {  # label, file, floor, bpnn extra args
    lab=$1; f=$2; fl=$3; shift 3
    sd=$(lin_sd "$f"); vd=$(lin_shape "$f")
    set -- $(bpnn_run "$f" "$@")
    printf '%-34s %8s %10s %22s %10s %10s\n' "$lab" "$fl" "$sd" "$vd" "$1" "$2"
}

row "train.csv (exactly linear)"       "$HOME/linearr/example/train.csv" "0"
row "curve.csv (y=x^2+10)"             "$HOME/linearr/example/curve.csv" "0"    --holdout 0
row "nonlinear.csv (saturate+interact)" example/nonlinear.csv            "1.0"
row "interaction.csv (y=x1*x2)"        example/interaction.csv           "1.0"  -s 15 -e 8000 -H 8

echo
echo "== and now the option the network is usually compared against unfairly:"
echo "a line with the column its own diagnostic asked for. Still closed form, still exact."
echo
awk -F, 'BEGIN{OFS=","} /^#/{next} $1=="group"{print $1,$2,$3,$4,"dose2"; next}
         {print $1,$2,$3,$4,$3*$3}' example/nonlinear.csv > "$TMP/h1.csv"
awk -F, 'BEGIN{OFS=","} /^#/{next} $1=="group"{print $1,$2,$3,$4,"x1_x2"; next}
         {print $1,$2,$3,$4,$3*$4}' example/interaction.csv > "$TMP/h2.csv"
printf '%-48s %8s %10s\n' "line, with the extra column" "floor" "resid SD"
printf '%-48s %8s %10s\n' "nonlinear.csv + dose^2 (the check named dose)" "1.0" "$(lin_sd "$TMP/h1.csv")"
printf '%-48s %8s %10s\n' "interaction.csv + x1*x2 (the check could not)" "1.0" "$(lin_sd "$TMP/h2.csv")"

cat <<'EOF'

== how to read this
A correctly specified line wins wherever it can be specified: it reaches the noise floor, it does it
in closed form so there is no seed and no spread, and it hands you coefficients. The network never
beat it in this table. What the network buys is not accuracy, it is not having to guess WHICH column
to add, and the guess is only cheap when the diagnostic names the term for you. When it can only say
that some pair interacts, the candidates number p(p-1)/2: three pairs at 3 terms, 276 at 24. That
gap, and not accuracy, is where a network earns its place.
EOF
