#!/bin/sh
# elos.sh -- both programs on a table shaped like the 2011 length-of-stay model.
#
# 24 binary procedure indicators, 8 groups, 600 rows each, per-group intercepts: the shape the
# deployed predictor had. What changes between the three files is the truth underneath, because
# the only question worth settling is which program to reach for, and that turns on whether the
# truth is a line. example/mkelos.c generates them and states the noise floor in each header.
#
#     bench/elos.sh
#
# Needs linearr; set LINEARR= if it is not at ~/linearr/linearr. About two minutes.

set -u
cd "$(dirname "$0")/.." || exit 1
LINEARR=${LINEARR:-$HOME/linearr/linearr}
[ -x "$LINEARR" ] || { echo "elos: linearr not found at $LINEARR (set LINEARR=)" >&2; exit 2; }
[ -x ./bpnn ] || { echo "elos: run make first" >&2; exit 2; }
TMP=$(mktemp -d) || exit 2
trap 'rm -rf "$TMP"' EXIT

(cd example && cc -std=c99 -W -Wall -O2 -o mkelos mkelos.c ../c/rng.c -lm \
  && for m in linear saturating interaction; do ./mkelos --$m > elos-$m.csv; done) || exit 2

lin_sd() {
    "$LINEARR" -t "$1" --stats "$TMP/s" >/dev/null 2>&1
    awk -F, 'NR>1 && $1!="group" && $5+0>m {m=$5+0} END{printf "%.3f", m}' "$TMP/s"
}
lin_warn() {
    "$LINEARR" -t "$1" --residuals /dev/null >/dev/null 2>"$TMP/e"
    if grep -q 'still depend on the prediction itself' "$TMP/e"; then echo "pair unnamed"
    elif grep -q 'still depend on' "$TMP/e"; then echo "term named"
    else echo "SILENT"; fi
}
# worst held-out over the groups, which is the figure a deployment lives with
# At the defaults, because a table fitted at flags the reader will not type is a table about
# something else. The saturating row was 6% better under -H 8 -e 400 -s 3 than ./bpnn -t gives.
bp() { ./bpnn -t "$1" 2>&1 >/dev/null \
       | awk '$1 ~ /^00/ {if ($7+0>m) m=$7+0} END{printf "%.3f", m}'; }

printf '%-14s %7s %10s %15s %12s\n' "truth" "floor" "linearr" "its verdict" "bpnn"
printf '%s\n' "--------------------------------------------------------------------"
for m in linear saturating interaction; do
    printf '%-14s %7s %10s %15s %12s\n' "$m" "1.0" \
        "$(lin_sd example/elos-$m.csv)" "$(lin_warn example/elos-$m.csv)" \
        "$(bp example/elos-$m.csv)"
done

# The interaction file again, with the pair handed to the line. This is the comparison that
# decides the case, and it is the one a network is usually not put through.
echo
awk -F, 'BEGIN{OFS=","} /^#/{next} $1=="group"{print $0,"p06_p13"; next} {print $0,$9*$16}' \
    example/elos-interaction.csv > "$TMP/named.csv"
printf 'interaction, with the pair added as a column: %s\n' "$(lin_sd "$TMP/named.csv")"

cat <<'EOF'

== how to read this
On the linear truth the line reaches the noise floor and the network does not, which is the
expected order and the reason to fit a line first. On the saturating truth the line is wrong, its
own check says so, and the network nearly reaches the floor: that is the escalation working.

The interaction row is the one to take seriously. The line is wrong, the network barely improves
on it at any capacity, and linearr's residual check says NOTHING, so escalate.sh would report
that a line was enough. Handed the pair, the line reaches the floor immediately. With 24 binary
terms and an 18% prevalence each, only about 3% of rows carry both procedures, so the effect is
present in a few dozen rows per group -- too few for the network to find and too subtle for the
diagnostic to see. On tables of this shape, neither program will tell you the pair; a domain
expert naming it is worth more than either.
EOF
