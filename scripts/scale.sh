#!/bin/sh
# scale.sh -- does --stream hold its memory when the rows go up tenfold?
#
# The claim --stream makes is that memory is the networks plus the shuffle windows, and that no
# figure in it contains a row count. This is the check: fit the same model over N rows and over
# 10N, and compare peak resident memory. The default path is measured beside it, which is where
# the row store shows up.
#
#     scripts/scale.sh [N]          # N defaults to 100000
#
# Epochs are set to 2. This measures memory, not fit quality.

set -u
cd "$(dirname "$0")/.." || exit 1
[ -x ./bpnn ] || { echo "scale: run make first" >&2; exit 2; }

N=${1:-100000}
TMP=$(mktemp -d) || exit 2
trap 'rm -rf "$TMP"' EXIT

# peak resident set size in kB, from the shell's own time builtin if GNU time is absent
peak() {
    /usr/bin/time -f '%M' "$@" 2>&1 >/dev/null | tail -1
}

gen() {  # rows, file: four groups, two terms, a saturating response
    awk -v n="$1" 'BEGIN{
        srand(7); print "group,y,dose,age"
        for (i = 0; i < n; i++) {
            g = sprintf("%03d", i % 4 + 1)
            d = rand() * 10; a = 20 + rand() * 70
            y = 12 * d / (2 + d) + 0.05 * a + 0.4 * (rand() - 0.5)
            printf "%s,%.4f,%.4f,%.2f\n", g, y, d, a
        }
    }' > "$2"
}

echo "generating $N and $((N * 10)) rows"
gen "$N"            "$TMP/small.csv"
gen "$((N * 10))"   "$TMP/big.csv"

printf '%-38s %12s %12s\n' "" "$N rows" "$((N * 10)) rows"
printf '%-38s %12s %12s\n' "--stream, peak RSS (kB)" \
       "$(peak ./bpnn -t "$TMP/small.csv" --stream -e 2)" \
       "$(peak ./bpnn -t "$TMP/big.csv"   --stream -e 2)"
printf '%-38s %12s %12s\n' "default, peak RSS (kB)" \
       "$(peak ./bpnn -t "$TMP/small.csv" -e 2)" \
       "$(peak ./bpnn -t "$TMP/big.csv"   -e 2)"

echo
echo "The --stream row should not move with the rows. The default row should scale with them:"
echo "it holds every row at $(( (64 + 1) * 8 + 8 )) bytes, which ./bpnn --footprint prints."
