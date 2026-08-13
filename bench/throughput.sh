#!/bin/sh
# throughput.sh -- how fast does it fit, and how fast does it score?
#
# Everything here is measured on the machine it is run on and printed with that machine named,
# because a rate without a machine is not a measurement. Best of three, so one scheduling
# accident does not become the published number.
#
#     bench/throughput.sh [rows]        # rows defaults to 200000
#
# The fit rate is quoted as row-updates a second: rows x epochs x refits, divided by the wall
# clock. That is the unit that stays constant when any one of the three changes, and it is the
# one to use when sizing a run. The score rate is cases a second through the -c path, which is
# a different program shape: one model load, then a forward pass per case.

set -u
cd "$(dirname "$0")/.." || exit 1
[ -x ./bpnn ] || { echo "throughput: run make first" >&2; exit 2; }

N=${1:-200000}
TMP=$(mktemp -d) || exit 2
trap 'rm -rf "$TMP"' EXIT

best3() {  # runs a command three times, prints the shortest wall clock
    b=""
    for _ in 1 2 3; do
        t=$( { /usr/bin/time -f '%e' "$@" >/dev/null 2>>"$TMP/t"; } 2>&1; tail -1 "$TMP/t" )
        : > "$TMP/t"
        case $b in "") b=$t ;; *) b=$(awk -v a="$b" -v c="$t" 'BEGIN{print (c<a)?c:a}') ;; esac
    done
    echo "$b"
}

gen() {  # rows, terms, groups, file
    awk -v n="$1" -v p="$2" -v g="$3" 'BEGIN{
        srand(7)
        printf "group,y"
        for (j = 1; j <= p; j++) printf ",x%d", j
        printf "\n"
        for (i = 0; i < n; i++) {
            printf "%03d", i % g
            y = 0; line = ""
            for (j = 1; j <= p; j++) { v = rand() * 10; y += v / (1 + j); line = line sprintf(",%.4f", v) }
            printf ",%.4f%s\n", y + 0.4 * (rand() - 0.5), line
        }
    }' > "$4"
}

echo "machine: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
echo "compiler: $(cc --version | head -1)"
echo

# ---- fitting -------------------------------------------------------------
# Epochs and refits are pinned so that the only thing moving is the row count, and --patience 0
# so that a fit that converges early does not make the rate look better than the engine is.
E=20; S=2
printf '%-12s %8s %8s %10s %12s %12s\n' "rows" "terms" "groups" "wall (s)" "rows/s" "updates/s"
for rows in $((N / 10)) $N $((N * 5)); do
    gen "$rows" 4 4 "$TMP/f.csv"
    w=$(best3 ./bpnn -t "$TMP/f.csv" -e $E -s $S --patience 0)
    printf '%-12s %8s %8s %10s %12s %12s\n' "$rows" 4 4 "$w" \
        "$(awk -v r="$rows" -v w="$w" 'BEGIN{printf "%.3g", r/w}')" \
        "$(awk -v r="$rows" -v w="$w" -v e=$E -v s=$S 'BEGIN{printf "%.3g", r*e*s/w}')"
done

echo
printf '%-12s %8s %8s %10s %12s %12s\n' "rows" "terms" "groups" "wall (s)" "rows/s" "updates/s"
for p in 2 8 24; do
    gen "$N" "$p" 4 "$TMP/f.csv"
    w=$(best3 ./bpnn -t "$TMP/f.csv" -e $E -s $S --patience 0)
    printf '%-12s %8s %8s %10s %12s %12s\n' "$N" "$p" 4 "$w" \
        "$(awk -v r="$N" -v w="$w" 'BEGIN{printf "%.3g", r/w}')" \
        "$(awk -v r="$N" -v w="$w" -v e=$E -v s=$S 'BEGIN{printf "%.3g", r*e*s/w}')"
done

# ---- scoring -------------------------------------------------------------
# One model, one case, over and over: this is the latency of the whole program, process start
# included, which is what a pipeline stage actually pays.
echo
gen 4000 4 4 "$TMP/s.csv"
./bpnn -t "$TMP/s.csv" -e 50 -s 2 > "$TMP/m.txt" 2>/dev/null
w=$(best3 ./bpnn -c "$TMP/m.txt" 000 x1=1 x2=2 x3=3 x4=4)
printf 'one case, cold start: %s s per invocation\n' "$w"
echo "  (the model file is read every time; there is no server mode and no batch scoring path)"
