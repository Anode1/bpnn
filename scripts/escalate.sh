#!/bin/sh
# escalate.sh -- fit a line; if the line is the wrong shape, fit a network.
#
# The two programs are a pair, and this is the pipeline that joins them. linearr fits straight
# lines and, with --residuals, checks whether what is left over still depends on a term. When it
# does, a line is the wrong shape and linearr says so and stops, because a line is all it has.
# That warning is the trigger: this script reads it and escalates to bpnn, which can fit a curve
# and cannot read you a coefficient.
#
# The order matters and is not arbitrary. The line is tried FIRST because when it is right it is
# better in every way that matters: exact rather than approximate, one answer rather than a
# distribution over random starts, and coefficients you can hand to somebody. The network is what
# you fall back to, not what you reach for. Escalating on a measured diagnostic rather than on
# taste is the whole point; "we used a neural network" is not a modelling decision.
#
#     scripts/escalate.sh data.csv [-y RESPONSE]
#
# Exit status is 0 if a line was enough, 1 if it was not and the network was fitted, 2 on error.
# Both models are written next to the input as <base>.linearr.csv and <base>.bpnn.txt.

set -u
LINEARR=${LINEARR:-$HOME/linearr/linearr}
BPNN=${BPNN:-$(dirname "$0")/../bpnn}

if [ $# -lt 1 ]; then
    echo "usage: $0 data.csv [linearr options]" >&2
    exit 2
fi
DATA=$1; shift
[ -f "$DATA" ] || { echo "escalate: no such file: $DATA" >&2; exit 2; }
[ -x "$LINEARR" ] || { echo "escalate: linearr not found at $LINEARR (set LINEARR=)" >&2; exit 2; }
[ -x "$BPNN" ] || { echo "escalate: bpnn not found at $BPNN (run make)" >&2; exit 2; }

BASE=${DATA%.csv}
LMODEL=$BASE.linearr.csv
NMODEL=$BASE.bpnn.txt
LERR=$(mktemp) || exit 2
trap 'rm -f "$LERR"' EXIT

echo "== step 1: fit a straight line"
"$LINEARR" -t "$DATA" --residuals /dev/null --stats - "$@" > "$LMODEL" 2> "$LERR"
rc=$?
sed 's/^/   /' "$LERR"
if [ $rc -ne 0 ]; then
    echo "escalate: linearr failed; not escalating, because a failure to fit is not evidence" >&2
    echo "that the shape is wrong. Fix the input first." >&2
    exit 2
fi

# KNOWN WEAKNESS, stated because it fails in the dangerous direction. The trigger below is a text
# match on linearr's warning, since as of this writing linearr exits 0 whether or not the shape is
# wrong and its --stats table has no column for it. So if that wording ever changes, this script
# stops escalating and says a line was enough, which is the wrong answer given silently. The fix
# belongs upstream: a shape column in --stats, or a distinct exit status. Until then, prove the
# diagnostic ran at all before trusting its silence.
if ! grep -q '^fit:' "$LERR"; then
    echo "escalate: linearr printed no 'fit:' summary, so its residual check cannot be assumed to" >&2
    echo "have run. Refusing to conclude that a line was enough from an absence of output." >&2
    exit 2
fi

# The trigger: the residuals still depend on a term, or on the prediction itself, after the line is
# subtracted. Either is a wrong shape, and a wrong shape is the one thing that justifies a curve.
if ! grep -q 'residuals still depend on' "$LERR"; then
    echo
    echo "== a line was enough"
    echo "linearr found no leftover dependence on any term or on the prediction, so there is no"
    echo "measured reason to fit a curve here. The coefficients are in $LMODEL and you can read"
    echo "them. Least squares solves in closed form, so that answer is exact and has no seed;"
    echo "escalating anyway would trade those away and buy nothing this data can show."
    exit 0
fi

echo
echo "== step 2: the line is the wrong shape"
grep 'residuals still depend on' "$LERR" | sed 's/^/   /'
echo

# Which kind of wrong shape it is decides what you should actually do, and bench/compare.sh
# measures the difference. When the check NAMES a term, adding that one column and refitting keeps
# you in closed form and reached the noise floor on both test files, beating the network. When it
# can only say that some pair interacts, the candidate columns number p(p-1)/2 and there is nothing
# to add without guessing; that is where the network stops being a shortcut and starts being the
# cheaper option.
if grep -q 'still depend on the prediction itself' "$LERR"; then
    echo "   The check cannot name the pair: it sees an interaction or a curve that no single term"
    echo "   shows. With p terms there are p(p-1)/2 candidate columns to guess among, so this is"
    echo "   the case where fitting a network is cheaper than searching. Escalating."
else
    echo "   The check NAMED the term above. Before accepting a network, try adding that term's"
    echo "   square as a column and refitting: on bench/compare.sh's files that stayed in closed"
    echo "   form, kept readable coefficients, and matched or beat the network. Escalating anyway"
    echo "   so you have both to compare, but the line with one more column is the better bet."
fi
echo
"$BPNN" -t "$DATA" "$@" > "$NMODEL" || { echo "escalate: bpnn failed" >&2; exit 2; }

echo
echo "== both models are on disk"
echo "   line:    $LMODEL   (readable coefficients, wrong shape here)"
echo "   network: $NMODEL   (fits the shape, no coefficients to read)"
echo
echo "Read bpnn's held-out column, not its training column, before believing the improvement,"
echo "and read the refit spread beside it: if the spread is a large share of the error then one"
echo "fit of this configuration has not pinned down how good it is."
exit 1
