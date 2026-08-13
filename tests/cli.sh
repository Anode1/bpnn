#!/bin/sh
# cli.sh: black-box tests: the built binary, driven through a shell, the way a user meets it.
#
# `make ut` cannot see any of this. It links the engine and calls functions, so it never sees an
# exit code, never sees the usage text, and never sees a refusal message. Every guard in the CSV
# reader and every option error is therefore only testable from here.
#
#   make cliut          # or: sh tests/cli.sh
set -e

root=$(cd "$(dirname "$0")/.." && pwd)
# BPNN= points the same checks at another build, which is how they run under the sanitizers:
# the reader's guards are the newest code in the program and none of them is reached by `make ut`.
bin=${BPNN:-$root/bpnn}
[ -x "$bin" ] || { echo "build first: make" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$root"

pass=0; fail=0
ok()   { pass=$((pass+1)); }
no()   { fail=$((fail+1)); echo "  FAIL $1"; }
check(){ if [ "$2" = "$3" ]; then ok; else no "$1: expected [$3], got [$2]"; fi; }
# the first line of stderr, which is where every refusal starts
firstline(){ printf '%s' "$1" | head -1; }

# Fits here are deliberately small: these tests are about what the program refuses and what it
# says, not about how well it fits. FAST keeps the whole suite inside a few seconds.
FAST="-e 50 -s 2"

# A small well-formed file, and one with two groups.
cat > "$tmp/ok.csv" <<'EOF'
group,y,x
A,11,1
A,14,2
A,19,3
A,26,4
A,35,5
A,46,6
A,59,7
A,74,8
EOF

# ---------------------------------------------------------------- invocation

out=$("$bin" 2>&1 || true); rc=0; "$bin" >/dev/null 2>&1 || rc=$?
check "a bare run prints usage" "$(firstline "$out")" \
      "bpnn -- a backpropagation network for tabular data, where a line is not enough"
check "and exits 2, not 0"      "$rc" "2"

out=$("$bin" --help); rc=$?
check "--help exits 0"          "$rc" "0"
check "--help goes to stdout"   "$(printf '%s' "$out" | grep -c '^  bpnn ')" "3"
check "--help lists the options" "$(printf '%s' "$out" | grep -c '^  -')" "6"

out=$("$bin" --selftest); rc=$?
check "--selftest exits 0"      "$rc" "0"
check "--selftest says so"      "$(printf '%s' "$out" | tail -1)" "self-test PASSED"
check "--selftest has no FAIL"  "$(printf '%s' "$out" | grep -c FAIL || true)" "0"

# ------------------------------------------------------------------- options
# Every one of these used to be accepted silently: an unknown option was ignored, a missing
# value was ignored with it, and a junk number became zero.

badopt() {  # label, expected first line of stderr, then the arguments
    lab=$1; want=$2; shift 2
    set +e
    out=$("$bin" "$@" 2>&1 >/dev/null); rc=$?
    set -e
    check "$lab" "$(firstline "$out")" "$want"
    check "$lab exit" "$rc" "2"
}

badopt "an unknown option"   "bpnn: --typo is not an option this program has. Try --help." \
       -t "$tmp/ok.csv" --typo
badopt "a missing value"     "bpnn: --holdout needs a value after it" \
       -t "$tmp/ok.csv" --holdout
badopt "a non-numeric -H"    "bpnn: -H six is not a number" \
       -t "$tmp/ok.csv" -H six
badopt "an unknown -a"       "bpnn: -a bogus is not an activation. Use sigmoid, tanh or relu." \
       -t "$tmp/ok.csv" -a bogus
badopt "a holdout above 1"   "bpnn: --holdout is a fraction of the rows, so it must be in [0, 1)." \
       -t "$tmp/ok.csv" --holdout 1.5
badopt "a negative holdout"  "bpnn: --holdout is a fraction of the rows, so it must be in [0, 1)." \
       -t "$tmp/ok.csv" --holdout -0.5
badopt "zero hidden units"   "bpnn: hidden units, epochs and refits must each be at least 1" \
       -t "$tmp/ok.csv" -H 0
badopt "a zero learning rate" "bpnn: the learning rate must be positive and the momentum in [0, 1)" \
       -t "$tmp/ok.csv" -r 0
badopt "a momentum of 1"     "bpnn: the learning rate must be positive and the momentum in [0, 1)" \
       -t "$tmp/ok.csv" -m 1

# ------------------------------------------------------------- what it reads
# A file the reader must refuse, named by what is wrong with it. Every one of these produced a
# fitted model and exit 0 before the guards existed; the nan case produced a model whose every
# weight was nan.

bad() {  # label, expected first line of stderr, file contents
    printf '%b' "$3" > "$tmp/bad.csv"
    set +e
    out=$("$bin" -t "$tmp/bad.csv" $FAST 2>&1 >/dev/null); rc=$?
    set -e
    check "$1" "$(firstline "$out")" "$2"
    check "$1 exit" "$rc" "1"
}

bad "a nan term" \
    "bpnn: the term 'x' is 'nan' on line 3. A nan or an infinity reaches every" \
    'group,y,x\nA,1,1\nA,2,nan\n'
bad "an infinite response" \
    "bpnn: 'y', the value being predicted, is 'inf' on line 3. A nan or an infinity reaches every" \
    'group,y,x\nA,1,1\nA,inf,3\n'
bad "a non-numeric term" \
    "bpnn: the term 'x' is 'oops' on line 3, which is not a number." \
    'group,y,x\nA,1,1\nA,2,oops\n'
bad "an empty response" \
    "bpnn: 'y', the value being predicted, is empty on line 3. An empty field is not a zero, and" \
    'group,y,x\nA,1,1\nA,,3\n'
bad "an empty term" \
    "bpnn: the term 'x' is empty on line 3. An empty field is not a zero, and" \
    'group,y,x\nA,1,1\nA,2,\n'
bad "a short row" \
    "bpnn: line 3 has 2 fields; the header declared 3 (the group," \
    'group,y,x\nA,1,1\nA,2\n'
bad "a long row" \
    "bpnn: line 3 has 4 fields; the header declared 3 (the group," \
    'group,y,x\nA,1,1\nA,2,3,4\n'
bad "a term named twice" \
    "bpnn: the header names the term 'x' twice, so a case" \
    'group,y,x,x\nA,1,1,2\n'
bad "a header of two columns" \
    "bpnn: the header on line 1 needs a group column, the value" \
    'group,y\nA,1\n'
bad "an empty group code" \
    "bpnn: the group is empty on line 2." \
    'group,y,x\n,1,1\n'

set +e
out=$("$bin" -t /dev/null $FAST 2>&1 >/dev/null); rc=$?
set -e
check "an empty file"      "$(firstline "$out")" "bpnn: /dev/null has no header line."
check "an empty file exit" "$rc" "1"

printf 'group,y,x\n' > "$tmp/hdr.csv"
set +e
out=$("$bin" -t "$tmp/hdr.csv" $FAST 2>&1 >/dev/null); rc=$?
set -e
check "a header and nothing else" "$(firstline "$out")" "bpnn: $tmp/hdr.csv has a header and no data rows"
check "a header and nothing else exit" "$rc" "1"

# 513 groups, one row each: refused at the ceiling, before any fitting is attempted.
{ echo 'group,y,x'; i=0; while [ $i -lt 513 ]; do echo "g$i,1,1"; i=$((i+1)); done; } > "$tmp/many.csv"
set +e
out=$("$bin" -t "$tmp/many.csv" $FAST 2>&1 >/dev/null); rc=$?
set -e
check "more groups than the build holds" "$(firstline "$out")" "bpnn: more than 512 groups."
check "and it says so before fitting"    "$rc" "1"

# A group name too long to store would be truncated, and two long names sharing a prefix would
# then merge into one fit with nothing printed.
long=$(printf 'g%.0s' $(seq 1 70))
printf 'group,y,x\n%s,1,1\n' "$long" > "$tmp/long.csv"
set +e
out=$("$bin" -t "$tmp/long.csv" $FAST 2>&1 >/dev/null); rc=$?
set -e
check "an over-long group name" "$(firstline "$out")" \
      "bpnn: the group on line 2 is longer than 63 characters: '$long'."

# Comments and blank lines are ordinary, not errors.
printf '# a note\n\ngroup,y,x\n# another\nA,1,1\nA,2,2\nA,3,3\nA,4,4\nA,5,5\n' > "$tmp/cmt.csv"
set +e
"$bin" -t "$tmp/cmt.csv" $FAST >/dev/null 2>&1; rc=$?
set -e
check "comments and blank lines are skipped" "$rc" "0"

check "-t - fits from stdin" \
      "$("$bin" -t - $FAST < "$tmp/ok.csv" 2>/dev/null | grep -c '^GROUP A')" "1"

# ---------------------------------------------------------------- what it fits

"$bin" -t "$tmp/ok.csv" $FAST > "$tmp/m1.txt" 2>"$tmp/r1.txt"
check "the fit exits 0"          "$?" "0"
check "the report has a header"  "$(head -1 "$tmp/r1.txt" | awk '{print $1, $NF}')" "group floor"
check "one report row per group" "$(grep -c '^A ' "$tmp/r1.txt")" "1"
check "the model names the response" "$(grep '^response' "$tmp/m1.txt")" "response y"
check "the model names the terms"    "$(grep '^terms' "$tmp/m1.txt")" "terms 1 x"
check "the model carries its diagnostics" "$(grep -c '^diag ' "$tmp/m1.txt")" "1"

# The seed is fixed, so the same input gives the same model byte for byte. This is the property
# every archived number in this tree depends on.
"$bin" -t "$tmp/ok.csv" $FAST > "$tmp/m2.txt" 2>/dev/null
check "the same input gives the same model" "$(cmp -s "$tmp/m1.txt" "$tmp/m2.txt" && echo same)" "same"

# A group with too few rows is skipped and named, and the others are still fitted.
{ cat "$tmp/ok.csv"; printf 'B,1,1\nB,2,2\n'; } > "$tmp/two.csv"
out=$("$bin" -t "$tmp/two.csv" $FAST 2>&1 >/dev/null)
check "a group too small to fit is named" \
      "$(firstline "$out")" "bpnn: group B has 2 rows; a network needs more than that to say"
check "and the other group is still fitted" \
      "$("$bin" -t "$tmp/two.csv" $FAST 2>/dev/null | grep -c '^GROUP A')" "1"

# A response that never varies makes every error near zero, which looks like a perfect fit.
printf 'group,y,x\nA,7,1\nA,7,2\nA,7,3\nA,7,4\nA,7,5\nA,7,6\n' > "$tmp/flat.csv"
out=$("$bin" -t "$tmp/flat.csv" $FAST 2>&1 >/dev/null)
check "a constant response is reported" \
      "$(printf '%s' "$out" | grep -c 'NEVER VARIES')" "1"

# --holdout 0 removes the only measurement of generalization, and says so.
out=$("$bin" -t "$tmp/ok.csv" $FAST --holdout 0 2>&1 >/dev/null)
check "--holdout 0 warns" "$(printf '%s' "$out" | grep -c 'NO HELD-OUT ROWS')" "1"

# --------------------------------------------------------------- what it scores

M=$tmp/m1.txt
check "a case is scored" "$("$bin" -c "$M" A x=3 | head -1 | cut -d= -f1)" "A y "
check "the fit-time error comes with it" \
      "$("$bin" -c "$M" A x=3 | sed -n 2p | cut -d' ' -f1-4)" "held-out RMSE at fit"

badscore() {  # label, expected first line, then arguments after -c MODEL
    lab=$1; want=$2; shift 2
    set +e
    out=$("$bin" -c "$M" "$@" 2>&1 >/dev/null); rc=$?
    set -e
    check "$lab" "$(firstline "$out")" "$want"
    check "$lab exit" "$rc" "2"
}

badscore "a term left out"    "bpnn: this model has 1 term and 1 was not given: x" A
badscore "a non-numeric case" "bpnn: x=abc is not a finite number" A x=abc
badscore "an infinite case"   "bpnn: x=inf is not a finite number" A x=inf
badscore "an unknown term"    "bpnn: 'z' is not a term in this model" A z=1
badscore "an unknown group"   "bpnn: group 'Q' is not in this model" Q x=1
badscore "no group named"     "bpnn: name the group to score, e.g. ./bpnn -c $M A x=3" x=1

set +e
out=$("$bin" -c "$tmp/ok.csv" A x=1 2>&1 >/dev/null); rc=$?
set -e
check "a CSV offered as a model" "$(firstline "$out")" "bpnn: $tmp/ok.csv is not a bpnn model"
check "a CSV offered as a model exits 1" "$rc" "1"

# The extrapolation check: inside the training range it stays quiet, outside it does not.
check "inside the range, no warning" \
      "$("$bin" -c "$M" A x=4 | grep -c 'OUTSIDE THE TRAINING RANGE' || true)" "0"
check "outside it, the term and the distance" \
      "$("$bin" -c "$M" A x=99 | grep -c 'OUTSIDE THE TRAINING RANGE')" "1"
check "and what that means" \
      "$("$bin" -c "$M" A x=99 | grep -c 'does not extrapolate')" "1"

# --------------------------------------------------- numerics, not just parsing
# A suite whose inputs are all around 1 is not a test of numerics. These three cases are the
# ones linearr's suite grew after a review found a wrong number printed with exit 0 in each.
# Here the inputs are scaled into [0, 1] per term and the response into [0.1, 0.9], so the fit
# should be indifferent to units and to an offset; these assert that it is, rather than assume.

awk -F, 'NR==1{print;next}{printf "%s,%s,%.10g\n",$1,$2,$3*1000000}' "$tmp/ok.csv" > "$tmp/units.csv"
awk -F, 'NR==1{print;next}{printf "%s,%.10g,%s\n",$1,$2+100000000,$3}' "$tmp/ok.csv" > "$tmp/offset.csv"
"$bin" -t "$tmp/units.csv"  $FAST > "$tmp/mu.txt" 2>/dev/null
"$bin" -t "$tmp/offset.csv" $FAST > "$tmp/mo.txt" 2>/dev/null

check "a term in millions fits identically" \
      "$(grep '^diag' "$tmp/mu.txt")" "$(grep '^diag' "$tmp/m1.txt")"
check "and predicts identically" \
      "$("$bin" -c "$tmp/mu.txt" A x=3000000 | head -1)" "$("$bin" -c "$M" A x=3 | head -1)"
check "a response offset by 1e8 fits identically" \
      "$(grep '^diag' "$tmp/mo.txt")" "$(grep '^diag' "$tmp/m1.txt")"
# and the prediction must still show the part that moves: %g's six digits would print 1e+08
# for every case in this group, which is the offset and none of the answer.
check "and the prediction still resolves the response" \
      "$("$bin" -c "$tmp/mo.txt" A x=3 | head -1 | sed 's/.*= *//' | cut -c1-9)" \
      "100000017"

# Two columns that differ in the sixth decimal: the fit must not produce a nan or refuse.
{ echo 'group,y,x1,x2'
  i=1; while [ $i -le 12 ]; do
      echo "A,$((i*3+1)),$i,$i.000001"; i=$((i+1)); done; } > "$tmp/coll.csv"
out=$("$bin" -t "$tmp/coll.csv" $FAST 2>&1 >/dev/null); rc=$?
check "near-identical columns still fit" "$rc" "0"
check "and produce no nan"               "$(printf '%s' "$out" | grep -ci nan || true)" "0"

# ------------------------------------------------------------------------ end

echo "$((pass+fail)) checks, $fail failed"
[ "$fail" -eq 0 ]
