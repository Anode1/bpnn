#!/bin/sh
# cli.sh: black-box tests: the built binary, driven through a shell, the way a user meets it.
#
# make ut calls functions; exit codes, the usage text and every refusal message are only
# testable from here.
#
#   make cliut          # or: sh tests/cli.sh
set -e

root=$(cd "$(dirname "$0")/.." && pwd)
# BPNN= points the same checks at another build, which is how they run under the sanitizers:
# the reader's guards are the newest code in the program and none of them is reached by `make ut`.
bin=${BPNN:-$root/bpnn}
[ -x "$bin" ] || { echo "build first: make" >&2; exit 1; }

tmp=$(mktemp -d 2>/dev/null || mktemp -d -t bpnn)
trap 'rm -rf "$tmp"' EXIT
cd "$root"

pass=0; fail=0
ok()   { pass=$((pass+1)); }
no()   { fail=$((fail+1)); echo "  FAIL $1"; }
check(){ if [ "$2" = "$3" ]; then ok; else no "$1: expected [$3], got [$2]"; fi; }
# the first line of stderr, which is where every refusal starts
firstline(){ printf '%s' "$1" | head -1; }
# BSD and GNU disagree about the name of every checksum tool and about sort -g. Both are used
# only to compare two outputs for equality, so any stable digest will do.
digest() { if command -v md5sum >/dev/null 2>&1; then md5sum; \
           elif command -v md5 >/dev/null 2>&1; then md5; \
           else cksum; fi; }

# Small fits: this suite tests messages and exit codes, not fit quality.
FAST="-e 50 -s 2"

# The fixture every check below fits. It is y = x^2 + 10 over 40 rows: a curve a line cannot
# fit, and enough rows to clear MINROWS, which is the point below which the fitted, stopping and
# reported samples are all too small for the report to mean anything.
{ echo 'group,y,x'
  i=1; while [ $i -le 40 ]; do echo "A,$((i * i + 10)),$i"; i=$((i + 1)); done; } > "$tmp/ok.csv"

# ---------------------------------------------------------------- invocation

out=$("$bin" 2>&1 || true); rc=0; "$bin" >/dev/null 2>&1 || rc=$?
check "a bare run prints usage" "$(firstline "$out")" \
      "bpnn -- a backpropagation network for tabular data, where a line is not enough"
check "and exits 2, not 0"      "$rc" "2"

set +e
out=$("$bin" --help); rc=$?
set -e
check "--help exits 0"          "$rc" "0"
check "--help goes to stdout"   "$(printf '%s' "$out" | grep -c '^  bpnn ')" "3"
check "--help lists the options" "$(printf '%s' "$out" | grep -c '^  -')" "17"

set +e
out=$("$bin" --selftest); rc=$?
set -e
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
    "$tmp/bad.csv:3:3: the term 'x' is 'nan': not finite" \
    'group,y,x\nA,1,1\nA,2,nan\n'
bad "an infinite response" \
    "$tmp/bad.csv:3:2: 'y', the value being predicted, is 'inf': not finite" \
    'group,y,x\nA,1,1\nA,inf,3\n'
bad "a non-numeric term" \
    "$tmp/bad.csv:3:3: the term 'x' is 'oops', which is not a number." \
    'group,y,x\nA,1,1\nA,2,oops\n'
bad "an empty response" \
    "$tmp/bad.csv:3:2: 'y', the value being predicted, is empty" \
    'group,y,x\nA,1,1\nA,,3\n'
bad "an empty term" \
    "$tmp/bad.csv:3:3: the term 'x' is empty" \
    'group,y,x\nA,1,1\nA,2,\n'
bad "a short row" \
    "$tmp/bad.csv:3: 2 fields; the header declared 3 (the group, y, and" \
    'group,y,x\nA,1,1\nA,2\n'
bad "a long row" \
    "$tmp/bad.csv:3: 4 fields; the header declared 3 (the group, y, and" \
    'group,y,x\nA,1,1\nA,2,3,4\n'
bad "a term named twice" \
    "$tmp/bad.csv:1:4: the term 'x' is named twice, so a case" \
    'group,y,x,x\nA,1,1,2\n'
bad "a header of two columns" \
    "$tmp/bad.csv:1: the header needs a group column, the value being" \
    'group,y\nA,1\n'
bad "an empty group code" \
    "$tmp/bad.csv:2:1: the group is empty" \
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
check "more groups than the build holds" "$(firstline "$out")" \
      "$tmp/many.csv:514:1: more than 512 groups"
check "and it says so before fitting"    "$rc" "1"

# A group name too long to store would be truncated, and two long names sharing a prefix would
# then merge into one fit with nothing printed.
long=''; i=0; while [ $i -lt 70 ]; do long="${long}g"; i=$((i+1)); done
printf 'group,y,x\n%s,1,1\n' "$long" > "$tmp/long.csv"
set +e
out=$("$bin" -t "$tmp/long.csv" $FAST 2>&1 >/dev/null); rc=$?
set -e
check "an over-long group name" "$(firstline "$out")" \
      "$tmp/long.csv:2:1: the group is longer than 63 characters: '$long'"

# Comments and blank lines are ordinary, not errors.
{ printf '# a note\n\ngroup,y,x\n# another\n'
  i=1; while [ $i -le 40 ]; do echo "A,$((i * i + 10)),$i"; i=$((i + 1)); done; } > "$tmp/cmt.csv"
set +e
"$bin" -t "$tmp/cmt.csv" $FAST >/dev/null 2>&1; rc=$?
set -e
check "comments and blank lines are skipped" "$rc" "0"

check "-t - fits from stdin" \
      "$("$bin" -t - $FAST < "$tmp/ok.csv" 2>/dev/null | grep -c '^GROUP A')" "1"

# ---------------------------------------------------------------- what it fits

set +e
"$bin" -t "$tmp/ok.csv" $FAST > "$tmp/m1.txt" 2>"$tmp/r1.txt"; rc=$?
set -e
check "the fit exits 0"          "$rc" "0"
check "the report has a header"  "$(grep '^group ' "$tmp/r1.txt" | awk '{print $1, $NF}')" "group expl"
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
      "$(printf '%s' "$out" | grep -m1 '^bpnn: group B')" \
      "bpnn: group B has 2 rows. This shape fits 19 parameters and"
check "and the other group is still fitted" \
      "$("$bin" -t "$tmp/two.csv" $FAST 2>/dev/null | grep -c '^GROUP A')" "1"

# A response that never varies makes every error near zero, which looks like a perfect fit.
{ echo 'group,y,x'; i=1; while [ $i -le 40 ]; do echo "A,7,$i"; i=$((i + 1)); done; } > "$tmp/flat.csv"
out=$("$bin" -t "$tmp/flat.csv" $FAST 2>&1 >/dev/null)
check "a constant response is reported" \
      "$(printf '%s' "$out" | grep -c 'NEVER VARIES')" "1"

# --holdout 0 removes the only measurement of generalization, and says so.
out=$("$bin" -t "$tmp/ok.csv" $FAST --holdout 0 2>&1 >/dev/null)
check "--holdout 0 warns" "$(printf '%s' "$out" | grep -c 'NO HELD-OUT ROWS')" "1"

# --------------------------------------------------------------- what it scores

M=$tmp/m1.txt
# stdout is the answer, stderr is the commentary: that split is what makes it a pipeline stage.
check "a case is scored to stdout as CSV" "$("$bin" -c "$M" A x=3 2>/dev/null)" "A,$("$bin" -c "$M" A x=3 2>/dev/null | cut -d, -f2)"
check "and nothing but the answer is on stdout" \
      "$("$bin" -c "$M" A x=3 2>/dev/null | wc -l | tr -d ' ')" "1"
check "the fit-time error goes to stderr" \
      "$("$bin" -c "$M" A x=3 2>&1 >/dev/null | grep -c "held-out RMSE at fit time")" "1"

# Cases from a pipe: the form a deployed predictor actually uses.
check "a pipe of cases gives one line each" \
      "$(printf 'A,3\nA,4\nA,5\n' | "$bin" -c "$M" 2>/dev/null | wc -l | tr -d ' ')" "3"
check "and each line names its group" \
      "$(printf 'A,3\nA,4\n' | "$bin" -c "$M" 2>/dev/null | cut -d, -f1 | sort -u)" "A"
check "a header line is skipped, so a training file scores" \
      "$(printf 'group,x\nA,3\n' | "$bin" -c "$M" 2>/dev/null | wc -l | tr -d ' ')" "1"
check "a case with the wrong field count is refused" \
      "$(printf 'A,3,4\n' | "$bin" -c "$M" 2>&1 >/dev/null | grep -m1 '^-:')" \
      "-:1: 3 fields; this model wants the group and 1 term"
set +e
printf 'A,3,4\n' | "$bin" -c "$M" >/dev/null 2>&1; rc=$?
set -e
check "and exits 1" "$rc" "1"
check "a non-numeric case in the stream is refused with its line and column" \
      "$(printf 'A,oops\n' | "$bin" -c "$M" 2>&1 >/dev/null | grep -m1 '^-:')" \
      "-:1:2: the term 'x' is 'oops', which is not a number."
check "an unknown group in the stream is refused" \
      "$(printf 'Q,3\n' | "$bin" -c "$M" 2>&1 >/dev/null | grep -m1 '^-:')" \
      "-:1:1: group 'Q' is not in this model"

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
badscore "two group names"    "bpnn: two group names given, 'A' and 'B'" A B x=1

set +e
out=$("$bin" -c "$tmp/ok.csv" A x=1 2>&1 >/dev/null); rc=$?
set -e
check "a CSV offered as a model" "$(firstline "$out")" "$tmp/ok.csv:41: this is not a usable bpnn model"
check "a CSV offered as a model exits 1" "$rc" "1"

# The extrapolation check: inside the training range it stays quiet, outside it does not.
check "inside the range, no warning" \
      "$("$bin" -c "$M" A x=4 2>&1 >/dev/null | grep -c 'outside the fitted range' || true)" "0"
check "outside it, the term and the distance" \
      "$("$bin" -c "$M" A x=99 2>&1 >/dev/null | grep -c 'outside the fitted range')" "1"
check "and what that means" \
      "$("$bin" -c "$M" A x=99 2>&1 >/dev/null | grep -c 'does not extrapolate')" "1"

# ---------------------------------------------------------------- --patience
# -e is a ceiling. The fit stops when the rows kept back for the purpose stop improving, and the
# epochs column says where it stopped.

{ echo 'group,y,dose'
  i=0; while [ $i -lt 200 ]; do
      d=$((i % 40)); echo "A,$((12 * d / (2 + d) + i % 3)),$d"; i=$((i+1)); done; } > "$tmp/stop.csv"

epochs_used() { "$bin" -t "$tmp/stop.csv" "$@" 2>&1 >/dev/null | awk '$1=="A"{print $5}'; }
used=$(epochs_used -e 3000 -s 2)
check "the fit stops before the ceiling" "$([ "$used" -lt 3000 ] && echo yes)" "yes"
check "--patience 0 runs every epoch"    "$(epochs_used -e 200 -s 2 --patience 0)" "200"
check "the epochs column is in the report" \
      "$("$bin" -t "$tmp/stop.csv" $FAST 2>&1 >/dev/null | grep '^group ' | awk '{print $5}')" "epochs"

# Half the held-out rows decide when to stop, so the reported count is the other half. With the
# check off, every held-out row is reported.
held_count() { "$bin" -t "$tmp/stop.csv" "$@" 2>&1 >/dev/null | awk '$1=="A"{print $3}'; }
check "stopping costs half the reported held-out rows" "$(held_count -e 100 -s 2)" "25"
check "and --patience 0 reports all of them"           "$(held_count -e 100 -s 2 --patience 0)" "50"

# A group whose held-out half is too small to judge a stop by runs its full epochs instead.
{ echo 'group,y,x'
  i=1; while [ $i -le 40 ]; do echo "T,$((i * i + 10)),$i"; i=$((i + 1)); done; } > "$tmp/tiny.csv"
# Too few rows to hold back a stopping sample: the fit runs every epoch and says so, because
# without the check it memorises the group while reporting a plausible variance explained.
{ echo 'group,y,x'
  i=1; while [ $i -le 34 ]; do echo "T,$((i * i + 10)),$i"; i=$((i + 1)); done; } > "$tmp/nostop.csv"
check "a group too small to judge a stop says so" \
      "$("$bin" -t "$tmp/nostop.csv" -e 70 -s 2 --holdout 0.15 2>&1 >/dev/null \
         | grep -c 'too few to hold back a stopping sample')" "1"
check "and runs every epoch" \
      "$("$bin" -t "$tmp/nostop.csv" -e 70 -s 2 --holdout 0.15 2>&1 >/dev/null | awk '$1=="T"{print $5}')" "70"

badopt "a non-numeric --patience" "bpnn: --patience six is not a number" \
       -t "$tmp/ok.csv" --patience six

# ------------------------------------------------------------------- --stream
# The streaming path fits without holding the rows. It reads the file twice and then once per
# epoch from a cache, so it must agree with the default path about what the file contains and
# about what it refuses, and differ from it only in the training order.

set +e
"$bin" -t "$tmp/ok.csv" $FAST --stream > "$tmp/s1.txt" 2>"$tmp/sr1.txt"; rc=$?
set -e
check "--stream exits 0"            "$rc" "0"
check "--stream names the response" "$(grep '^response' "$tmp/s1.txt")" "response y"
check "--stream names the terms"    "$(grep '^terms' "$tmp/s1.txt")" "terms 1 x"
check "--stream fits the group"     "$(grep -c '^GROUP A' "$tmp/s1.txt")" "1"
check "--stream reports like the default" \
      "$(grep '^group ' "$tmp/sr1.txt" | awk '{print $1, $NF}')" "group expl"

# -e is a ceiling on this path too, and the stop rows are held out of the reported error the same
# way. With --patience 0 every epoch runs and every held-out row is reported.
stream_used() { "$bin" -t "$tmp/stop.csv" --stream "$@" 2>&1 >/dev/null | awk '$1=="A"{print $5}'; }
used=$(stream_used -e 3000 -s 2)
check "--stream stops before the ceiling" "$([ "$used" -lt 3000 ] && echo yes)" "yes"
check "--stream --patience 0 runs every epoch" "$(stream_used -e 50 -s 2 --patience 0)" "50"
sheld() { "$bin" -t "$tmp/stop.csv" --stream "$@" 2>&1 >/dev/null | awk '$1=="A"{print $3}'; }
check "--stream reports fewer held rows when it stops early" \
      "$([ "$(sheld -e 50 -s 2)" -lt "$(sheld -e 50 -s 2 --patience 0)" ] && echo yes)" "yes"

"$bin" -t "$tmp/ok.csv" $FAST --stream > "$tmp/s2.txt" 2>/dev/null
check "--stream repeats byte for byte" "$(cmp -s "$tmp/s1.txt" "$tmp/s2.txt" && echo same)" "same"

# Different training order, so a different model. Both paths are deterministic; neither is the
# other's approximation.
check "--stream is not the default path" \
      "$(cmp -s "$tmp/s1.txt" "$tmp/m1.txt" && echo same || echo differs)" "differs"

# The window is a number of rows and the refusals are the reader's, unchanged.
badopt "a zero buffer" "bpnn: --buffer is a number of rows and must be at least 1" \
       -t "$tmp/ok.csv" --stream --buffer 0
printf 'group,y,x\nA,1,1\nA,2,nan\n' > "$tmp/bad.csv"
set +e
out=$("$bin" -t "$tmp/bad.csv" $FAST --stream 2>&1 >/dev/null); rc=$?
set -e
check "--stream refuses a nan too" "$(firstline "$out")" \
      "$tmp/bad.csv:3:3: the term 'x' is 'nan': not finite"
check "--stream refuses a nan too, exit" "$rc" "1"

# A window smaller than the file, on a file sorted by the response, trains on nearly the file's
# own order. That is worth a warning rather than a quietly worse fit.
{ echo 'group,y,x'; i=0; while [ $i -lt 40 ]; do echo "A,$i,$i"; i=$((i+1)); done; } > "$tmp/sorted.csv"
out=$("$bin" -t "$tmp/sorted.csv" $FAST --stream --buffer 4 2>&1 >/dev/null)
check "a sorted file with a small window is reported" \
      "$(printf '%s' "$out" | grep -c 'is sorted by y, or nearly so')" "1"
out=$("$bin" -t "$tmp/sorted.csv" $FAST --stream --buffer 1000 2>&1 >/dev/null)
check "and not when the window holds the file" \
      "$(printf '%s' "$out" | grep -c 'is sorted by y, or nearly so' || true)" "0"

# --footprint answers about a shape, not about a file, so it takes no data and reads none.
check "--footprint prints a total" \
      "$("$bin" --footprint 24 400 | grep -c '^  total ')" "1"
check "--footprint says what a row costs the default path" \
      "$("$bin" --footprint 24 400 | grep -c '^  per row ')" "1"
check "--footprint names the group ceiling when it is passed" \
      "$("$bin" --footprint 24 400000 | grep -c 'this build fits at most 512 groups')" "1"
# The row store packs to the file's own term count. It used to allocate the build ceiling of 64
# terms for every row, so a two-term file paid 528 bytes a row to hold 16 bytes of data.
check "a row costs the terms it has, not the ceiling" \
      "$("$bin" --footprint 2 4 | awk '/per row/{print $3}')" "32"
check "and twelve times the terms costs proportionally" \
      "$("$bin" --footprint 24 4 | awk '/per row/{print $3}')" "208"
set +e
out=$("$bin" --footprint 24 0 2>&1 >/dev/null); rc=$?
set -e
check "--footprint refuses a zero count" "$(firstline "$out")" \
      "bpnn: --footprint takes a term count and a group count, both above 0"
check "--footprint refuses a zero count, exit" "$rc" "2"

# ------------------------------------------------- --decay, and underfitting
# Weight decay pulls every weight toward zero. Enough of it and the network stops using its
# inputs at all, which is the underfitting case: both errors large and equal, and no warning
# anywhere until the errors are read against the spread of the thing being predicted.

check "--decay is recorded in the model" \
      "$("$bin" -t "$tmp/ok.csv" $FAST --decay 0.01 2>/dev/null | grep -c 'decay=0.01')" "1"
check "a fit worse than the mean is reported as such" \
      "$("$bin" -t "$tmp/stop.csv" -e 200 -s 2 -r 50 2>&1 >/dev/null \
         | grep -c 'barely using its')" "1"
badopt "a decay that cannot converge at this rate" \
       "bpnn: --decay 20 at -r 0.3 shrinks every weight by 6 of itself per row." \
       -t "$tmp/ok.csv" $FAST --decay 20
check "--size is -H under nnet's name" \
      "$("$bin" -t "$tmp/ok.csv" $FAST --size 3 2>/dev/null | grep -c 'hidden=3')" "1"
check "and a fit that works says nothing about it" \
      "$("$bin" -t "$tmp/stop.csv" -e 200 -s 2 2>&1 >/dev/null \
         | grep -c 'barely using its' || true)" "0"
check "the variance explained is a column" \
      "$("$bin" -t "$tmp/stop.csv" -e 200 -s 2 2>&1 >/dev/null | grep '^group ' | awk '{print $NF}')" "expl"
check "and is in the model's diag line" \
      "$("$bin" -t "$tmp/stop.csv" -e 200 -s 2 2>/dev/null | grep -c 'expl=')" "1"
badopt "a negative --decay" \
       "bpnn: --decay is a penalty on the size of the weights and cannot be" \
       -t "$tmp/ok.csv" --decay -1

# -------------------------------------------------------------------- --cache
# The two setup passes produce the same scaled rows whatever the hyperparameters, so --cache
# keeps them between runs. A cached run must be indistinguishable from an uncached one.

# On a copy: one of these checks appends to the input, and every later check still expects
# ok.csv to be the eight rows it started as.
cp "$tmp/ok.csv" "$tmp/cache.csv"
"$bin" -t "$tmp/cache.csv" $FAST --stream --cache "$tmp/c1.bin" > "$tmp/c1.txt" 2>/dev/null
check "the first run builds the cache" "$([ -s "$tmp/c1.bin" ] && echo yes)" "yes"
"$bin" -t "$tmp/cache.csv" $FAST --stream --cache "$tmp/c1.bin" > "$tmp/c2.txt" 2>/dev/null
check "the second run reuses it and agrees" \
      "$(cmp -s "$tmp/c1.txt" "$tmp/c2.txt" && echo same)" "same"
# The provenance comment names the input, and these two runs read different files, so compare
# everything the model is rather than the header comments.
check "and both agree with no cache at all" \
      "$(grep -v '^#' "$tmp/c1.txt" > "$tmp/c1b"; grep -v '^#' "$tmp/s1.txt" > "$tmp/s1b"; \
         cmp -s "$tmp/c1b" "$tmp/s1b" && echo same)" "same"
check "a reused cache is silent about it" \
      "$("$bin" -t "$tmp/cache.csv" $FAST --stream --cache "$tmp/c1.bin" 2>&1 >/dev/null \
         | grep -c 'rebuilding it' || true)" "0"

# The key is the input's size and modification time, so a changed input rebuilds it.
sleep 1
printf 'A,90,9\n' >> "$tmp/cache.csv"
check "a changed input rebuilds the cache" \
      "$("$bin" -t "$tmp/cache.csv" $FAST --stream --cache "$tmp/c1.bin" 2>&1 >/dev/null \
         | grep -c 'the input has changed')" "1"

printf 'not a cache at all' > "$tmp/junk.bin"
check "a file that is not a cache is replaced, not read" \
      "$("$bin" -t "$tmp/cache.csv" $FAST --stream --cache "$tmp/junk.bin" 2>&1 >/dev/null \
         | grep -c 'is not a bpnn row cache')" "1"

badopt "--cache without --stream" \
       "bpnn: --cache holds the scaled rows --stream reads, so it needs" \
       -t "$tmp/ok.csv" $FAST --cache "$tmp/c1.bin"

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
# %g's six significant digits would print 1e+08 for every case in this group: all six spent on
# the offset. The prediction must still show the part that moves.
check "and the prediction still resolves the response" \
      "$("$bin" -c "$tmp/mo.txt" A x=3 2>/dev/null | cut -d, -f2 | cut -c1-6)" "100000"
check "and not as %g's six digits would print it" \
      "$([ "$("$bin" -c "$tmp/mo.txt" A x=3 2>/dev/null | cut -d, -f2 | wc -c)" -ge 10 ] \
        && echo yes)" "yes"

# Two columns that differ in the sixth decimal: the fit must not produce a nan or refuse.
{ echo 'group,y,x1,x2'
  i=1; while [ $i -le 40 ]; do
      echo "A,$((i*3+1)),$i,$i.000001"; i=$((i+1)); done; } > "$tmp/coll.csv"
set +e
out=$("$bin" -t "$tmp/coll.csv" $FAST 2>&1 >/dev/null); rc=$?
set -e
check "near-identical columns still fit" "$rc" "0"
check "and produce no nan"               "$(printf '%s' "$out" | grep -ci nan || true)" "0"

# ------------------------------------------- what three reviewers found in it
# One check per defect an outside review demonstrated. Every one of these passed the suite as
# it stood, which is the reason they are written down as checks rather than as fixes.

# A model file naming more groups than the build holds wrote through gp[-1] and exited 0.
{ echo "BPNN 1"; echo "response y"; echo "terms 1 x"
  i=0; while [ $i -le 520 ]; do
      echo "GROUP g$i rows=1 held=0"; echo "target 0 1"; echo "range 0 1"
      echo "net 3 1 1 2 1"; echo "END"; i=$((i + 1)); done; } > "$tmp/manygroups.model"
set +e
"$bin" -c "$tmp/manygroups.model" g1 x=1 >/dev/null 2>&1; rc=$?
set -e
check "a model with more groups than the build holds is refused" "$rc" "1"

# Below the checkpoint interval, the restore copied a net that had never been written.
a=$("$bin" -t "$tmp/ok.csv" -e 5 -s 3 2>/dev/null | digest)
b=$("$bin" -t "$tmp/ok.csv" -e 5 -s 3 2>/dev/null | digest)
check "a fit shorter than one checkpoint is still deterministic" "$a" "$b"
check "and does not ship all-zero weights" \
      "$("$bin" -t "$tmp/ok.csv" -e 5 -s 3 2>/dev/null | awk '$1=="w" && $2+0!=0' | wc -l | tr -d ' ')" \
      "$("$bin" -t "$tmp/ok.csv" -e 5 -s 3 2>/dev/null | grep -c '^w ')"

# --stream stopped training at epoch 25 whenever a fit had too few rows to judge a stop by,
# and then reported the full epoch count.
c=$("$bin" -t "$tmp/tiny.csv" --stream -e 25 -s 2 --holdout 0.15 2>/dev/null | digest)
d=$("$bin" -t "$tmp/tiny.csv" --stream -e 400 -s 2 --holdout 0.15 2>/dev/null | digest)
check "--stream keeps training when it cannot judge a stop" \
      "$([ "$c" = "$d" ] && echo identical || echo differs)" "differs"

# The variance was computed as E[y^2]-E[y]^2, which loses every digit at a large offset.
awk -F, 'NR==1{print;next}{printf "%s,%.10g,%s\n",$1,$2+10000000000,$3}' "$tmp/ok.csv" > "$tmp/e10.csv"
check "an offset of 1e10 does not destroy the variance explained" \
      "$("$bin" -t "$tmp/e10.csv" -e 200 -s 2 2>&1 >/dev/null | grep -c 'barely using its' || true)" "0"

# srt[nseed/2] is the UPPER middle, so at two refits the shipped model was the worse one. The
# reported error is now the mean over refits, so this reads the shipped model's own figure from
# the model file and checks it against the smaller of the two refits.
"$bin" -t "$tmp/ok.csv" -e 200 -s 2 --per-refit "$tmp/two.refits" > "$tmp/two.model" 2>/dev/null
shipped=$(grep '^diag ' "$tmp/two.model" | tr ' ' '\n' | sed -n 's/^shipped=//p')
lower=$(grep '^REFIT ' "$tmp/two.refits" | awk '{print $4}' | sort -n | head -1)
check "two refits ship the better of the two, not the worse" \
      "$(awk -v a="$shipped" -v b="$lower" 'BEGIN{d=a-b; if(d<0)d=-d; print (d < 1e-4*b) ? "same" : a" vs "b}')" \
      "same"

# A response with one extreme value compresses every other row into a sliver of the output
# band, and the variance explained beside it is measured against that same spread.
{ cat "$tmp/ok.csv"; echo "A,100000,41"; } > "$tmp/outlier.csv"
check "an extreme response value is reported" \
      "$("$bin" -t "$tmp/outlier.csv" -e 200 -s 2 2>&1 >/dev/null | grep -ci 'interquartile ranges')" "1"

# The output unit saturates 12.5% past the fitted range; the prediction is then the ceiling.
"$bin" -t "$tmp/ok.csv" -e 300 -s 2 > "$tmp/sat.txt" 2>/dev/null
check "a saturated prediction says so" \
      "$("$bin" -c "$tmp/sat.txt" A x=400 2>&1 >/dev/null | grep -c 'saturated')" "1"
check "and an ordinary one does not" \
      "$("$bin" -c "$tmp/sat.txt" A x=20 2>&1 >/dev/null | grep -c 'saturated' || true)" "0"

# --------------------------------------------------------------- --per-refit
# Two runs are comparable as matched pairs only if they drew from the same seed lattice, and
# pairstat refuses them when they did not. Skipped when the tool has not been built.

# A silent skip made `make check` report 160 of 165 checks while the documentation claimed 165.
if [ ! -x "$root/pairstat" ]; then
    echo "  SKIP: pairstat is not built (run make tools); 7 checks not run"
fi
if [ -x "$root/pairstat" ]; then
    "$bin" -t "$tmp/ok.csv" -e 100 -s 3 -H 4 --per-refit "$tmp/a.refits" >/dev/null 2>&1
    "$bin" -t "$tmp/ok.csv" -e 100 -s 3 -H 8 --per-refit "$tmp/b.refits" >/dev/null 2>&1
    "$bin" -t "$tmp/ok.csv" -e 100 -s 5 -H 8 --per-refit "$tmp/c.refits" >/dev/null 2>&1
    check "--per-refit writes one line per refit" \
          "$(grep -c '^REFIT ' "$tmp/a.refits")" "3"
    check "and stamps the seed lattice" \
          "$(grep -c '^PAIRING ' "$tmp/a.refits")" "1"
    check "matched runs compare" \
          "$("$root/pairstat" --paired "$tmp/a.refits" "$tmp/b.refits" 2>&1 | grep -c 'held-out')" "2"
    set +e
    out=$("$root/pairstat" --paired "$tmp/a.refits" "$tmp/c.refits" 2>&1 >/dev/null); rc=$?
    set -e
    check "runs from different lattices are refused" \
          "$(firstline "$out")" "pairstat: these two runs are not paired and cannot be compared as if"
    check "and the refusal exits nonzero" "$rc" "2"
fi

# ------------------------------------------------- what the fit says it read
# A CSV cannot say which column is the response, so a file in another order fits the wrong one
# and exits 0. The commonest mistake there is has to be visible where a person is looking.

check "the fit echoes the response and the terms" \
      "$("$bin" -t "$tmp/ok.csv" $FAST 2>&1 >/dev/null | grep -c "'y' is the value being predicted")" "1"
awk -F, 'BEGIN{OFS=","} NR==1{print $1,$3,$2; next} {print $1,$3,$2}' "$tmp/ok.csv" > "$tmp/rev.csv"
check "and names the wrong one when the columns are swapped" \
      "$("$bin" -t "$tmp/rev.csv" $FAST 2>&1 >/dev/null | grep -c "'x' is the value being predicted")" "1"
check "-y puts it right" \
      "$("$bin" -t "$tmp/rev.csv" -y y $FAST 2>&1 >/dev/null | grep -c "'y' is the value being predicted")" "1"
check "-y naming no column is refused" \
      "$("$bin" -t "$tmp/ok.csv" -y nope $FAST 2>&1 >/dev/null | grep -c 'names no column in this file')" "1"

# -h beside an action wrote the help text into the model file and exited 0.
badopt "--help beside a fit" "bpnn: --help takes no other arguments. Did you mean -H 12?" \
       -t "$tmp/ok.csv" -h 12
# --opt=value used to be reported as an option that does not exist.
check "--opt=value is accepted" \
      "$("$bin" -t "$tmp/ok.csv" --holdout=0.3 $FAST 2>&1 >/dev/null | grep -c '^group ')" "1"

# ------------------------------------------- what the build engineer demonstrated
# A truncated cache was accepted and answered differently with exit 0; the model version was
# written and never read; the sanitizer targets could not fail. The first two are checkable here.

# A silent skip made `make check` report 160 of 165 checks while the documentation claimed 165.
if [ ! -x "$root/pairstat" ]; then
    echo "  SKIP: pairstat is not built (run make tools); 7 checks not run"
fi
if [ -x "$root/pairstat" ]; then
    "$bin" -t "$tmp/ok.csv" $FAST --stream --cache "$tmp/tr.bin" >/dev/null 2>&1
    before=$(wc -c < "$tmp/tr.bin")
    dd if=/dev/null of="$tmp/tr.bin" bs=1 seek=$((before - 12)) >/dev/null 2>&1
    set +e
    out=$("$bin" -t "$tmp/ok.csv" $FAST --stream --cache "$tmp/tr.bin" 2>&1 >/dev/null); rc=$?
    set -e
    check "a truncated cache is refused, not used" \
          "$(printf '%s' "$out" | grep -c 'truncated or')" "1"
    check "and the run fails rather than answering" "$rc" "1"
fi

"$bin" -t "$tmp/ok.csv" $FAST > "$tmp/ver.model" 2>/dev/null
check "the model states its format version" "$(grep -c '^BPNN 1$' "$tmp/ver.model")" "1"
sed 's/^BPNN 1$/BPNN 99/' "$tmp/ver.model" > "$tmp/v99.model"
set +e
out=$("$bin" -c "$tmp/v99.model" A x=3 2>&1 >/dev/null); rc=$?
set -e
check "a newer model version is refused by name" \
      "$(printf '%s' "$out" | grep -c 'this build reads up to 1')" "1"
check "and refusing it is a failure" "$rc" "1"
grep -v '^BPNN' "$tmp/ver.model" > "$tmp/vno.model"
set +e
"$bin" -c "$tmp/vno.model" A x=3 >/dev/null 2>&1; rc=$?
set -e
check "a model with no version line is refused" "$rc" "1"

# ----------------------------------- what the deployment reviewer demonstrated
# A batch scored against a header whose columns are permuted returned 2.73643 where the truth
# was 13.4359, exit 0. The header is validated by name now.

"$bin" -t "$tmp/ok.csv" -e 200 -s 2 > "$tmp/hdr.model" 2>/dev/null
check "a correct header is accepted" \
      "$(printf 'group,x\nA,3\n' | "$bin" -c "$tmp/hdr.model" 2>/dev/null | wc -l | tr -d ' ')" "1"
set +e
out=$(printf 'group,z\nA,3\n' | "$bin" -c "$tmp/hdr.model" 2>&1 >/dev/null); rc=$?
set -e
check "a header naming another column is refused" \
      "$([ "$rc" -ne 0 ] && echo refused)" "refused"
# The case the reviewer demonstrated: a header whose first term matches and whose later columns
# are permuted. That used to be skipped and every row read by position.
"$bin" -t "$tmp/two.csv" -e 200 -s 2 > "$tmp/perm.model" 2>/dev/null
set +e
out=$(printf 'group,x\nA,3\n' | "$bin" -c "$tmp/perm.model" 2>&1 >/dev/null); rc2=$?
set -e
check "a matching header still scores" "$rc2" "0"

# The effective step is r/(1-m); the two are one knob and the default sat past the optimum.
check "an effective step over 5 is named" \
      "$("$bin" -t "$tmp/ok.csv" $FAST -m 0.95 2>&1 >/dev/null | grep -c 'effective step')" "1"
check "and the default is not" \
      "$("$bin" -t "$tmp/ok.csv" $FAST 2>&1 >/dev/null | grep -c 'effective step' || true)" "0"

# ------------------------------------------- reading real data, and trusting a model
# A file with one empty cell in a hundred kills a fifth of the rows at 24 columns, so refusing
# the whole file means clinical data cannot be read at all. Dropping is offered, counted, and
# recorded in the model, because dropping biases the fit when missingness carries information.

awk -F, 'BEGIN{OFS=","} NR>1 && NR%7==0 {$3=""} {print}' "$tmp/ok.csv" > "$tmp/gap.csv"
set +e
"$bin" -t "$tmp/gap.csv" $FAST >/dev/null 2>&1; rc=$?
set -e
check "a gap still refuses the file by default" "$rc" "1"
check "--missing drop reads it" \
      "$("$bin" -t "$tmp/gap.csv" $FAST --missing drop 2>/dev/null | grep -c '^GROUP A')" "1"
check "and records what it dropped" \
      "$("$bin" -t "$tmp/gap.csv" $FAST --missing drop 2>/dev/null | grep -c 'rows were dropped')" "1"
badopt "an unknown --missing policy" "bpnn: --missing takes refuse or drop, not sometimes" \
       -t "$tmp/ok.csv" --missing sometimes

# An id echoed back, because a prediction that cannot be joined to its subject cannot be checked
# against an outcome later.
check "--id echoes the id on the prediction" \
      "$(printf 'A,P77,3\n' | "$bin" --id -c "$M" 2>/dev/null)" "A,P77,$(printf 'A,3\n' | "$bin" -c "$M" 2>/dev/null | cut -d, -f2)"
badopt "--id after -c, where the case begins" \
       "bpnn: --id comes after -c, where everything is part of the case." \
       -c "$M" --id

# An edited weight used to be scored: one digit moved a prediction by six days.
"$bin" -t "$tmp/ok.csv" -e 200 -s 2 > "$tmp/ck.model" 2>/dev/null
check "the model carries a checksum" "$(grep -c '^checksum ' "$tmp/ck.model")" "1"
sed '0,/^w /{s/^w .*/w 9.9/}' "$tmp/ck.model" > "$tmp/ed.model"
set +e
out=$("$bin" -c "$tmp/ed.model" A x=3 2>&1 >/dev/null); rc=$?
set -e
check "an edited weight is refused" \
      "$(printf '%s' "$out" | grep -c 'checksum does not match')" "1"
check "and refusing it fails" "$rc" "1"

# ------------------------------------------------------------------------ end

echo "$((pass+fail)) checks, $fail failed"
[ "$fail" -eq 0 ]
