# bpnn

**A backpropagation feed-forward neural network in C99, and tools for measuring how much of a
neural-network experiment's result is real.**

The engine derives from a 1997 seminar thesis on the backpropagation delta rule in tensor form, was
prototyped in Java in the early 2000s, and was re-implemented in C99 as part of
[SMBPANN](https://github.com/Anode1/SMBPANN). This is a fork of that work with the evolutionary
architecture search removed.

## Why the search is gone

SMBPANN used an evolutionary search as an instrument rather than an optimiser, asking which pieces of
a convolution a search recovers unaided under a cost on connections. That produced one paper, one
retracted draft, and a third result that survived three rounds of adversarial review only after
shrinking a great deal. The recurring problem was the search: on the spaces we could afford it is a
weak hill-climber that stalls, and measurements about it kept turning out to be confounded by the
stalling rather than informative. On a public benchmark it lost to matched-budget random selection.

The engine, the unit suite and the statistics tooling were never in doubt, so they are what carried
over. SMBPANN is untouched and remains the reference implementation for the paper that cites it.

## What is here

    make            build ./bpnn
    make ut         32 unit checks, the commit gate
    make ut-asan    the same under AddressSanitizer
    make ut-ubsan   the same under UBSan
    make pedantic   -pedantic plus -Wextra -Wshadow -Wconversion, must be clean
    make tools      pairstat, nb101_trials, nb101_signal (runs pairstat's self-test)

The engine is a flat-array feed-forward network with the generalized delta rule and momentum,
selectable activations, weight-shared 1-D convolution whose gradients are checked against finite
differences, a 2-D convolutional front-end, a Mark/Release arena allocator, and plain-text datasets.
Training allocates nothing on the hot path and peak footprint is computable by hand from the struct
sizes and the topology.

`validation/pairstat.c` is the piece most worth borrowing. It computes paired statistics with named
tests: Wilcoxon signed-rank as primary, using an exact null by dynamic programming where ties permit
and a tie-corrected normal approximation otherwise, always reporting which it used; an exact sign
test; Hodges-Lehmann estimates with distribution-free intervals; a paired *t* as secondary; Holm
correction across a declared family; and a minimum detectable effect, so a null result is bounded
rather than merely asserted. It refuses to be trusted before passing `--selftest` against
hand-computable cases.

## What it is for now

Measuring how much of a reported result survives replication. The first question is how much of a
NAS benchmark's architecture ranking is training noise. NAS-Bench-101 trains every one of its 423,624
architectures three separate times, and the distributed tables average those runs away;
`validation/nb101_trials.c` keeps them. Retraining the same architecture moves validation accuracy by
0.33 percentage points at the median and 42.7 at the 99th percentile, because about one architecture
in a hundred sometimes trains and sometimes collapses to the accuracy of guessing. Architecture-search
papers routinely contest differences smaller than the median.

That question needs no search and no training, only table lookups, so nothing in it can be confounded
by an optimiser. See `AGENTS.md` for the roadmap and for the specific mistakes this project has
already paid for and intends not to repeat.

## Provenance

`validation/PROVENANCE_nas.md` gives the exact URLs and commands for both benchmarks, including a
NAS-Bench-201 source that keeps per-seed values and needs neither Google Drive nor PyTorch. The
archives are gitignored because they are large and re-fetchable.
