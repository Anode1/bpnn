# bpnn

**A backpropagation feed-forward neural network in C99, and tools for measuring how much of a
neural-network experiment's reported result is real.**

## The goal

Papers report that one architecture, one hyper-parameter setting, or one training recipe beats another
by a fraction of a percentage point. Training is random, so any single measurement is the thing's real
quality plus a draw of noise, and choosing the best of several such measurements keeps the noise along
with the quality. This project measures how much of a reported improvement survives replication, on
data where the question can actually be settled.

The first measurement is done and it is blunt. On NAS-Bench-101, judging which of two architectures is
better from one training run each is **backwards 42% of the time** when the observed difference is 0.1
to 0.2 percentage points. Turned into what it implies: **a single training run per architecture can
only establish differences larger than 2.5 percentage points**, and establishing a 0.1 point difference
would take roughly 650 runs per architecture. Architecture-search papers routinely report CIFAR-10
improvements of 0.1 to 0.3 points from single runs. See `doc/FINDINGS.md`.

## The network

A feed-forward network trained by backpropagation, derived from a 1997 seminar thesis that works the
delta rule through in tensor form; `net.c` and `train.c` cite it by section. Roughly 1,800 lines of
C99, no dependencies beyond libm.

- **Layers** are flat weight matrices, dense or weight-shared 1-D convolution, with a 2-D
  convolutional front-end (F filters of K×K, pooled) available in front of the network.
- **Training** is the generalized delta rule with momentum. Convolutional layers share gradients across
  positions, and that sharing is checked against finite differences in the unit suite rather than
  assumed.
- **Activations** are sigmoid, tanh or ReLU per network; the output layer stays a sigmoid.
- **Memory** is allocated only in the construction and load paths. The train and inference hot paths
  allocate nothing, every path frees on exit through a single-exit `goto`, and peak footprint is
  computable by hand from the struct sizes plus the topology. A Mark/Release arena allocator is
  available where many short-lived networks are built and discarded.
- **Determinism** is deliberate: a seeded PRNG for initialisation, so a result table can be reproduced
  byte for byte and a refactor can be proven not to have moved a number.

This is not a framework and is not trying to become one. It exists to run experiments whose answers
must be exactly reproducible, and it is fast where it matters for that: many small independent networks,
where per-operation dispatch overhead would dominate in a general-purpose library.

## Build and test

    make            build ./bpnn
    make ut         32 unit checks: rng, act, net, xor, arena, data, conv1d, conv2f -- the commit gate
    make ut-asan    the same under AddressSanitizer
    make ut-ubsan   the same under UBSan
    make pedantic   -pedantic with -Wextra -Wshadow -Wconversion; must be clean
    make tools      the measurement tools; runs pairstat's self-test
    make clean

A warning is a defect. `make ut` gates every commit and the sanitizers run before one.

## The measurement tools

**`validation/pairstat.c`** is the piece most worth borrowing. Paired statistics with named tests:
Wilcoxon signed-rank as primary, using an exact null by dynamic programming where ties permit and a
tie-corrected normal approximation otherwise, always reporting which it used; an exact sign test;
Hodges-Lehmann estimates with distribution-free intervals; a paired *t* as secondary; Holm correction
across a declared family; and a minimum detectable effect, so that a null result is bounded rather than
merely asserted. It refuses to be trusted until `--selftest` passes against hand-computable cases.

**`validation/nb101_flip.c`** is the flip measurement: how often one training run gets the ordering of
two architectures wrong, and what resolution a given number of runs buys.

**`validation/nb101_signal.c`** measures the other half of the same question, how much a search
inflates its own reported score by selecting on it.

**`validation/nb101_trials.c`** and **`validation/nb201_extract.c`** recover the per-run values from
NAS-Bench-101 and NAS-Bench-201. Both benchmarks train every architecture three times, and both of the
distributed tables average those runs away, which is the right table for ranking architectures and
destroys the only structure a study of the noise can use. `validation/PROVENANCE_nas.md` has the URLs
and the commands.

## Origin

Forked from [SMBPANN](https://github.com/Anode1/SMBPANN), which used an evolutionary search over
topology as an instrument for asking which pieces of a convolution a search recovers unaided. The
engine, the unit suite and the statistics tooling carried over; the search did not, because on
affordable problem sizes it was a weak hill-climber whose stalling confounded the measurements built on
it, and on a public benchmark it lost to matched-budget random search. SMBPANN is untouched and remains
the reference implementation for the paper that cites it.

`AGENTS.md` holds the operating manual, including the specific mistakes this line has already paid for.
