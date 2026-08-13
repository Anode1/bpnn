# Benchmarks

Every figure here was measured on one machine, named below, and reproduced by a script in this
repository. A rate without a machine is not a measurement, and a number nobody can regenerate is
not a benchmark.

    machine   11th Gen Intel Core i7-1165G7 @ 2.80GHz, 4 cores / 8 threads, 12 MB L3
    memory    62 GB
    compiler  gcc 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1), -O2
    system    Ubuntu 24.04.4, Linux 6.8.0

Four cores, eight threads. The engine is single-threaded, so nothing here uses more than one of
them; the thread count matters only because sizing a run as if eight cores were available
overran one estimate in the predecessor project by 40%.

## Memory does not depend on the number of rows

`scripts/scale.sh` fits the same model over N rows and over 10N and compares peak resident
memory. Two terms, four groups, `-e 2`:

    $ scripts/scale.sh 100000
                                            100000 rows 1000000 rows
    --stream, peak RSS (kB)                        8832         8832
    default, peak RSS (kB)                         6088        39308

The default path holds every row and grows with them. `--stream` does not.

At the size where it matters, a 235 MB file of ten million rows, four groups, two terms:

| | wall | peak RSS |
|---|---|---|
| `--stream -e 2 -s 2` | 17.6 s | **4.7 MB** |
| default `-e 2 -s 2` | 19.3 s | 373 MB |

The streaming figure is the networks plus the shuffle windows and nothing else. The file is 50
times larger than the memory used to fit it.

**`--stream` is not free.** It pays for one shuffle window per refit whatever the file size, so
below the crossover it costs *more* than holding the rows. A row costs `(terms + 1) * 8 + 8`
bytes held, so the crossover is around 200,000 rows at two terms and 170,000 at twenty-four.
`./bpnn --footprint TERMS GROUPS` prints both figures for a shape before you commit to one.

## Fitting rate

`bench/throughput.sh` measures it, best of three, with `--patience 0` so that a fit which
converges early does not make the engine look faster than it is. `-e 20 -s 2` throughout, so
the only thing moving is the shape of the data.

    $ bench/throughput.sh 100000
    rows            terms   groups   wall (s)       rows/s    updates/s
    10000               4        4       0.07     1.43e+05     5.71e+06
    100000              4        4       0.72     1.39e+05     5.56e+06
    500000              4        4       3.86      1.3e+05     5.18e+06

    rows            terms   groups   wall (s)       rows/s    updates/s
    100000              2        4       0.62     1.61e+05     6.45e+06
    100000              8        4       0.91      1.1e+05      4.4e+06
    100000             24        4       1.68     5.95e+04     2.38e+06

The rate to size a run with is **row-updates a second**: rows × epochs × refits over the wall
clock. It holds when any one of the three changes, which the rows-per-second figure does not.
About 5.5 million a second at four terms and six hidden units, falling to 2.4 million at
twenty-four terms, roughly with the weight count, which is what a dense forward-and-back pass
should cost.

So a fit of one million rows, 24 terms, 5 refits, 100 epochs is 500 million updates, about three
and a half minutes. Early stopping usually makes it far less: `-e` is a ceiling and the fits on
`example/nonlinear.csv` stop at 80 to 110 epochs out of 3000.

## Scoring rate

Scoring is a different shape of program: one model read, then a forward pass per case.

| | |
|---|---|
| 200,000 cases from a pipe | 0.12 s, 2.4 MB peak |
| one case, whole process | 0.4 ms, model read included |

**1.7 million cases a second** through the pipe, against 2,500 a second one process at a time. The
difference is the argument for the pipe. Neither figure comes from `bench/throughput.sh`, whose
timer has 10 ms resolution and no pipe case; both were taken by hand and should be.

## The comparison against linearr

`bench/compare.sh` is the measurement that decides whether this program should exist: both
programs on four files whose true relation is written down, so the best attainable error is
known in advance. It regenerates the data first, so the table is never quoted from a previous
run. About 20 seconds.

| data | best possible | linearr | linearr's verdict | bpnn held-out |
|---|---|---|---|---|
| exactly linear, no noise | 0 | **0** | no complaint | 4.18 |
| y = x² + 10, no noise | 0 | 13.49 | wrong shape, term named | **0.98** |
| saturating dose + interaction | 1.0 | 1.86 | wrong shape, term named | 1.21 |
| y = x₁·x₂ | 1.0 | 8.62 | wrong shape, pair *unnamed* | 1.46 |

And the option a network is usually not measured against: the same line with the one extra
column its own diagnostic asked for:

| line, plus the extra column | best possible | resid SD |
|---|---|---|
| + dose² (the check named dose) | 1.0 | 1.20 |
| + x₁·x₂ (the check could not name the pair) | 1.0 | **1.02** |

The network never won this table. What it saves is the guess about which column to add, and that
guess is cheap when the diagnostic names the term. See the README for the full argument.

## What is not measured here

No comparison against R's `nnet`, PyTorch, or scikit-learn's `MLPRegressor`. `bench/nnet.R` fits
the same data with `nnet` under the same scaling and split, and it is a fair comparison of *fit
quality*, not of speed: `nnet` minimises by BFGS over the whole training set where this takes one
stochastic step per row, and the two have no reason to land in the same place. Speed comparisons
against a Python or R stack measure the interpreter's startup as much as the arithmetic, and are
not worth publishing without saying so.

No multi-core figures. The engine is single-threaded by construction.

No figures for `bpnn_worker`, the older single-topology CLI. It exists for the XOR demonstration
and the checkpoint paths, not for production work.
