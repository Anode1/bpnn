# bpnn: a neural network in C for the data a straight line gets wrong

### It reports three of the ways a network misleads you, and it tells you when a line would have been the better buy

A backpropagation feed-forward network as a command-line program. Reads a CSV and returns a fitted
model; reads a case and returns a prediction.

**This answers a question somebody asked in 2011 and nobody measured.** Fitting a least-squares
length-of-stay predictor for industry, the author put it to the scientist he was working with that
length of stay might not be linear, and offered a neural network instead of the usual repairs: log the
response, bin the age, add a squared column. It was too early, and the idea went nowhere. The two
programs now exist, so the comparison can be run. [What it showed](#what-the-comparison-shows).

## How it pairs with linearr

[linearr](https://github.com/Anode1/linearr) fits straight lines in closed form, then checks whether
the residuals it leaves still depend on a term. When they do, a line is the wrong shape, and linearr
says so and stops, because a line is all it has. This is the program you run next. Same CSV layout,
same per-group fitting, same fit-then-score split, a network in place of the line.

    ./bpnn -t data.csv > model.txt      fit one network per group
    ./bpnn -c model.txt A x=3           score one case
    scripts/escalate.sh data.csv        the line first, the network only if the line is wrong shape
    bench/compare.sh                    the measured comparison, against a known noise floor

The order in that third line is the point. Fit the line first and escalate only when the line's own
diagnostic says the shape is wrong. "We used a neural network" is not a modelling decision.

## What the comparison shows

`bench/compare.sh` puts both programs on four files whose true relation is written down and whose noise
floor is therefore known in advance, so neither program is graded against the other's weaknesses. Every
number below comes from that script. RMSE, lower is better.

| data | best possible | linearr | linearr's verdict | bpnn held-out |
|---|---|---|---|---|
| exactly linear, no noise | 0 | **0** | no complaint | 4.51 |
| y = x² + 10, no noise | 0 | 13.49 | wrong shape, term named | **0.96** |
| saturating dose + interaction | 1.0 | 1.86 | wrong shape, term named | 1.26 |
| y = x₁·x₂ | 1.0 | 8.62 | wrong shape, pair *unnamed* | 1.44 |

And then the option a network is usually not measured against: the same line, with the one extra column
its own diagnostic asked for. Still closed form, still exact, still readable coefficients.

| line, plus the extra column | best possible | resid SD |
|---|---|---|
| + dose² (the check named dose) | 1.0 | 1.20 |
| + x₁·x₂ (the check could not name the pair) | 1.0 | **1.02** |

So the network never won this table outright. A correctly specified line reaches the noise floor, gets
there in closed form with no seed and no spread, and hands you coefficients you can give to somebody.
What the network buys is not accuracy. It is **not having to guess which column to add**, and that
guess is cheap exactly when the diagnostic names the term for you. It stops being cheap when the check
can only say that some pair interacts, because the candidates then number p(p−1)/2: one pair at two
terms, 276 at twenty-four. That gap is where a network earns its place, and it is a narrower place than
the usual framing suggests.

### The measurement direction is closed

An earlier goal of this fork was measuring how much of a reported architecture improvement survives
replication. That direction is closed and produced no publishable finding: its headline numbers were
either prior art or inflated. `doc/CLOSED.md` records every claim and what became of it, so that none
of them gets revived from memory. The tooling it left behind is still here and still useful: `resolve`
(is this comparison real, and how many runs would make it real), `pairstat` (paired tests behind a
self-test gate), and the per-seed extractors for both NAS benchmarks.

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
