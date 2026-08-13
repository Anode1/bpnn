# AGENTS.md -- how to develop bpnn (for humans and AI agents)

`bpnn` is a backpropagation feed-forward neural network in **C99**, fitted from a CSV and scored a case
at a time. It is the companion to [linearr](https://github.com/Anode1/linearr): same file layout, same
per-group fitting, same fit-then-score split, a network where a line is the wrong shape. It is a fork
of [SMBPANN](https://github.com/Anode1/SMBPANN) with the evolutionary search removed.

## What this project is now

A tool, not a research programme. The measurement direction that justified the fork is closed and
`doc/CLOSED.md` records why; what is left is an engine that was never in doubt, a unit suite, and two
self-testing statistics tools. The work from here is engineering: guards, tests, real data, and a
memory bound that holds on files larger than memory.

The bar it is being built to is the job linearr was built from -- a length-of-stay predictor deployed
across hospitals in 2011, where a network was offered as an option and a regression was chosen. That
is the shape of problem this must handle without apology: tens of terms, hundreds of groups, rows
beyond memory, an answer that has to be reproducible digit for digit.

**SMBPANN itself is untouched and must stay that way.** The paper under review cites it as the
reference implementation and its appendix tells referees how to regenerate every experiment from that
tree. Nothing here removes or renames anything that tree depends on; this is a copy.

## The contract (read first)

- **`doc/dev/STYLE.md`** -- coding ideology, inherited unchanged: K&R/Robbins C99, one concept per
  `.c`/`.h`, stack-first, allocation only at construction, bounded strings, return codes, single-exit
  `goto` cleanup, sanitizer-gated. Non-negotiable.
- **`README.md`** -- what the program promises: the two directions (fit and score), the CSV layout it
  shares with linearr, the options, and the three failure modes it reports. Behaviour that contradicts
  it is a defect in one of them.
- **`c/bpnn.c`'s header comment** -- the specification of the tabular CLI: the input format, what each
  reported number means, and why the shipped model is the median of the seeds and not the best.
- **`c/net.h`, `c/train.h`, `c/arena.h`, `c/data.h`** -- the engine's public API.
- **The 1997 thesis** (`~/articles/BPFNN_Coursework`) -- the mathematics the engine executes;
  `c/net.c` and `c/train.c` cite it by section.

Do not change behaviour without changing these first.

## Build and test

    make            # build ./bpnn and ./bpnn_worker
    make check      # ut + cliut -- the commit gate
    make ut         # unit suite, 32 checks
    make cliut      # black-box, 120 checks: the built binary through a shell
    make ut-asan    # both suites under AddressSanitizer
    make ut-ubsan   # both suites under UBSan
    make pedantic   # -pedantic plus -Wextra -Wshadow -Wconversion; must be clean
    make tools      # resolve, pairstat and the benchmark extractors; runs both self-tests
    make clean

`make check` is the commit gate. Run the sanitizers before committing; they build the binary as well
as the suite, because the CSV reader is only reachable through it. A warning is a defect.

**`make ut` cannot see a refusal.** It links the engine and calls functions, so it never sees an exit
code, a usage text or a message. Every guard belongs in `tests/cli.sh` with one check per refusal, and
when a user hits something no test caught, the first question is which suite could have caught it.

## Module map

Sources are in `c/`, the unit suite in `tests/`, the measurement tools in `validation/`. Binaries build
at the root.

    c/common.h      smb_real (=float), SMB_MAX_LAYERS, SMB_LINE_MAX
    c/rng           deterministic xorshift PRNG. NOTE: 32-bit, period 2^32-1. Fine for a training
                    run; NOT fine for a study drawing more than ~4e9 numbers. A measurement that
                    exhausted this period silently reused draws and understated its own standard
                    error by 2.5x. If a study is that large, use a 64-bit generator and say so.
    c/act           sigmoid, tanh, relu
    c/net           flat weight matrices, forward pass, thesis init; dense and weight-shared conv1d
    c/train         the generalized delta rule with momentum; conv layers share gradients across
                    positions (gradient-checked against finite differences)
    c/conv2f        a 2-D convolutional front-end, F filters of KxK with pooling
    c/arena         marker/Mark-Release allocator
    c/data          plain-text datasets, train/test split
    c/ckpt          weight checkpoints
    c/bpnn.c        the tabular CLI: linearr's CSV in, one network per group, a case scored
    c/main.c        the older single-topology worker; prints a machine-readable RESULT line
    tests/tests.c   -DUNIT_TEST unit suite: rng act net xor arena data conv1d conv2f
    tests/cli.sh    black-box: usage, exit codes, every refusal, and the numeric invariances

    validation/resolve.c       is a comparison real, how many runs would make it real, what ceiling
    validation/pairstat.c      paired statistics with named tests -- see below
    validation/nb101_trials.c  NAS-Bench-101 tfrecord -> text, KEEPING the three training runs
    validation/nb201_extract.c the same for NAS-Bench-201's converted release
    validation/PROVENANCE_nas.md  where the benchmark data comes from and how to regenerate it

## The statistics tool

`pairstat` is the most reusable thing in the repository and should be used rather than hand-rolled
summaries. It computes Wilcoxon signed-rank as primary (exact null by dynamic programming where ties
permit, tie-corrected normal approximation otherwise, and it always reports which was used), an exact
sign test, Hodges-Lehmann estimates with distribution-free confidence intervals, a paired *t* as
secondary, Holm correction across a declared family, and a minimum detectable effect so a null result
is bounded rather than asserted. It is gated by `--selftest` against hand-computable cases.

Paired binary outcomes want an exact McNemar test, computed as a two-sided binomial on the discordant
pairs.

## What we learned the hard way, and will not repeat

These are not general advice. Each one cost real work in the predecessor.

1. **Score the target before tuning anything against it.** Hand-build the answer, score it under the
   exact objective at full seed count, and confirm it is the argmax of an enumerated family. "Beats
   the arms I chose" is a statement about the author's imagination.
2. **A rank is a statistic, not a fact.** An argmax over noisy estimates favours whichever candidate
   was luckiest. Report the winning margin beside the standard error of the scores that produced it,
   and believe the ordering only when the margin clears it.
3. **Enumerate the shapes the method actually visits**, not the shapes you expect. A family restricted
   to the tidy cases certifies nothing about the untidy ones.
4. **Check what the outcome shares with the selection statistic.** If quality is the mean of three
   measurements and the method selected on two of them, it is selecting partly on the ground truth,
   and roughly half of any apparent advantage is that overlap. Report a quantity the method could not
   see.
5. **Match the null on the property being claimed**, and include a matched-budget random control.
   Without one, an arm comparison cannot be shown to rank methods worth running at all.
6. **Verify pairing rather than asserting it.** Arms that consume different amounts of randomness
   desynchronise immediately; re-seed per run from a run-indexed seed.
7. **Pre-register the primary comparison, the equivalence margin, and what would refute it**, to a
   timestamped file, before the pilot. Score the predictions afterwards including the wrong ones.
8. **Keep an oracle.** After any refactor, rebuild from the previous source and confirm the result
   table is byte-identical. This is how you prove a fix did not silently move an archived number.
9. **Get an adversarial review from someone with no stake in the result**, and give one of them only
   the code and outputs without your conclusions. Two such reviews reversed conclusions here twice.

## The tree stays clean

Nothing derived is committed: no binaries, no per-seed run outputs, no benchmark archives, no table a
Makefile rule or a documented command rebuilds. `.gitignore` names each class and
`validation/PROVENANCE_nas.md` holds the commands. For experiments that need to cache millions of
intermediate results, use **AIS** rather than inventing storage here.

## Where the compute actually goes

Measured on a 4-core i7-1165G7, which is 8 hyperthreads and does **not** give 8 cores' throughput on
compute-bound work; sizing a run as if it did overran one estimate by 40%.

- One training run of the small engine: about 5.8 ms for 200 epochs over 96 examples.
- A benchmark lookup: free. Experiments over precomputed benchmarks are not compute-bound at all,
  and the most credible result in the predecessor cost essentially nothing.
- Build the scaled trainer when an experiment demands shapes we do not have, and not before. A
  general N-dimensional tensor library with autodiff is a framework, and frameworks are for humans.

## Memory: why linearr streams and what streaming means here

linearr holds no rows because least squares has a fixed-size sufficient statistic: fold a row into the
centered co-moments, forget it, solve once at the end. Backpropagation has no such statistic. The
gradient depends on the current weights, so the data has to be visited once per epoch and no summary
of it can stand in.

**But "visited" is not "resident".** What backpropagation needs is a re-readable stream, not memory.
The bound that is actually available here is *memory O(model), time O(epochs x rows)*, and that is the
one to build to. It differs from linearr's promise and must be stated differently: linearr is one pass
and O(1) in rows; this is E passes and O(1) in rows.

Five things break if that is done naively, and each is a decision to write down rather than discover:

1. **Scaling ranges need a first pass.** The network stores every term's range in order to scale it,
   and that pass is exactly linearr's accumulator: min, max, mean, variance per term per group, O(p)
   memory. It is also what the out-of-range guard at scoring time reads.
2. **Shuffling.** SGD wants a fresh order each epoch, and a permutation of n rows costs 8n bytes: 8 GB
   at a billion rows, which throws the bound away. The answer is a sliding shuffle buffer of B rows,
   O(B) memory, with B written into the model file so a result reproduces. The failure mode to guard
   is a file already sorted by the response or by group, where a small B shuffles nothing; that is
   detectable in the first pass and should be a warning, not a silent bad fit.
3. **The held-out split** must not consume the training PRNG. Assign it from a deterministic hash of
   (group, row ordinal), so the split is identical whichever way the training draws move. Lesson 6
   below is this same mistake in its earlier form.
4. **Refits.** `-s` seeds must not mean S passes over the file. Hold S sets of weights, update all of
   them from one row stream with a per-seed draw order, and the spread costs one pass. Whether the
   held-out split is shared across seeds is a decision with consequences: sharing it isolates the
   refit noise from split noise, which is what the reported spread claims to be.
5. **Epochs stop being free.** Three thousand epochs over a billion rows is not a run anybody will
   make. Past roughly a million rows the control has to become a budget plus early stopping on the
   held-out error, and the parsing cost has to go: pack the CSV once into a fixed-width binary cache
   and let each epoch be a sequential read of it.

Per-group memory is bounded by the topology and the number of groups, never by rows, and the way to
state that honestly is linearr's: a `--footprint` that prints the figure for a given shape rather than
a sentence asking to be trusted. When G x model does not fit, fit the groups in K batches and pay K
passes; that trade is fine as long as it is printed.

## Roadmap

The direction is a companion tool for linearr, up to the shape of the 2011 length-of-stay job. In
order, because each step is the ground the next one stands on:

1. ~~**Guards.**~~ Done. The reader refused nothing: a non-numeric term read as zero, a row with an
   unparseable response was dropped without a word, and `nan` and `inf` parsed and propagated into
   every weight with exit 0. Each is now refused naming the line and the column, along with a field
   count that disagrees with the header, a duplicated term name, a group code too long to store, an
   unknown option, an option missing its value, and a case that leaves a term unnamed.
2. ~~**A black-box suite.**~~ Done: `tests/cli.sh`, 91 checks, one per refusal plus the numeric
   invariances (a term in millions and a response offset by 1e8 must fit identically, and the
   printed prediction must still resolve the response, which at `%g`'s six digits it did not).
3. ~~**Streaming.**~~ Done as `--stream`, with `--footprint` and `scripts/scale.sh`. Measured over a
   tenfold increase in rows: 7.2 MB to 8.7 MB streaming, against 53 MB to 514 MB holding them. The
   rise is the shuffle window filling and stopping at `--buffer`. Points 1 to 4 of the section above
   are implemented. Point 5 is half done: -e is a ceiling on the default path, where the fit stops
   when the rows kept back for it stop improving, but --stream still runs every epoch and says so,
   and the CSV is re-parsed on both setup passes rather than cached across runs.

   Early stopping on the streaming path needs the stop rows evaluated inside the training pass,
   since they are already being skipped there, and a second hash bit to divide the held rows into
   a stop half and a reported half. That is the next piece.
4. **Real data.** The 24-term length-of-stay shape from linearr's example data is the target; FSDD
   (`data/fsdd/PROVENANCE.md`) is the non-tabular check the engine already has paths for.
5. **A classifier**, which needs more than one output and a softmax with cross-entropy. Not before the
   four above.
