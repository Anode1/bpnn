# AGENTS.md -- how to develop bpnn (for humans and AI agents)

`bpnn` is a backpropagation feed-forward neural network in **C99**, and a set of tools for measuring
how much of a neural-network experiment's reported result is real. It is a fork of
[SMBPANN](https://github.com/Anode1/SMBPANN) with the evolutionary search removed.

## Why this fork exists

SMBPANN used an evolutionary search as an instrument for asking which pieces of a convolution a search
recovers on its own. That line produced one published paper, one retracted draft, and a third result
that survived three rounds of adversarial review only after shrinking considerably. The common factor
in the failures was the search itself: on the spaces we could afford, the GA is a weak hill-climber
that stalls, and nearly every measurement about it turned out to be confounded by that stalling rather
than informative about anything else.

So the search is gone and the engine stays. What we keep is the part that was never in doubt: a
gradient-checked trainer, a unit suite, and a statistics tool that self-tests before it reports a
p-value.

**SMBPANN itself is untouched and must stay that way.** The paper under review cites it as the
reference implementation and its appendix tells referees how to regenerate every experiment from that
tree. Nothing here removes or renames anything that tree depends on; this is a copy.

## The contract (read first)

- **`doc/dev/STYLE.md`** -- coding ideology, inherited unchanged: K&R/Robbins C99, one concept per
  `.c`/`.h`, stack-first, allocation only at construction, bounded strings, return codes, single-exit
  `goto` cleanup, sanitizer-gated. Non-negotiable.
- **`net.h`, `train.h`, `arena.h`, `data.h`** -- the public API.
- **The 1997 thesis** (`~/articles/BPFNN_Coursework`) -- the mathematics the engine executes; `net.c`
  and `train.c` cite it by section.

## Build and test

    make            # build ./bpnn
    make ut         # unit suite, 32 checks -- the commit gate
    make ut-asan    # under AddressSanitizer
    make ut-ubsan   # under UBSan
    make pedantic   # -pedantic plus -Wextra -Wshadow -Wconversion; must be clean
    make tools      # pairstat, nb101_trials, nb101_signal; runs pairstat's self-test
    make clean

`make ut` is the commit gate. Run the sanitizers before committing. A warning is a defect.

## Module map

    common.h        smb_real (=float), SMB_MAX_LAYERS, SMB_LINE_MAX
    rng             deterministic xorshift PRNG. NOTE: 32-bit, period 2^32-1. Fine for a training
                    run; NOT fine for a study drawing more than ~4e9 numbers. A measurement that
                    exhausted this period silently reused draws and understated its own standard
                    error by 2.5x. If a study is that large, use a 64-bit generator and say so.
    act             sigmoid, tanh, relu
    net             flat weight matrices, forward pass, thesis init; dense and weight-shared conv1d
    train           the generalized delta rule with momentum; conv layers share gradients across
                    positions (gradient-checked against finite differences)
    conv2f          a 2-D convolutional front-end, F filters of KxK with pooling
    arena           marker/Mark-Release allocator
    data            plain-text datasets, train/test split
    ckpt            weight checkpoints
    main.c          the CLI worker; trains a topology and prints a RESULT line
    tests.c         -DUNIT_TEST unit suite: rng act net xor arena data conv1d conv2f

    validation/pairstat.c      paired statistics with named tests -- see below
    validation/nb101_trials.c  NAS-Bench-101 tfrecord -> text, KEEPING the three training runs
    validation/nb101_signal.c  selection inflation against search effort
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

## Storage for large experiments

For experiments that need to cache millions of intermediate results, use **AIS** rather than inventing
storage here. This repository stays a compute-and-measure tree; per-seed outputs live as
`scratch_*.out` files and anything larger belongs in AIS.

## Where the compute actually goes

Measured on a 4-core i7-1165G7, which is 8 hyperthreads and does **not** give 8 cores' throughput on
compute-bound work; sizing a run as if it did overran one estimate by 40%.

- One training run of the small engine: about 5.8 ms for 200 epochs over 96 examples.
- A benchmark lookup: free. Experiments over precomputed benchmarks are not compute-bound at all,
  and the most credible result in the predecessor cost essentially nothing.
- Build the scaled trainer when an experiment demands shapes we do not have, and not before. A
  general N-dimensional tensor library with autodiff is a framework, and frameworks are for humans.

## Roadmap

1. **Benchmark noise.** How much of a NAS benchmark's architecture ranking is training noise? The
   per-seed tables are already extracted, so this is table lookups: how often the ranking of two
   architectures flips between training seeds, what fraction of published-sized differences sit
   inside single-run noise, and what that implies for any result reported from one training run.
   Needs no search and no training, so nothing is confounded by an optimiser.
2. Undecided, pending (1). The candidate with an audience is measuring how much of test-time
   training's reported gain survives an honest held-out split, since its per-problem choices are
   made on the same handful of examples it adapts on.
