# bpnn: a neural network in C, for the data a straight line gets wrong

### It warns when the fit has memorised the rows, when refitting moves the answer, and when a case falls outside the data

A backpropagation feed-forward network as a command-line program. Reads a CSV and returns a fitted
model; reads a case and returns a prediction. No dependencies beyond libm, and the same CSV layout
its sibling [linearr](https://github.com/Anode1/linearr) reads.

    make
    ./bpnn -t mydata.csv > model.txt          # fit one network per group
    ./bpnn -c model.txt A dose=5 age=60       # score a case against it

**This answers a question somebody asked in 2011 and nobody measured.** Fitting a least-squares
length-of-stay predictor for industry, the author put it to the scientist he was working with that
length of stay might not be linear, and offered a neural network instead of the usual repairs: log
the response, bin the age, add a squared column. It was too early and the idea went nowhere. Both
programs now exist, so the comparison can be run. It is in
[When a network is worth it](#when-a-network-is-worth-it-and-when-a-line-with-one-more-column-is-worth-more),
and the network does not win it.

## How it pairs with linearr

linearr fits straight lines in closed form and then reads its own residuals: with `--residuals` it
checks whether what is left over still depends on a term. When it does, a line is the wrong shape,
and linearr says so and stops, because a line is all it has. That warning is this program's cue, and
`scripts/escalate.sh` is the pipeline that joins the two. Give it data a line fits:

    $ scripts/escalate.sh ~/linearr/example/simple-train.csv
    == step 1: fit a straight line
       residuals: /dev/null
       reading: column 1 is the group, 'minutes' is the value being predicted, and the other 2 columns are terms. Use -y NAME if that is the wrong column
       fit: 2 groups, 13 rows, worst R2=1.0000, worst resid SD<5.17e-07, least df=3, worst cond=1.03 (normal equations)

    == a line was enough
    linearr found no leftover dependence on any term or on the prediction, so there is no
    measured reason to fit a curve here. The coefficients are in ~/linearr/example/simple-train.linearr.csv and you can read
    them. Least squares solves in closed form, so that answer is exact and has no seed;
    escalating anyway would trade those away and buy nothing this data can show.

It stops there, exit 0, and fits no network. Given data a line gets wrong it escalates, exit 1, and
leaves both models on disk. The line is tried first because a correct line is exact, has no seed and
has readable coefficients, and the escalation happens on a measurement rather than on preference.

One weakness of the script, also written where the trigger is: linearr exits 0 whether or not the
shape is wrong, so the escalation is a text match on the warning. If that wording changes, the script
stops escalating and reports that a line was enough. That is the wrong answer, and it is not
announced. The script therefore refuses to conclude anything when linearr printed no `fit:` summary
at all. The fix belongs upstream, in a shape column or a distinct exit status.

## What backpropagation is

**What it does.** A network of weighted sums with a squashing function between them. It is fitted by
taking the error at the output, passing it back through the layers, and moving each weight by its own
share of that error. The method is Rumelhart, Hinton and Williams, 1986. It fits a shape that was not
written down in advance. The arithmetic here follows a 1997 seminar thesis' tensor derivation, cited
by section in `c/net.c` and `c/train.c`.

**What `c/train.c` is doing, basically.** With `a` the activations of a layer, `g'` the derivative of
the squashing function and `d` the target, the whole of the generalized delta rule is four lines:

    beta[L] = (d - a[L]) * g'(a[L])                     # output layer: what the answer missed by
    beta[l] = g'(a[l]) * W[l+1]' beta[l+1]              # the same error, pushed back a layer
    dw[l]   = rate * beta[l] a[l-1]' + momentum * dw[l] # this step, plus a share of the last one
    W[l]   += dw[l]

Four things turn them into the 139 lines of code in `c/train.c`. Layers are flat arrays rather than a
matrix library, so the products are written out. A convolutional layer shares one filter across every
position, so its gradient is a sum over those positions. Every scratch buffer is allocated once in
`trainer_new`, so the training loop allocates nothing. And the derivative is checked rather than
assumed: `make ut` compares it against finite differences for the dense layer and both convolutional
layers. A gradient-sharing bug passes a loss curve and fails that check.

## Fit and score

Two commands, and the same split as linearr: standard output is the model, standard error is the
commentary, so the redirect is the whole workflow.

    $ ./bpnn -t example/nonlinear.csv > model.txt
    group        rows   held  weights  epochs      train   held-out   refit sd      floor
    001           400     50       18     225     1.0544     1.1134   0.055775     0.1546
    002           400     50       18     250      1.116      1.133    0.15539    0.43072

    $ ./bpnn -c model.txt 001 dose=5 age=60
    001 los = 16.778
    held-out RMSE at fit time 1.11337, spread over refits 0.0557752

`-t` fits every group in one pass; `-c` names the model to score against and is required, because
which model produced a number is part of the number. **Column 2 is the value being predicted**, and
columns 3 onward are the terms, exactly as in linearr. Nothing in the data can say which column you
meant.

The four numbers on the right are the report. `train` is the error on the rows the fit saw, `held-out`
the error on the rows it did not. `refit sd` is the spread of the held-out figure over repeated fits of
the same configuration. `floor` is 2.77 times that spread: two configurations closer together than the
floor cannot be told apart on this data, whichever way the comparison came out.

Scoring prints the fit-time error and the spread again on every prediction. A prediction given to four
decimals, from a model whose own error is 1.11, would otherwise read as more precise than it is.

## Groups

**A group is one fitted network.** Rows sharing a group code are fitted together and get their own
network, so one file and one pass produce one model per subset. This is linearr's convention, and it
matters more here. Pool the groups and a network can pick up the group's identity from whatever term
happens to correlate with it. Nothing in the output shows when that has happened.

## Options

| option | default | meaning |
| --- | --- | --- |
| `-t FILE` | | fit: one network per group, model to stdout, report to stderr |
| `-c FILE` | *(required to score)* | the fitted model to score a case against |
| `-H N` | 6 | hidden units |
| `-e N` | 3000 | epochs, as a ceiling; see `--patience` |
| `--patience N` | 8 | stop after N checks, 25 epochs apart, with no improvement; 0 runs every epoch |
| `-s N` | 5 | refits, which is what the spread and the floor are measured over |
| `-r X` | 0.3 | learning rate |
| `-m X` | 0.9 | momentum |
| `-a NAME` | tanh | hidden activation: `sigmoid`, `tanh` or `relu`; the output stays a sigmoid |
| `--holdout X` | 0.25 | fraction of rows kept out of the fit |
| `--stream` | | fit without holding the rows; see [Memory](#memory-and-files-larger-than-it) |
| `--buffer N` | 65536 | rows in the shuffle window under `--stream` |
| `--footprint T G` | | what a fit of T terms and G groups costs in memory |
| `--selftest` | | check the arithmetic and exit |

`--holdout 0` disables the split. The reported error then says nothing about generalization, and the
report says so. It is there for the case where the relation is noiseless and known.

## Memory, and files larger than it

By default every row is held in memory. `--stream` does not hold them: it reads the file twice, once
for the ranges and once to write a cache of the scaled rows, and then reads that cache once per epoch.
Memory is then the networks plus the shuffle windows, and neither depends on the number of rows.

Least squares needs one pass because its objective has a fixed-size sufficient statistic.
Backpropagation has none: the gradient depends on the current weights, so every epoch has to see the
rows again. Streaming removes the storage, not the passes.

`scripts/scale.sh` measures it, fitting the same model over ten times the rows (two terms, four
groups):

    $ scripts/scale.sh 100000
                                            100000 rows 1000000 rows
    --stream, peak RSS (kB)                        8704         8576
    default, peak RSS (kB)                         7620        56772

The default path grows with the rows. `--stream` does not. But note the left-hand column: at 100,000
rows `--stream` costs *more*, because it pays for one shuffle window per refit whatever the file
size. It is worth turning on when the row store stops fitting, not before. `--footprint TERMS GROUPS`
prints both figures for a shape before you run anything:

    $ ./bpnn --footprint 24 400
    24 terms, 400 groups, 6 hidden units, 5 refits

    fitting with --stream
      per group per refit    1.25 kB
      networks in total      2.9 MB
      shuffle windows        33.8 MB
      total                  36.7 MB

    fitting without --stream, add the row store, which does take a row count:
      per row                208 bytes

    The --stream cache is a temporary file of 108 bytes a row, removed on exit.

A row costs `(terms + 1) * 8 + 8` bytes held, so the crossover is around 200,000 rows at two terms
and 35,000 at twenty-four. If the store does not fit, the fit says what it was holding and names the
flag rather than printing `out of memory`; and once the store passes 64 MB the report prints what it
cost, since that is the figure that decides whether the next file will fit.

The two paths give different numbers, and neither is an approximation of the other. The default
shuffles each group's rows completely every epoch. `--stream` shuffles through a window of `--buffer`
rows, so the training order differs. Both repeat byte for byte from the same input, and the default
is the one every number elsewhere in this file comes from.

One case to know about: a file already sorted by the response, with a window smaller than the file.
The window then leaves that order nearly intact and the fit trains on it. That is reported rather
than left to show up as a worse fit:

    $ ./bpnn -t rows.csv --stream --buffer 4
    bpnn: rows.csv is sorted by y, or nearly so, and the shuffle window holds
    4 of its 40 rows, so each pass trains on close to that order. Shuffle
    the file, or raise --buffer above the row count.

## Three ways the fit can be wrong

linearr checks for collinearity, exhausted degrees of freedom and lost digits. A network fails in
different ways. These three are the ones this program can check.

**The fit memorised the rows.** Two coefficients cannot memorise thirteen points. Thirty weights can,
and then the training error measures recall, not accuracy. So the rows are split, and the error on
the fitted rows and on the held-out rows are printed side by side. If the second is much larger, the
model learned this table rather than the relation behind it.

    $ ./bpnn -t ~/linearr/example/simple-train.csv > /dev/null
    group        rows   held  weights  epochs      train   held-out   refit sd      floor
    A               7      2       18    3000 1.4796e-05     5.2607     1.5361     4.2577
      18 weights fitted to 5 training rows. There are more free
      parameters than examples, so some of what it learned is the rows
      themselves. Reduce -H, or use linearr if the relation may be linear.
      held-out error is 355556.7x the training error: this network is fitting
      the rows rather than the relation. Fewer hidden units, or more rows.

**When it stops.** `-e` is a ceiling, not a count. The fit checks its error every 25 epochs on rows
kept back for that purpose, keeps the weights from the best check, and gives up after `--patience`
checks with no improvement. The `epochs` column says how many it used. On `example/nonlinear.csv`
that is 225 and 250 out of 3000, the fit takes 0.20 s instead of 1.51 s, and the held-out error is
*lower*: 1.11 and 1.13, against 1.15 and 1.26 for the full 3000. The extra epochs were overfitting.

The rows that decide when to stop are not the rows the error is reported on. Choosing a stopping
point by a number makes that number optimistic by however much was selected for, so the held-out
rows are split in half: one half stops the fit, the other is reported and is not looked at during it.
That halves the reported `held` count, which is the cost. Below four rows either way the split means
nothing and the fit runs its full `-e` epochs, which is what happens on the seven-row groups above.
`--patience 0` turns the whole thing off.

**Refitting gives a different answer.** Least squares has one solution. A network starts from random
weights and shuffles its examples, so fitting the same rows twice gives two models and two scores.
`-s` sets the number of refits. The spread over them is printed next to the error, and 2.77 times
that spread is the resolution: two configurations closer than that cannot be told apart on this data.
The model written out is the median refit, not the best one. The best is optimistic by however much
was selected for, so both numbers go in the model file.

**A case can fall outside the data.** linearr lists this as something a regression cannot check. This
program can check it, because scaling an input needs the range it was trained on, and that range is
stored in the model. Outside the range the units saturate and the prediction goes flat:

    $ ./bpnn -c model.txt 001 dose=25 age=60
    001 los = 22.0926
    held-out RMSE at fit time 1.11337, spread over refits 0.0557752
    OUTSIDE THE TRAINING RANGE: dose=25, trained on [0.0118, 9.9804], out by 1.51 of that span
    A network does not extrapolate. Past the range above its units saturate and it
    returns a flat value with no warning of its own, so treat this number as a guess.

Three it cannot check: a term missing from the table, rows that are not independent of each other,
and a relation that changed after the training rows were collected.

## What it refuses to read

A bad value in the input is not caught later. An empty field read as a zero, or one `nan`, spreads
through the training to every weight in the group and to every number the model then prints. So these
are refused when the file is read, with the line and the column named:

    $ awk 'NR==118{sub(/,[^,]*$/,",nan")}1' example/nonlinear.csv > rows.csv
    $ ./bpnn -t rows.csv
    bpnn: the term 'age' is 'nan' on line 118. A nan or an infinity reaches every
    weight in the group and every number the model then prints.
    $ echo $?
    1

Also refused: a row whose field count disagrees with the header, an empty field, a header naming the
same term twice, a group code too long to store whole (two such codes would otherwise be merged into
one fit), an unknown option, an option missing its value, and a case that leaves a term unnamed, since
scoring it as zero answers a different question. `tests/cli.sh` has one check per refusal. None of
them can be reached from the unit suite.

## When a network is worth it, and when a line with one more column is worth more

`bench/compare.sh` puts both programs on four files whose true relation is written down, so the best
attainable error is known in advance and neither program is graded on the other's weaknesses. It
regenerates the data first, so the table is never quoted from a previous run. About twenty seconds on
a 4-core i7-1165G7; RMSE, lower is better.

| data | best possible | linearr | linearr's verdict | bpnn held-out |
|---|---|---|---|---|
| exactly linear, no noise | 0 | **0** | no complaint | 4.51 |
| y = x² + 10, no noise | 0 | 13.49 | wrong shape, term named | **0.96** |
| saturating dose + interaction | 1.0 | 1.86 | wrong shape, term named | 1.13 |
| y = x₁·x₂ | 1.0 | 8.62 | wrong shape, pair *unnamed* | 1.23 |

Then the comparison a network is usually not put through: the same line, with the one extra column
its own diagnostic asked for. Still closed form, still exact, still readable coefficients.

| line, plus the extra column | best possible | resid SD |
|---|---|---|
| + dose² (the check named dose) | 1.0 | 1.20 |
| + x₁·x₂ (the check could not name the pair) | 1.0 | **1.02** |

The network never won this table. A correctly specified line reaches the noise floor, has no seed and
no spread, and gives coefficients you can hand to somebody. What a network saves is the guess about
which column to add. That guess is cheap when the diagnostic names the term, and expensive when it
can only report that some pair interacts: the candidate pairs number p(p−1)/2, so one pair at two
terms and 276 at twenty-four. The second case is the one this program is for.

## The example data

Everything in `example/` is synthetic and generated by `example/mknonlinear.c` from a fixed seed, so
the files regenerate byte for byte. `nonlinear.csv` is a saturating dose response plus an interaction,
where linearr's check names the term; `interaction.csv` is `y = x1*x2`, where it can only say that
something interacts. The linear cases in the table above are linearr's own example files, read from
`~/linearr/example`. No real data is distributed with this project.

## Build and test

    make            build ./bpnn and ./bpnn_worker
    make check      ut + cliut: what must pass before a commit
    make ut         32 unit checks: rng, act, net, xor, arena, data, conv1d, conv2f
    make cliut      120 black-box checks: the built binary, through a shell
    make ut-asan    both suites under AddressSanitizer
    make ut-ubsan   both suites under UndefinedBehaviorSanitizer
    make pedantic   -pedantic with -Wextra -Wshadow -Wconversion; must be clean
    make tools      the measurement tools; runs the two self-tests
    make clean

A warning is a defect. `make check` gates every commit, and both sanitizers run before one. The
sanitizer targets build the binary as well as the suite, since the CSV reader is only reached through
the binary. `./bpnn --selftest` is a smaller check the binary carries itself: input scaling at both
ends, the target round-trip, the resolution coefficient, and a parabola fitted to a known error.

## Where this is the right tool, and where it is not

**It fits when** a line has been tried and its residual check says the shape is wrong; the machine has
no Python and is not going to get one; the answer has to be reproducible byte for byte from a seed;
or the job is many small independent fits, one per group, where a general-purpose library spends more
time dispatching operations than computing them.

**Reach for something else when** you need any of: regularization beyond early stopping, dropout, batch
normalisation, an optimiser other than momentum, more than one output, class probabilities, embeddings,
convolution over anything but the built-in 1-D and 2-D front ends, a GPU, or automatic differentiation.
None of that is here and none of it is planned. PyTorch, scikit-learn's `MLPRegressor`, and gradient
boosting for tabular data in particular do all of it well, and on tables of this shape boosting usually
beats a small network.

**Two limits.** The engine computes in `smb_real`, which is `float`, so about seven digits. linearr
validates against NIST reference values to eleven; nothing here should be trusted past six. And the
build ceilings are 64 terms and 512 groups, both fixed at compile time.

## The measurement tools

These are what a closed research direction (below) left behind. Each is one self-contained `.c` with
no dependencies and no link to the engine.

**`validation/resolve.c`** answers three questions from repeated scores, and refuses to guess when
given one run per candidate: is candidate 0 really better than the others, how many runs per candidate
would resolve a difference of a given size, and what is the best rank correlation any predictor could
score against labels this noisy.

**`validation/pairstat.c`** is the most reusable of them. Paired statistics with named tests:
Wilcoxon signed-rank as primary, exact by dynamic programming where ties permit and tie-corrected
normal otherwise, always reporting which it used; an exact sign test; Hodges-Lehmann estimates with
distribution-free intervals; a paired *t* as secondary; Holm correction across a declared family; and a
minimum detectable effect, so a null result is bounded rather than asserted. Both tools are gated by a
self-test against hand-computable cases and `make tools` runs it.

**`validation/nb101_trials.c`** and **`validation/nb201_extract.c`** recover the per-run values from
NAS-Bench-101 and NAS-Bench-201. Both benchmarks trained every architecture three times and both
distributed tables average those runs away, which is the right table for ranking architectures and the
wrong one for studying the noise. `validation/PROVENANCE_nas.md` has the URLs and the commands; the
archives themselves are not in the repository.

## The measurement direction is closed

An earlier goal of this fork was measuring how much of a reported architecture improvement survives
replication. It produced no publishable finding: the headline numbers were either prior art or
inflated, and `doc/CLOSED.md` lists every claim and what became of it, so that none of them gets
revived from a half-memory. Two numbers in particular are wrong wherever they still appear: the
rank-correlation ceiling of 0.084 is 0.293, and the indifference class of 3,558 architectures is 26.

## Origin

Forked from [SMBPANN](https://github.com/Anode1/SMBPANN), which used an evolutionary search over
topology to ask which pieces of a convolution a search recovers unaided. The engine, the unit suite and
the statistics tooling carried over; the search did not, because on affordable problem sizes it was a
weak hill-climber whose stalling confounded the measurements built on it, and on a public benchmark it
lost to matched-budget random search. SMBPANN is untouched and remains the reference implementation
for the paper that cites it.

## The rest of it

| document | what is in it |
| --- | --- |
| `c/bpnn.c` | the header comment is the specification: the input format, the three failure modes and what each check computes |
| `AGENTS.md` | the operating manual, and the specific mistakes this line has already paid for |
| `doc/dev/STYLE.md` | the C ideology: one concept per file, stack-first, allocation at construction, single-exit cleanup |
| `doc/CLOSED.md` | every claim the closed direction made, and what became of it |
| `bench/compare.sh` | the comparison above, and why each of the four files is in it |
| `validation/PROVENANCE_nas.md` | where the benchmark data comes from and the commands that regenerate it |

## See also

- [linearr](https://github.com/Anode1/linearr): the straight line, in closed form, which is the program
  to run first.
- [ais](https://github.com/Anode1/ais): the associative-memory engine these conventions come from.

## License

BSD 2-Clause; see `LICENSE`.
