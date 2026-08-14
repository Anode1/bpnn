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
[When a network is worth it](#when-a-network-is-worth-it), and the network does not win it.

## Which of the two to use

You have a table: one row per case, some columns describing it, one column to predict. The
decision is four steps and the first three do not involve this program.

**1. Fit the line.** `linearr -t data.csv`. It is exact, has no seed, and gives coefficients you
can say out loud: *dialysis adds 4.2 days*. A clinician can agree or disagree with that sentence,
which matters more than accuracy.

**2. Read what it says about its own residuals.** With `--residuals` it checks whether what is
left over still depends on a column. **If it says nothing, stop.** The relation is a line and a
network can only do worse: on the length-of-stay file whose truth is linear, the line reaches the
noise floor at 1.03 and this program gets 1.43.

**3. If it complains, it usually names the column.** Add that column's square and refit the line.
On every file in `bench/` that reached the noise floor in closed form, and you still have readable
coefficients. This step is the one most people skip, and it is usually the right answer.

**4. Only then reach for this program** — when the line is wrong and nobody can say what to add.
That is the saturating shape, where the fifth procedure does not add what the first did: the line
scores 2.57 there and this scores 1.15.

`scripts/escalate.sh data.csv` does all four and stops at the right one, on a measurement rather
than on taste.

**The case where neither works, stated here because it is the most likely one.** When two columns
*interact* — two procedures together costing more than apart — the line is wrong, linearr's check
stays **silent**, and this program does not find it either at any capacity. Measured on a
length-of-stay shape: line 1.75, network 1.65, and the line *told which pair* 1.07. With 24
indicators at 18% prevalence, only 3% of rows carry both, so the effect lives in a few dozen rows.
Neither program will name that pair. A clinician who can is worth more than either, and the
measurement is in [The length-of-stay shape](#the-length-of-stay-shape).

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
has readable coefficients, and the escalation happens on a measurement.

One weakness of the script, written where the trigger is: linearr exits 0 whether or not the shape is
wrong, so the escalation is a text match on the warning. If that wording changes, the script silently
stops escalating and reports that a line was enough. It therefore refuses to conclude anything when
linearr printed no `fit:` summary at all. The fix belongs upstream, in a shape column or a distinct
exit status.

## What backpropagation is

A network of weighted sums with a squashing function between them. It is fitted by taking the error
at the output, passing it back through the layers, and moving each weight by its own share of that
error. The method is Rumelhart, Hinton and Williams, 1986. It fits a shape that was not written down
in advance. The arithmetic here follows a 1997 seminar thesis' tensor derivation, cited by section in
`c/net.c` and `c/train.c`.

With `a` the activations of a layer, `g'` the derivative of the squashing function and `d` the
target, the whole of the generalized delta rule is four lines:

    beta[L] = (d - a[L]) * g'(a[L])                     # output layer: what the answer missed by
    beta[l] = g'(a[l]) * W[l+1]' beta[l+1]              # the same error, pushed back a layer
    dw[l]   = rate * beta[l] a[l-1]' + momentum * dw[l] # this step, plus a share of the last one
    W[l]   += dw[l]

Four things turn them into the 139 lines of code in `c/train.c`. Layers are flat arrays rather than a
matrix library, so the products are written out. A convolutional layer shares one filter across every
position, so its gradient is a sum over those positions. Every scratch buffer is allocated once in
`trainer_new`, so the training loop allocates nothing. And the derivative is checked: `make ut`
compares it against finite differences for the dense layer and both convolutional layers. A
gradient-sharing bug passes a loss curve and fails that check.

## Fit and score

Two commands, and the same split as linearr: standard output is the model, standard error is the
commentary, so the redirect is the whole workflow.

    $ ./bpnn -t example/nonlinear.csv > model.txt
    reading example/nonlinear.csv: column 1 is the group, 'los' is the value being predicted, and
    the other 2 columns are the terms: dose age
    Use -y NAME if that is the wrong column
    group        rows   held  weights  epochs      train   held-out   refit sd      floor   expl
    001           400     50       25      80     1.0728      1.135   0.046012    0.12754    92%
    002           400     50       25       9     1.1515     1.2135    0.16096    0.44616    92%

    $ ./bpnn -c model.txt 001 dose=5 age=60
    this model's held-out RMSE at fit time 1.12561, spread over refits 0.0460124
    001,16.9632

Standard output is the prediction and nothing else, so scoring is a pipeline stage. With no case on
the command line it reads them from standard input, one per line, group first and then one value per
term in the model's order:

    $ printf '001,5,60\n001,7,45\n002,3,70\n' | ./bpnn -c model.txt
    001,16.9632
    001,16.951
    002,17.874

A header line is skipped if its second field is the first term's name, so a cases file may carry one.
A *training* file cannot be fed back as cases: it still has the response column, and the field count
will not match.
Every caveat -- a term outside its fitted range, a saturated prediction, a group the model does not
have -- goes to standard error, named by the line it came from:

    $ printf '001,25,60\n' | ./bpnn -c model.txt > predictions.csv
    -:1: dose=25 is outside the fitted range [0.0118, 9.9804], by 1.51 of it
    -:1: a network does not extrapolate; past its range the units saturate

`-t` fits every group in one pass; `-c` names the model to score against and is required, because
which model produced a number is part of the number. **Column 2 is the value being predicted** and
columns 3 onward are the terms, exactly as in linearr. Nothing in the data can say which column you
meant, so the first line of the report says which one was taken and `-y NAME` overrides it. A file
saved in another order is the commonest mistake there is, and it fits cleanly and exits 0.

The numbers on the right are the report. `train` is the error on the rows the fit saw, `held-out` the
error on the rows it did not, and `expl` is the share of those rows' variance the fit accounts for:
zero means the group's own mean would have done as well. `weights` counts every fitted parameter,
biases included. `refit sd` is the spread of the held-out figure over repeated fits of the same
configuration, and `floor` is 2.77 times it.

The floor is a rough scale. It is the 95% critical value for a difference between two single fits, so
a true difference exactly equal to it is found about half the time; it is computed from a spread
estimated on 4 degrees of freedom at the default `-s 5`, which is itself uncertain by about a third;
and the number printed beside it is a mean over refits rather than a single fit. Two configurations
differing by less than the floor are not distinguished by this data. Two differing by more are worth
a closer look. The paired comparison in [`doc/DIAGNOSTICS.md`](doc/DIAGNOSTICS.md) does that closer
look properly and is several times sharper.

Scoring prints the fit-time error and the spread again on every prediction. A prediction given to four
decimals, from a model whose own error is 1.11, would otherwise read as more precise than it is.

## Groups

A group is one fitted network. Rows sharing a group code are fitted together and get their own
network, so one file and one pass produce one model per subset. This is linearr's convention, and it
matters more here. Pool the groups and a network can pick up the group's identity from whatever term
happens to correlate with it. Nothing in the output shows when that has happened.

## Options

| option | default | meaning |
| --- | --- | --- |
| `-t FILE` | | fit: one network per group, model to stdout, report to stderr |
| `-c FILE` | *(required to score)* | the fitted model to score against; with no case on the command line, cases are read from stdin |
| `-y NAME` | column 2 | the column to predict |
| `-H N`, `--size N` | 6 | hidden units (`size` is R `nnet`'s name for it) |
| `-e N` | 3000 | epochs, as a ceiling; see `--patience` |
| `--patience N` | 50 | stop after N epochs with no improvement on the rows kept back to judge it; 0 disables |
| `--min-rows N` | by shape | skip a group with fewer rows than its parameter count needs (31 at minimum) |
| `-s N` | 5 | refits, which is what the spread and the floor are measured over |
| `-r X` | 0.3 | learning rate |
| `-m X` | 0.7 | momentum. With `-r` this is one knob: `r/(1-m)` is the step a weight takes in the limit |
| `--decay X` | 0 | weight decay: each step also pulls every weight toward zero by `rate*X*w` |
| `-a NAME` | tanh | hidden activation: `sigmoid`, `tanh` or `relu`; the output stays a sigmoid |
| `--holdout X` | 0.25 | fraction of rows kept out of the fit |
| `--stream` | | fit without holding the rows; see [Memory](#memory) |
| `--buffer N` | 65536 | rows in the shuffle window under `--stream` |
| `--cache FILE` | | keep `--stream`'s scaled rows in FILE, so a later run skips both passes over the CSV |
| `--per-refit FILE` | | write each refit's held-out error, for `pairstat --paired` |
| `--missing P` | refuse | `drop` skips rows with an empty field or `NA`, counts them, and records the count in the model |
| `--id` | | a case carries an opaque id after the group, echoed on the prediction |
| `--unknown-group P` | fail | `skip` emits `NA` for a group the model does not have, so the output row count still matches the input |
| `--footprint T G` | | what a fit of T terms and G groups costs in memory |
| `--selftest` | | check the arithmetic and exit |

`--holdout 0` disables the split. The reported error then says nothing about generalization, and the
report says so. It is there for the case where the relation is noiseless and known.

## Memory

By default every row is held in memory. `--stream` does not hold them: it reads the file twice, once
for the ranges and once to write a cache of the scaled rows, and then reads that cache once per epoch,
plus once more per epoch to see whether the fits have stopped improving. Memory is then the
networks plus the shuffle windows, and neither depends on the number of rows.

Least squares needs one pass because its objective has a fixed-size sufficient statistic.
Backpropagation has none: the gradient depends on the current weights, so every epoch has to see the
rows again. Streaming removes the storage, not the passes.

`scripts/scale.sh` measures it, fitting the same model over ten times the rows (two terms, four
groups):

    $ scripts/scale.sh 100000
                                            100000 rows 1000000 rows
    --stream, peak RSS (kB)                        8832         8832
    default, peak RSS (kB)                         6088        39308

The default path grows with the rows. `--stream` does not. But note the left-hand column: at 100,000
rows `--stream` costs *more*, because it pays for one shuffle window per refit whatever the file
size. Turn it on when the row store stops fitting. `--footprint TERMS GROUPS` prints both figures for
a shape before you run anything:

    $ ./bpnn --footprint 24 400
    24 terms, 400 groups, 6 hidden units, 5 refits

    fitting with --stream
      per group per refit    1.25 kB
      networks in total      2.93 MB
      shuffle windows        33.8 MB
      total                  36.7 MB

    fitting without --stream, add the row store, which does take a row count:
      per row                208 bytes

    The --stream cache is a temporary file of 108 bytes a row, removed on exit.

A row costs `(terms + 1) * 8 + 8` bytes held, so the crossover is around 200,000 rows at two terms
and 170,000 at twenty-four. If the store does not fit, the fit says what it was holding and names the
flag rather than printing `out of memory`; and once the store passes 64 MB the report prints what it
cost, since that is the figure that decides whether the next file will fit.

`--cache FILE` keeps the scaled rows between runs. The two setup passes do not depend on any
hyperparameter, so the cache is valid across different `-H`, `-e`, `-s` and `--holdout`, which is
what makes it worth having while you are choosing them. On a million rows it takes a run from 2.8 s
to 1.9 s at `-e 1`, where the setup is nearly all there is
([`doc/BENCHMARKS.md`](doc/BENCHMARKS.md)). The key is the input's size and modification
time, and it is wrong in the way every make-like tool is wrong: a file rewritten inside the same
second, to the same length, with different contents reuses a stale cache. Detecting that would mean
reading the file, which is the cost the cache exists to avoid.

The two paths give different numbers, and neither is an approximation of the other. The default
shuffles each group's rows completely every epoch. `--stream` shuffles through a window of `--buffer`
rows, so the training order differs. Both repeat byte for byte from the same input, and the default
is the one every number elsewhere in this file comes from.

One case to know about: a file already sorted by the response, with a window smaller than the file.
The window then leaves that order nearly intact and the fit trains on it. That is reported:

    $ ./bpnn -t rows.csv --stream --buffer 4
    bpnn: rows.csv is sorted by y, or nearly so, and the shuffle window holds
    4 of its 40 rows, so each pass trains on close to that order. Shuffle
    the file, or raise --buffer above the row count.

## Four ways the fit can be wrong

linearr checks for collinearity, exhausted degrees of freedom and lost digits. A network fails in
different ways. These four are the ones this program can check.

### The fit memorised the rows

Two coefficients cannot memorise thirteen points. Thirty weights can,
and then the training error measures recall, not accuracy. So the rows are split, and the error on
the fitted rows and on the held-out rows are printed side by side. If the second is much larger, the
model learned this table rather than the relation behind it.

    $ ./bpnn -t curve.csv -H 24
    group        rows   held  weights  epochs      train   held-out   refit sd      floor   expl
    A              40      5       48      28     32.523     13.597     4.5637      12.65   100%
      48 weights fitted to 35 training rows. There are more free
      parameters than examples, so some of what it learned is the rows
      themselves. Reduce -H, or use linearr if the relation may be linear.
      the spread over refits is 34% of the error itself, so a single fit
      of this configuration does not pin down its quality.

### When it stops

`-e` is a ceiling, not a count. The fit checks its error after every epoch on
rows kept back for that purpose, keeps the weights from the best check, and gives up after
`--patience` epochs with no improvement. The `epochs` column reports the epoch whose weights were
kept, not the epoch the run gave up at. On `example/nonlinear.csv` that is 80 and 9 out of 3000,
and the fit takes 0.07 s instead of 1.59 s. The held-out error is no worse for it: 1.135 and
1.2135, against 1.1817 and 1.2141 for the full 3000.

The rows that decide when to stop are not the rows the error is reported on. Choosing a stopping
point by a number makes that number optimistic by however much was selected for, so the held-out
rows are split in half: one half stops the fit, the other is reported and is not looked at during it.
That halves the reported `held` count, which is the cost. Below four rows either way the split means
nothing and the fit runs its full `-e` epochs, which is what happens on the seven-row groups above.
`--patience 0` turns the whole thing off.

### Refitting gives a different answer

Least squares has one solution. A network starts from random
weights and shuffles its examples, so fitting the same rows twice gives two models and two scores.
`-s` sets the number of refits. The spread over them is printed next to the error, and 2.77 times
that spread is the resolution: two configurations closer than that cannot be told apart on this data.
The model written out is the median refit, not the best one. The best is optimistic by however much
was selected for, so both numbers go in the model file.

### A case can fall outside the data

linearr lists this as something a regression cannot check. This
program can check it, because scaling an input needs the range it was trained on, and that range is
stored in the model. Outside the range the units saturate and the prediction goes flat:

    $ ./bpnn -c model.txt 001 dose=25 age=60
    the case: held-out RMSE at fit time 1.13502, spread over refits 0.0460124
    the case: dose=25 is outside the fitted range [0.0118, 9.9804], by 1.51 of it
    the case: a network does not extrapolate; past its range the units saturate
    001,22.7231

There is a second limit under that one, and it can bite while every input is still in range. The
output is a sigmoid, so the prediction cannot leave the response's fitted range by more than about
an eighth of it in either direction. When a case pushes it against that wall the answer is the wall:

    $ ./bpnn -c curve.model A x=400
    the case: saturated: 1792.72 is the most extreme y this model can return (fitted over [11, 1610])
    A,1792.72

### One extreme response value breaks the whole group

The target is mapped onto the output unit by
its smallest and largest value, so a single row far from the rest squeezes every ordinary row into a
sliver of the range the network can resolve. The fit then cannot separate them, and the variance
explained is measured against that same inflated spread, so it reads high while the model is useless.
The distance from the quartiles is measured and reported:

      los REACHES 26089 INTERQUARTILE RANGES past its own quartiles in this
      group. The target is scaled by its smallest and largest value, so
      every ordinary row is squeezed into a sliver of the output's range
      and the fit cannot separate them.

Three it cannot check: a term missing from the table, rows that are not independent of each other,
and a relation that changed after the training rows were collected.

## What it refuses to read

A bad value in the input is not caught later: an empty field read as a zero, or one `nan`, reaches
every weight in the group. They are refused when the file is read, with the line and column named the
way a compiler names them:

    $ awk 'NR==118{sub(/,[^,]*$/,",nan")}1' example/nonlinear.csv > rows.csv
    $ ./bpnn -t rows.csv
    rows.csv:118:4: the term 'age' is 'nan': not finite
    $ echo $?
    1

Also refused: a row whose field count disagrees with the header, an empty field, a header naming the
same term twice, a group code too long to store whole (two such codes would otherwise be merged into
one fit), an unknown option, an option missing its value, and a case that leaves a term unnamed, since
scoring it as zero answers a different question. `tests/cli.sh` has one check per refusal. None of
them can be reached from the unit suite.

## When a network is worth it

`bench/compare.sh` puts both programs on four files whose true relation is written down, so the best
attainable error is known in advance and neither program is graded on the other's weaknesses. It
regenerates the data first, so the table is never quoted from a previous run. About twenty seconds on
a 4-core i7-1165G7; RMSE, lower is better.

| data | best possible | linearr | linearr's verdict | bpnn held-out |
|---|---|---|---|---|
| exactly linear, no noise | 0 | **0** | no complaint | 4.18 |
| y = x² + 10, no noise | 0 | 13.49 | wrong shape, term named | **0.98** |
| saturating dose + interaction | 1.0 | 1.86 | wrong shape, term named | 1.21 |
| y = x₁·x₂ | 1.0 | 8.62 | wrong shape, pair *unnamed* | 1.46 |

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

## The length-of-stay shape

`bench/elos.sh` puts both programs on a table shaped like the 2011 length-of-stay model: 24 binary
procedure indicators, 8 groups, 600 rows each, per-group intercepts. Three files, same shape,
different truth underneath, noise floor 1.0 by construction:

    truth            floor    linearr     its verdict         bpnn
    linear             1.0      1.034          SILENT        1.594
    saturating         1.0      2.574    pair unnamed        1.169
    interaction        1.0      1.751          SILENT        1.654

    interaction, with the pair added as a column: 1.073

Two rows go the way the pairing is meant to. On the linear truth the line reaches the floor and
the network does not. On the saturating truth the line is badly wrong, its own check says so, and
the network nearly reaches the floor: escalation working.

**The interaction row is the one to take seriously, and it does not flatter this program.** The
line is wrong, the network barely improves on it at any capacity between 8 and 24 hidden units,
and linearr's residual check says nothing at all, so `scripts/escalate.sh` would report that a
line was enough. Handed the pair as one extra column, the line reaches the floor immediately.

The reason is arithmetic. With 24 indicators at 18% prevalence each, about 3% of rows carry both
procedures, so the effect lives in a few dozen rows per group. That is too few for a network to
find and too subtle for a residual check to see. On tables of this shape, neither program will
tell you which pair interacts, and a clinician naming it is worth more than either.

## The example data

Everything in `example/` is synthetic, generated from a fixed seed, and regenerates byte for byte.
`example/mkelos.c` writes the three length-of-stay files above and states each one's noise floor in
its own header; `example/mknonlinear.c` writes the two smaller ones. `nonlinear.csv` is a saturating dose response plus an interaction,
where linearr's check names the term; `interaction.csv` is `y = x1*x2`, where it can only say that
something interacts. The linear cases in the table above are linearr's own example files, read from
`~/linearr/example`. No real data is distributed with this project.

## Build and test

    make            build ./bpnn and ./bpnn_worker
    make check      ut + cliut: what must pass before a commit
    make ut         32 unit checks: rng, act, net, xor, arena, data, conv1d, conv2f
    make cliut      199 black-box checks: the built binary, through a shell
    make ut-asan    both suites under AddressSanitizer
    make ut-ubsan   both suites under UndefinedBehaviorSanitizer
    make pedantic   -pedantic with -Wextra -Wshadow -Wconversion; must be clean
    make tools      the measurement tools; runs the two self-tests
    make install    /usr/local, or PREFIX=$HOME/.local
    make clean

A warning is a defect. `make check` gates every commit, and both sanitizers run before one. The
sanitizer targets build the binary as well as the suite, since the CSV reader is only reached through
the binary. `./bpnn --selftest` is a smaller check the binary carries itself: input scaling at both
ends, the target round-trip, the resolution coefficient, and a parabola fitted to a known error.

## What this is, and what it is not

This was not written to compete with a framework. It was written with
[linearr](https://github.com/Anode1/linearr) in mind, for the case that program runs out of: the
dependency is not linear, and you still want an answer you can defend. Everything about it follows
from that — one topology, one optimiser, per-group fitting, a CSV in and a case out, and a report
whose job is to tell you when not to believe it.

**The standard tools are better at almost everything.** Keras and TensorFlow train any architecture
you can draw, on GPUs, with autodiff; TFLite and LiteRT deploy those models to phones and
microcontrollers with quantisation and hardware delegates; and for tabular data specifically,
gradient boosting — LightGBM, XGBoost, CatBoost — usually beats a small network outright. If your
question is "what is the most accurate model", none of the answers is this program. Reach for
something else when you need class probabilities, more than one output, embeddings for
high-cardinality codes, dropout, batch normalisation, an optimiser other than momentum, a GPU, or
anything that is not a table. None of it is here and none of it is planned.

**The band this is for is narrow, and it is real.** A tabular relation that a line gets wrong,
where nobody can say which column to add; the machine has no Python and is not going to get one; the
answer must reproduce byte for byte years later, from an archived model, on another compiler; the
data per group is small enough that a framework's defaults will overfit it silently; and somebody
will have to defend the number rather than a leaderboard position.

**It is also an instrument, not only a predictor.** Most of the code is not the network — it is the
part that measures whether the fit is worth anything: the spread over refits and the resolution that
implies, the variance explained on rows the fit never saw, whether a case is outside the range each
term was fitted over, whether the output has saturated, whether one extreme response has wrecked the
scaling. `validation/pairstat.c` and `validation/resolve.c` answer the same question for any pair of
numbers, not only this program's. A framework will fit whatever you hand it and report a loss
without comment; this one is built to say what it cannot tell you.

Two hard limits. The engine computes in `smb_real`, which is `float`, so about seven digits. linearr
validates against NIST reference values to eleven; nothing here should be trusted past six. And the
build ceilings are 64 terms and 512 groups, both fixed at compile time.

## The measurement tools

Both are self-contained `.c` files with no dependency on the engine, and each carries a self-test
that `make tools` runs.

`validation/pairstat.c` compares two configurations. Paired statistics with named tests: Wilcoxon
signed-rank as primary, exact by dynamic programming where ties permit and tie-corrected normal
otherwise, always reporting which it used; an exact sign test; Hodges-Lehmann estimates with
distribution-free intervals; a paired *t* as secondary; Holm correction across a declared family;
and a minimum detectable effect, so a null result is bounded rather than asserted.

`bpnn --per-refit FILE` writes each refit's held-out error, and `pairstat --paired A B` compares
two such files on matched refits. This is the comparison to make; the printed `floor` is the rough
version of it. Both runs draw their split and their starting weights from the refit index alone, so
the noise they share cancels, and the comparison is several times sharper than the unpaired scale
the report prints. `pairstat` refuses two files whose pairing stamps disagree, because a paired test
on unpaired runs claims a precision that is not there.

`validation/resolve.c` answers three questions from repeated scores, and refuses to guess when given
one run per candidate: is candidate 0 really better than the others, how many runs per candidate
would resolve a difference of a given size, and what is the best rank correlation any predictor
could score against labels this noisy.

## Origin

Forked from [SMBPANN](https://github.com/Anode1/SMBPANN). Nothing here changes that engine or
its arithmetic; a submitted paper rests on its numbers and they stay reproducible.

## Documents

| document | what is in it |
| --- | --- |
| `c/bpnn.c` | the header comment is the specification: the input format, the failure modes and what each check computes |
| `AGENTS.md` | the operating manual, and the specific mistakes this line has already paid for |
| `doc/dev/STYLE.md` | the C ideology: one concept per file, stack-first, allocation at construction, single-exit cleanup |
| `doc/dev/PROSE.md` | how these documents are written |
| [`doc/DIAGNOSTICS.md`](doc/DIAGNOSTICS.md) | what every reported number is, how far it can be trusted, and where it is known to be wrong |
| [`doc/BENCHMARKS.md`](doc/BENCHMARKS.md) | every rate and footprint, the machine they came from, and the script that reproduces each |
| `bench/compare.sh` | the comparison above, and why each of the four files is in it |

## Platforms

Built and checked on Linux and macOS on every push (`.github/workflows/checks.yml`): the unit
suite, the black-box suite, `pedantic` over every source, and both sanitizers. A second job builds
under gcc 12, gcc 13 and clang at two optimisation levels, and a third fits the same file with all
four and compares the model files byte for byte. That last job is the measurement behind the
reproducibility claim above.

**Only Linux has been used by hand.** macOS passes its checks in CI and nobody has driven it.
Windows is not covered at all: `bpnn_worker` uses `getopt`, and `--stream` without `--cache` uses
`tmpfile()`, which on Windows wants the drive root.

## See also

- [linearr](https://github.com/Anode1/linearr): the straight line, in closed form, which is the program
  to run first.
- [ais](https://github.com/Anode1/ais): the associative-memory engine these conventions come from.

## License

BSD 2-Clause; see `LICENSE`.
