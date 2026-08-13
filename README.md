# bpnn: a neural network in C, for the data a straight line gets wrong

### It reports three of the ways a network misleads you, and they are not the three a regression has

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

and it stops there, exit 0, having fitted no network. Give it data a line gets wrong and it escalates,
exit 1, with both models on disk. The order is the point: the line is tried first because when it is
right it is better in every way that matters, and the escalation is triggered by a measurement rather
than by taste. "We used a neural network" is not a modelling decision.

One weakness of that script is worth knowing before you rely on it, and it is written into the source
where the trigger is: linearr exits 0 whether or not the shape is wrong, so the escalation is a text
match on the warning. If that wording ever changes, the script stops escalating and reports that a
line was enough, which is the wrong answer given quietly. It therefore refuses to conclude anything
when linearr printed no `fit:` summary at all. The fix belongs upstream, in a shape column or a
distinct exit status.

## What backpropagation is

**What it does.** A network of weighted sums with a squashing function between them, fitted by
pushing the error at the output backwards through the layers and moving every weight against its own
share of it. The method is Rumelhart, Hinton and Williams, 1986, and it is what fits a shape nobody
wrote down in advance. The arithmetic here is a 1997 seminar thesis' tensor derivation of it, and
`c/net.c` and `c/train.c` cite that thesis by section.

**What `c/train.c` is doing, basically.** With `a` the activations of a layer, `g'` the derivative of
the squashing function and `d` the target, the whole of the generalized delta rule is four lines:

    beta[L] = (d - a[L]) * g'(a[L])                     # output layer: what the answer missed by
    beta[l] = g'(a[l]) * W[l+1]' beta[l+1]              # the same error, pushed back a layer
    dw[l]   = rate * beta[l] a[l-1]' + momentum * dw[l] # this step, plus a share of the last one
    W[l]   += dw[l]

Four things turn them into the 139 lines of code in `c/train.c`. Layers are flat arrays rather than a
matrix library, so the products are written out. A convolutional layer shares one filter across every
position, so its gradient is the sum over the positions it visited rather than one term. Every scratch
buffer is allocated once in `trainer_new`, so the training loop allocates nothing. And the derivative
is not assumed: `make ut` checks it against finite differences, for the dense and both convolutional
paths, which is the check that catches a sharing bug the loss curve would hide.

## Fit and score

Two commands, and the same split as linearr: standard output is the model, standard error is the
commentary, so the redirect is the whole workflow.

    $ ./bpnn -t example/nonlinear.csv > model.txt
    group        rows   held  weights      train   held-out   refit sd      floor
    001           400    100       18     1.1706       1.15   0.097926    0.27143
    002           400    100       18     1.1877     1.2551    0.14005     0.3882

    $ ./bpnn -c model.txt 001 dose=5 age=60
    001 los = 16.2301
    held-out RMSE at fit time 1.14997, spread over refits 0.0979262

`-t` fits every group in one pass; `-c` names the model to score against and is required, because
which model produced a number is part of the number. **Column 2 is the value being predicted**, and
columns 3 onward are the terms, exactly as in linearr. Nothing in the data can say which column you
meant.

The four numbers on the right are the report, and they are the reason to prefer this over a fifteen-line
network somebody pastes into a script. `train` is the error on the rows the fit saw and `held-out` is
the error on rows it did not; `refit sd` is the spread of the held-out figure over repeated fits of the
same configuration; and `floor` is 2.77 times that spread, which is the smallest difference between two
configurations this pipeline can resolve at all. Two setups closer than the floor are not distinguished
by this data, whichever way the comparison came out.

Scoring repeats the fit-time error and the spread on every prediction, because a prediction printed to
four decimals from a model whose own error is 1.15 invites more confidence than it has earned.

## Groups

**A group is one fitted network.** Rows sharing a group code are fitted together and get their own
network, so one file and one pass produce one model per subset. This is linearr's convention and it
matters more here: a network fitted across pooled groups will happily learn the group's identity from
whatever term correlates with it, and say nothing about having done so.

## Options

| option | default | meaning |
| --- | --- | --- |
| `-t FILE` | | fit: one network per group, model to stdout, report to stderr |
| `-c FILE` | *(required to score)* | the fitted model to score a case against |
| `-H N` | 6 | hidden units |
| `-e N` | 3000 | epochs |
| `-s N` | 5 | refits, which is what the spread and the floor are measured over |
| `-r X` | 0.3 | learning rate |
| `-m X` | 0.9 | momentum |
| `-a NAME` | tanh | hidden activation: `sigmoid`, `tanh` or `relu`; the output stays a sigmoid |
| `--holdout X` | 0.25 | fraction of rows kept out of the fit |
| `--selftest` | | check the arithmetic and exit |

`--holdout 0` disables the split, which makes the reported error meaningless as a statement about
generalization. It exists for the case where the relation is noiseless and known, and the option says
so rather than quietly reporting a training error in a column headed held-out.

## The three ways it misleads you

linearr reports collinearity, exhausted degrees of freedom and lost digits. A network fails
differently, and these three the program can see from what it holds.

**It memorises.** A line with two coefficients cannot memorise thirteen points; a network with thirty
weights can, and its training error then measures recall rather than knowledge. So the rows are split
and both errors are printed next to each other, with the arithmetic stated when it goes badly wrong:

    $ ./bpnn -t ~/linearr/example/simple-train.csv > /dev/null
    group        rows   held  weights      train   held-out   refit sd      floor
    A               7      2       18 1.4796e-05     5.2607     1.5361     4.2577
      18 weights fitted to 5 training rows. There are more free
      parameters than examples, so some of what it learned is the rows
      themselves. Reduce -H, or use linearr if the relation may be linear.
      held-out error is 355556.7x the training error: this network is fitting
      the rows rather than the relation. Fewer hidden units, or more rows.

**Retraining changes the answer.** Least squares has one solution and returns it every time. A network
starts from random weights and shuffles its examples, so the same rows fitted twice give two models
and two scores. That spread is not a nuisance to be averaged away, it is the resolution of every
comparison made with this tool, which is why the fit is repeated over `-s` seeds and the spread is
printed beside the error. The model written out is the **median** of those seeds and not the best of
them: picking the best by held-out error makes that error optimistic by however much was selected for,
which is the oldest mistake in model selection. Both numbers are in the model file so the gap is
visible.

**It does not extrapolate, and will not say so unless asked.** This is the one linearr names as a way
regression misleads that it cannot check. Here it can be, because the network has to store the range
of every input in order to scale it:

    $ ./bpnn -c model.txt 001 dose=25 age=60
    001 los = 22.9534
    held-out RMSE at fit time 1.14997, spread over refits 0.0979262
    OUTSIDE THE TRAINING RANGE: dose=25, trained on [0.0118, 9.9804], out by 1.51 of that span
    A network does not extrapolate. Past the range above its units saturate and it
    returns a flat value with no warning of its own, so treat this number as a guess.

Three it cannot see, and does not claim to: a term that should have been in the table and is not, rows
that are not independent of each other, and a response whose relation to the inputs changed after the
training rows were collected.

## When a network is worth it, and when a line with one more column is worth more

`bench/compare.sh` puts both programs on four files whose true relation is written down, so the best
attainable error is known in advance and neither program is graded on the other's weaknesses. It
regenerates the data first, so the table is never quoted from a previous run. About twenty seconds on
a 4-core i7-1165G7; RMSE, lower is better.

| data | best possible | linearr | linearr's verdict | bpnn held-out |
|---|---|---|---|---|
| exactly linear, no noise | 0 | **0** | no complaint | 4.51 |
| y = x² + 10, no noise | 0 | 13.49 | wrong shape, term named | **0.96** |
| saturating dose + interaction | 1.0 | 1.86 | wrong shape, term named | 1.26 |
| y = x₁·x₂ | 1.0 | 8.62 | wrong shape, pair *unnamed* | 1.44 |

And then the option a network is usually not measured against: the same line, with the one extra column
its own diagnostic asked for. Still closed form, still exact, still coefficients you can read.

| line, plus the extra column | best possible | resid SD |
|---|---|---|
| + dose² (the check named dose) | 1.0 | 1.20 |
| + x₁·x₂ (the check could not name the pair) | 1.0 | **1.02** |

So the network never won this table. A correctly specified line reaches the noise floor, gets there
with no seed and no spread, and hands you coefficients you can give to somebody. What a network buys
is not accuracy but not having to guess which column to add, and that guess is cheap exactly when the
diagnostic names the term. It stops being cheap when the check can only say that some pair interacts,
because the candidates then number p(p−1)/2: one pair at two terms, 276 at twenty-four. That gap is
the case for running this program, and it is narrower than the usual framing suggests.

## The example data

Everything in `example/` is synthetic and generated by `example/mknonlinear.c` from a fixed seed, so
the files regenerate byte for byte. `nonlinear.csv` is a saturating dose response plus an interaction,
where linearr's check names the term; `interaction.csv` is `y = x1*x2`, where it can only say that
something interacts. The linear cases in the table above are linearr's own example files, read from
`~/linearr/example`. No real data is distributed with this project.

## Build and test

    make            build ./bpnn and ./bpnn_worker
    make ut         32 unit checks: rng, act, net, xor, arena, data, conv1d, conv2f -- the commit gate
    make ut-asan    the same under AddressSanitizer
    make ut-ubsan   the same under UndefinedBehaviorSanitizer
    make pedantic   -pedantic with -Wextra -Wshadow -Wconversion; must be clean
    make tools      the measurement tools; runs the two self-tests
    make clean

A warning is a defect. `make ut` gates every commit and both sanitizers run before one. `./bpnn
--selftest` is the smaller check the binary carries itself: input scaling at both ends, the target
round-trip, the resolution coefficient, and a parabola fitted to a known error.

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

**Two limits worth stating plainly.** The engine computes in `smb_real`, which is `float`, so about
seven digits; linearr validates against NIST reference values to eleven, and nothing here should be
trusted past six. And unlike linearr, this program holds every row in memory: the fit needs many
passes over the data, so the O(1)-in-rows property of its sibling is not available and is not claimed.
The built-in ceilings are 64 terms and 512 groups.

## The measurement tools

These came out of a closed research direction (below) and are the part of it that survived. They are
independent of the engine: one self-contained `.c` each, no dependencies.

**`validation/resolve.c`** answers three questions from repeated scores, and refuses to guess when
given one run per candidate: is candidate 0 really better than the others, how many runs per candidate
would resolve a difference of a given size, and what is the best rank correlation any predictor could
score against labels this noisy.

**`validation/pairstat.c`** is the piece most worth borrowing. Paired statistics with named tests:
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
