# What each reported number is, and where it is wrong

The report is the reason to use this program rather than a fifteen-line network. Every column in
it is an estimate, and three of them are estimates of things that are hard to estimate. This file
says what each one computes, and what it does not support.

An outside statistician was asked to attack these numbers. Most of what follows is what they
found; the parts that were fixed are described as fixed, and the parts that were not are
described as limits.

    group        rows   held  weights  epochs      train   held-out   refit sd      floor   expl
    001           400     50       18      80     1.0731     1.1259   0.046476    0.12882    92%

## rows, held

`rows` is every row of the group. `held` is the rows the reported error is measured on, and it is
smaller than you might expect for a reason worth knowing.

The rows are split three ways, not two: the fit trains on 75% by default, and the remaining 25%
is halved. One half decides when to stop training; the other is `held`, and is what `held-out`,
`refit sd`, `floor` and `expl` are computed from. The stopping half is not reported on, because
choosing a stopping point by a number makes that number optimistic by however much was selected
for. `--holdout` sets the 25%; `--patience 0` turns off the stopping check and gives all of it
back to `held`.

Below `--min-rows` (24) a group is skipped. At four rows the reported error was one row and the
floor printed beside it was ten times the error it bounded.

## epochs

The epoch whose weights were kept, not the epoch the run gave up at. Those differ by `--patience`
and only the first describes the model that shipped. The stopping check runs every epoch.

## train, held-out

RMSE in the response's own units, over the fitted rows and over the reported rows.

**What `held-out` supports:** an estimate of this configuration's error on rows of this kind.

**What it does not:** it is the error of the *shipped* refit, and the shipped refit is the one
whose error is the median of the refits' errors. So the reported number is also the statistic
that chose which model to ship. The median is unbiased for the median of the estimates, which is
not the same as unbiased for the shipped model's generalization error, and RMSE over 50 rows is
right-skewed, so the reported figure sits a little below what a fresh fit of the same
configuration would give. The clean fix is a fourth partition (train, stop, select, report) and
it is not implemented.

**A known leak:** the input ranges and the target's mapping onto the output band are computed
from every row of the group, including the reported ones. Minimum and maximum are exactly the
statistics one extreme row moves, so if a reported row is the group's extremum, the scaling was
calibrated to it. That also means the range in the `OUTSIDE THE TRAINING RANGE` message is the
group's range and not strictly the training range. Fixing it means per-refit ranges, which
changes what a model file contains.

## refit sd

The sample standard deviation of `held-out` across `-s` refits.

**What it contains:** the randomness of the starting weights, of the shuffling order, and of the
split, since each refit draws a fresh one.

**What that means:** it is *not* purely the optimiser's instability, though the README's framing
of "refitting gives a different answer" invites that reading. A part of it is the sampling noise
of scoring on a different 50 rows each time, and that part cancels in any comparison run on a
common test set.

**How well it is known:** from `-s 5` it has 4 degrees of freedom, and a sample SD on 4 df has a
95% interval of roughly [0.60·s, 2.87·s]. At `-s 2` that interval is [0.45·s, 31.9·s]. The number
is printed to five significant figures and is worth about one.

## floor

`2.7718 × refit sd`, where 2.7718 is `1.96 × √2`.

**Read it as a rough scale, not as a test.** Four things are wrong with taking it literally:

1. It is a critical value, not a minimum detectable effect. A true difference exactly equal to
   the floor is found about half the time. An 80%-power figure is about 43% larger.
2. It uses the normal critical value for a spread estimated on 4 degrees of freedom. The `t`
   value there is 2.78, which is nearly the same number by coincidence, not by construction.
3. It describes a difference between two *single* fits. The number printed next to it is a
   median over refits, whose spread is smaller.
4. At `-s 1` it is zero, and zero is printed with no warning.

It also assumes the two configurations are independent when they are not: the split seed and the
init seed depend only on the refit index, so two configurations get the same splits and the same
starting weights. A paired comparison is available and is far more sensitive: measured on this
tree, a paired 95% MDE of 0.026 where the printed floor was 0.248.

**Do the paired comparison instead.** Both runs draw their split and their starting weights from
the refit index alone, so refit 3 of one configuration and refit 3 of another saw the same rows
and the same initial weights. Comparing them pairwise cancels the noise they share:

    $ ./bpnn -t data.csv -H 4 --per-refit h4.refits > /dev/null
    $ ./bpnn -t data.csv -H 8 --per-refit h8.refits > /dev/null
    $ ./pairstat --paired h4.refits h8.refits

`pairstat` gives a Wilcoxon signed-rank test, a Hodges-Lehmann estimate with a distribution-free
interval, a paired *t*, Holm correction across the groups, and a minimum detectable effect. On
`example/nonlinear.csv` at five refits that MDE is 0.126 where the printed floor is 0.25, and it
sharpens further with more refits: an outside reviewer measured 0.026 at twelve.

The pairing holds only while both runs used the same `-s`, `--holdout` and `--patience`, so the
per-refit file stamps those and `pairstat` refuses two files that disagree. A paired test on
unpaired runs claims a precision that is not there, which is worse than the conservative floor.

`validation/resolve.c` answers the other question: how many runs a difference of a given size
would need, and it refuses to answer when given one run per candidate.

## expl

`1 − MSE/Var(y)`, both over the reported rows, with the reported rows' own mean as the baseline.
Zero means the group's own mean would have done as well; it goes negative when the fit is worse
than that mean, and the report says so below 5%.

**Where it misleads:** if the response has one extreme value, the variance it is measured against
is dominated by that value and the ratio reads high while the model is useless. That condition is
detected separately and reported as `REACHES n INTERQUARTILE RANGES`, because the ratio cannot be
trusted when it holds.

It was computed against the variance of *all* the group's rows until an outside review showed it
reporting 100% for a fit that did worse than the mean of the rows it was scored on.

## The advisories

Printed under a group's row when they apply.

| what fires it | what it means |
|---|---|
| `weights fitted to N training rows` | more free parameters than examples; some of what it learned is the rows |
| `held-out error is Nx the training error` | fitting the rows rather than the relation |
| `the spread over refits is N% of the error` | one fit of this configuration does not pin down its quality |
| `explains N% of the variance` | the group's own mean would have done as well |
| `REACHES n INTERQUARTILE RANGES` | one extreme response value has compressed every other row |
| `NEVER VARIES` | the response is constant; the errors are near zero because there was nothing to predict |
| `NO HELD-OUT ROWS` | `--holdout 0`; the held-out column is the training error |
| `DIVERGED` | the weights left the range of a number |

At scoring time, on standard error:

| what fires it | what it means |
|---|---|
| `outside the fitted range` | this term's value is beyond anything the fit saw |
| `saturated` | the output unit is at its limit; the answer is the limit, not a prediction |

## What none of it can see

A term that should have been in the table and is not. Rows that are not independent of each
other. A relation that changed after the training rows were collected. These are the three the
program does not claim to check, and no amount of held-out data reveals them.
