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

**What `held-out` is:** the mean over `-s` refits. It estimates the expected error of *refitting
this configuration* on data of this kind, averaged over that many splits of this table. Nothing
selected on it, which is why it replaced the shipped refit's error.

**What it is not:** the shipped model's own error. The shipped model is the lower-median refit,
so its estimand is the median of `-s` draws, not their mean. Measured by resampling from `-s 60`
runs, the gap runs from −0.5% to +4.7% depending on the skew of the refit distribution, usually
with the mean the higher of the two, so the column is conservative for the shipped model rather
than unbiased for it. The shipped model's own figure is `shipped=` in the model file, and it *is*
a selected statistic.

**It is conditional on this table.** Five splits of one 400-row group share about 87.5% of their
training rows, so the spread across refits says nothing about a fresh 400 rows. Six independent
draws from one generating process gave Hodges-Lehmann differences of 0.33 to 0.68 between the same
two configurations, a scatter as large as any one run's confidence interval.

**And it is not a fair estimate of the configuration you chose.** If you compare several `-H`
values and keep the winner, the winner's held-out error is optimistic by the usual winner's-curse
amount. Report the paired *difference* instead, which is unbiased, or refit the chosen shape
against rows that took no part in the comparison.

**Scaling.** Each refit's input ranges and target mapping come from its own training rows, so the
reported rows take no part in the scaling and the range stored in the model is the range the
shipped refit trained over. `--stream` is the exception: its cache is scaled once and shared
across refits, so on that path the scaling still sees every row and its reported error is very
slightly optimistic for that reason.

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
interval, a paired *t*, Holm correction across the groups, and a minimum detectable effect. On `example/nonlinear.csv` at five refits the paired MDE measures 0.124 and 0.139 for the two
groups, against printed floors of 0.112 and 0.443 -- sharper on one group and not on the other, so
this is not the uniform win an earlier draft of this file claimed. At twelve refits it measures
0.054 and 0.047.

Two cautions on those figures. `pairstat` treats the refits as independent when they are five
splits of one table; the Nadeau-Bengio correction for that overlap widens the interval by about a
third at five refits and three quarters at twelve, and no amount of refitting drives it to zero.
And the interval is about *this table*, not about the configuration in general.

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
reporting 100% for a fit that did worse than the mean of the rows it was scored on. The
denominator comes from one refit's reported rows, so it moves by a few points depending on which;
a pooled sum over all refits would be steadier and is not implemented.

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
