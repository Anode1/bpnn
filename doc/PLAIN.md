# The essence, in plain words

Written to be read start to finish without the paper. No statistics background assumed.

## The one-paragraph version

Training a network is random, so training the same design twice gives two different scores. Any single
score is therefore the design's real quality plus a bit of luck. Everything in this project follows from
separating those two things and asking what each is worth. It turns out that almost everything a
practitioner wants to know — is my improvement real, how many runs do I need, can my predictor possibly
be as good as it looks, have I already found the best design — is determined by the *ratio* of the luck
to the real spread of quality across designs. Two numbers, and the rest is arithmetic.

## The two numbers

Imagine a big pile of network designs. Two different kinds of variation exist in that pile.

**Between designs.** Some designs really are better than others. Call the spread of true quality across
the pile **σ_B** (B for between). On a good CIFAR-10 benchmark the good designs sit within a percentage
point or two of each other, so σ_B at the top of the pile is small.

**Within a design.** Train one design repeatedly and its score wobbles. Call that spread **σ_W** (W for
within). On NAS-Bench-101 at full training the typical design wobbles by about 0.33 percentage points.

The awkward fact is that **a single measurement cannot separate them.** One score contains both, and
nothing you do with that one number will tell you how much of it was quality and how much was luck. This
is why repeated runs are not a nicety; they are the only way to see the two apart. And it is why the two
benchmarks we use are valuable out of all proportion to their age: their authors trained everything three
times, so the separation is available.

## Why the ratio is what matters

Suppose you score a design once, and separately score it again. Both scores contain the same real
quality and different luck. So the two scores agree with each other only to the extent that quality
dominates luck. Write **r = σ_W / σ_B**. When r is small, luck is a rounding error and the two scores
rank designs almost identically. When r is around 1, luck is as large as the real differences, and the
two scores agree barely better than chance.

That is the whole mechanism, and it has a formula. The correlation between two independent noisy views
of the same thing is

    rho = 1 / sqrt( (1 + r²) (1 + r²/2) )

for a one-run score against a two-run average. Everything else in the project is this quantity wearing
different clothes.

**This is not new mathematics and it is important to say so.** It is the classical relationship between
measurement error and observed correlation, worked out in psychometrics over a century ago and known
there as reliability and attenuation. The value here is not the formula; it is that nobody had measured
the inputs for a machine-learning benchmark, so nobody knew what the numbers were.

## Why it gets much worse when you look at good designs only

This is the part that surprised us, and it is the most useful single insight.

Take the top thousand designs instead of the whole pile. The designs are now all good, so the real
differences between them are *tiny*: σ_B has shrunk enormously. But the training wobble σ_W has not
changed at all, because how much a design's score wobbles has nothing to do with which designs you chose
to look at. So r shoots up, and everything degrades.

Measured on NAS-Bench-101: over the whole pile, one run agrees with an independent reference well
(correlation 0.83). Among the top thousand, that falls to **0.084**. Nearly nothing.

And the top of the pile is exactly where all the interesting work happens. Every search, every
predictor, every comparison anyone cares about operates there.

## What this is good for, concretely

### 1. Reallocating compute you are currently wasting

If you evaluate candidates cheaply (few epochs, small subset, one seed) and rank them, you are only
sorting real differences to the extent that r is small at that fidelity. We measured this directly. On
NAS-Bench-101 at 4 epochs a difference of one to two full percentage points is called backwards 45% of
the time; at 108 epochs the same-sized difference is wrong 4% of the time. **At low fidelity the ranking
carries essentially no information at any effect size.**

The practical consequence is a reallocation, not a lament: if your ranking signal is that noisy, more
candidates does not help you, because you are sorting noise faster. Fewer candidates trained longer, or
the same candidates evaluated twice, converts the same compute into an actual ordering. And the
crossover point is computable from two variances rather than guessed.

### 2. Knowing when you have already finished

We counted how many designs a benchmark cannot distinguish from its own reported best. On
NAS-Bench-101 it is 3,558 of 423,624. So "we found the optimum" there means "we landed somewhere inside
a set of 3,558 that the data cannot order", and a search that keeps going is choosing among
indistinguishables. Any curve plotted as distance-from-optimum inherits that width.

For your own work the same count tells you when to stop: once your candidates are inside each other's
error bars, further search is not improving the model, it is relabelling the winner.

### 3. Interpreting your own numbers, and other people's

A large literature builds cheap predictors that guess a design's quality without training it, and scores
itself by rank correlation against the benchmark's numbers. But the benchmark's numbers are themselves
noisy, so there is a **ceiling** on that score: the correlation the benchmark achieves with itself. On
NAS-Bench-101's top thousand that ceiling is 0.084.

This changes how a number is read, in both directions. A predictor scoring 0.7 where the ceiling is 0.95
is mediocre. A predictor scoring 0.7 where the ceiling is 0.72 is nearly perfect and should be
celebrated. And a predictor reported at 0.3 in a regime where the ceiling is 0.084 has not beaten the
noise; something else is going on, usually a differently-defined subset or a leak.

Nobody reports the ceiling, which means absolute correlations in that literature currently cannot be
interpreted at all. That is the finding with the widest audience.

### 4. A one-column diagnostic for your own pipeline

Whenever you pick the best of several candidates using a held-out score and then report that score, part
of what you report is the luck you selected on. To measure how much: hold out *two* samples instead of
one, choose using only the first, and report the gap. For a candidate chosen without looking at either,
that gap is zero on average, so anything above zero is self-deception, denominated in the same units as
your headline. It costs one extra column.

### 5. Detecting unstable configurations from aggregate statistics

This one falls out of the theory failing. If the measured self-correlation is *much higher* than the
formula predicts from your variances, it means your training noise is not one distribution but a mixture:
a minority of configurations are wildly unstable and are inflating the average wobble without affecting
how the well-behaved majority rank. We saw exactly this on NAS-Bench-101, where the prediction was off by
0.41 until we removed the 1.4% of designs that sometimes train and sometimes collapse to chance. After
that the prediction was off by 0.02.

So the size of that discrepancy is a detector: it says "some of your runs are unstable, go find them"
without your having to inspect every run.

## What we do not claim

The phenomenon is common knowledge in outline; the numbers were not known. The mathematics is a century
old; its inputs for these benchmarks were not measured. And none of this indicts work that used a
benchmark's published averaged numbers in the ordinary way — it bears on methods that train designs
themselves, on predictor evaluation, on cheap-proxy search, and on the benchmarks' own claimed optima.

## The honest state of it

The measurements are done and reproducible from the code in `validation/`. The theory fits well where
noise is below about sixty per cent of the quality spread and degrades beyond it, always over-predicting,
which is consistent with the mixture effect above and is not yet fully explained. The largest weakness is
that one of the two benchmarks reaches us through a conversion that fills in a few missing runs with the
average of the others, which deflates exactly the quantity we are measuring, by an amount we have not
been able to bound. `doc/FINDINGS.md` lists the rest.
