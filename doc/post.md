# The Good Day Problem

> **RETRACTED, 2026-08-12.** This document belongs to a closed direction. Several of its headline
> numbers are wrong (the 0.084 ceiling should be 0.293) or inflated (the 3,558 indifference class
> should be 26), and its central claims turned out to be prior art. Read `doc/CLOSED.md` first: it
> lists every claim and what became of it. Kept for the record, not for citation.

*Every AI benchmark number is part skill and part luck. Somebody had to measure which part, so we did,
and the answer is worse than we expected in the place where it matters most.*

Give a hundred students one exam and crown the top scorer. That student is genuinely able. They also
probably had a good day. Set a second exam and the score drops, not because they got worse, but because
part of what you selected for was the good day, and good days do not repeat on command. A student picked
at random shows no such drop. The difference between the chosen and the unchosen is the whole of what
follows.

Machine learning does this constantly. Train a network design, score it on data held back from training,
keep the best one. The scores are honest, because the data really was held back. The choice is not,
because training is random. The starting weights are drawn at random, the examples are shuffled, and on a
GPU the arithmetic does not even add up in the same order twice. Train the same design twice and you get
two different numbers. So every score is real quality plus a draw of luck, and picking the winner keeps
the luck along with the quality.

Everyone in the field knows this in outline. Almost nobody knows the size of it, and for a good reason:
measuring it means training the same thing over and over, which is exactly the expense the whole
enterprise is organised to avoid.

## Somebody already paid for the experiment

Around 2019 two research groups built catalogues of neural network designs, to give architecture-search
researchers a fixed target instead of everyone burning their own compute. One catalogue holds 423,624
designs, the other 15,625. Both trained **every single design three separate times**.

They did that to be careful, not to study noise. And then both published their tables with the three runs
averaged into one number per design, which is the sensible thing to do if you want to rank designs and
which quietly discards the only evidence in the dataset about how much the numbers move. Almost every
paper since has used the averaged tables.

So the first thing we did was dig the individual runs back out of the original archives. That is
unglamorous work: one of them ships as a 2 GB binary log, the other only through a file format that
normally needs a specific Python library to open. But it means the question becomes answerable with a
laptop and no training at all, because somebody else already did the training.

Then the useful part: with three runs per design you can play one against the others. Use one run to make
a judgement, use the other two as a reference, and the two are statistically independent, so nothing is
being compared against itself.

## The numbers

**When two designs look 0.1 to 0.2 percentage points apart on a single training run each, that comparison
is backwards about 40% of the time.** At 0.2 to 0.3 points it is around 35%. It only becomes reliable once
the apparent gap exceeds a full point. This replicates across both catalogues and all three datasets, and
an independent recomputation with forty million comparisons landed within a third of a percentage point of
ours.

**If you train briefly to save money, it collapses entirely.** Training networks partway and ranking them
on that is standard practice, because full training is expensive; the assumption is that a rough ranking
early transfers to the real ranking later. At four epochs instead of the full hundred and eight, a gap of
one to two full percentage points — a gap that is reliable at 4% error after full training — is called
backwards **45% of the time**. Not degraded. Gone. At that fidelity there is essentially no ordering to
transfer.

**A famous "best design" is not a design.** One catalogue's reported optimum cannot be distinguished from
**3,558 other designs** using the catalogue's own three runs. So "we found the optimum" means "we landed
somewhere inside a set of three and a half thousand that the data cannot order", and every graph plotted
as distance-from-the-best inherits that width.

## The part that surprised us

Here is the counterintuitive bit, and it is the one worth carrying away.

Measured across the whole catalogue, a single training run ranks designs *well*. The correlation with an
independent reference is 0.83. You would look at that and conclude the noise is a minor nuisance.

Now look only at the top thousand designs. The correlation falls to **0.084.**

The reason is almost trivial once you see it. Restricting to good designs means they are all similar to
each other, so the real differences between them shrink to almost nothing. But how much a design's score
wobbles when you retrain it does not care which designs you chose to look at, so the wobble stays exactly
where it was. Small real differences, unchanged noise, and the ratio between them explodes.

And the top of the catalogue is the only place anyone works. Every search, every comparison, every claim
of an improvement lives there. The regime where the measurement is most trustworthy is the regime nobody
is interested in.

## What to do about it, which is the point

None of this is an argument for despair, and it is not really an argument about other people's papers. It
is an instrument, and instruments are for using.

**Stop buying candidates when you should be buying certainty.** If your cheap evaluation cannot order
anything, running more candidates through it does not help you; it sorts noise faster. The same compute
spent training fewer candidates for longer, or evaluating the same candidates twice and averaging,
produces an actual ordering. Which of those is the better buy is not a matter of taste — it follows from
two numbers you can measure in an afternoon.

**Know when you are finished.** Once your remaining candidates sit inside each other's error bars, further
search is not improving the model. It is relabelling which indistinguishable thing won. That is a stopping
rule, and most search procedures do not have one.

**Read a correlation against its ceiling, not against 1.0.** A large body of work builds cheap predictors
that guess a design's quality without training it, and grades itself by rank correlation against a
catalogue's numbers. Since those numbers are themselves noisy, there is a maximum any predictor could
possibly score: the correlation the catalogue achieves with itself. That ceiling is almost never reported.
A predictor scoring 0.7 where the ceiling is 0.95 is mediocre. The same 0.7 where the ceiling is 0.72 is
close to perfect and deserves applause. And 0.7 where the ceiling is 0.084 needs an explanation.

**And one extra column will tell you how much you are fooling yourself.** Whenever you choose the best of
several things using a held-out score and then report that score, hold out a *second* sample, choose using
only the first, and report both. For a thing chosen without looking at either, the two agree on average, so
any gap is exactly the luck you selected for, measured in the same units as your headline. It is one
column, it has an honest value of zero, and it is very hard to argue with.

## The hundred-year-old punchline

While writing this up I went looking for who had done it before, expecting to find nobody and hoping to
find nobody.

The mathematics is from 1904. Psychologists studying exam scores worked out precisely this relationship
between measurement error and observed correlation, along with the corrections for what happens when you
restrict attention to high scorers. It has a name, a textbook treatment, and a century of refinement. I
had rederived it from scratch over an afternoon, badly, and then measured it on neural networks.

I find that reassuring rather than deflating, for two reasons. It means the theory is not resting on my
algebra. And it is a useful reminder about where new territory actually is: not always in the frontier
paper, sometimes in the gap between a well-understood piece of mathematics and a field that has never
checked whether it applies to them. The formula was never the missing thing. The missing thing was anyone
measuring the two numbers it needs.

*The code and the recovered per-run tables are open, and the measurements regenerate on a laptop in
minutes. If you work on this and want the diagnostic pointed at your own numbers, that takes one table
with three columns and nothing leaves your machine.*
