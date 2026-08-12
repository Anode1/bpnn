# What to ask a team doing post-training on open-weight models

Written for a meeting, 2026-08-12. The goal is one measurement on data they already have.

## The one sentence

"When you run a sweep of fine-tuning configs and pick the best one by validation score, some of that
score is luck rather than quality, and I can tell you how much, from a table you already have."

## Why they should care, in their terms

They pick a checkpoint, a learning rate, a data mix, or a stopping point by comparing candidates on a
validation set, then report the winner's validation number. Picking the maximum of noisy scores keeps
the noise, so the winner is genuinely better *and* got a good day, and the good day does not ship. Two
consequences, and the second is the one that costs money:

1. Their reported improvements are optimistic by an amount nobody has measured.
2. More importantly, if the noise is large relative to the differences between candidates, then the
   config they shipped may not be the best one they trained. That is not a reporting problem, it is a
   worse model in production.

The fix, if the number turns out to be large, is cheap: evaluate the shortlist twice and average, or
split the validation set and select on one half. Both are hours of compute, not a research project.

## The exact data, and it is small

One table, one row per run in a sweep, three columns:

    run_id,  validation_metric_used_for_selection,  metric_on_a_set_never_used_for_any_decision

That is all. Names, prompts, weights, and data stay on their side. Every serious tracking setup
(Weights & Biases, MLflow, a CSV of sweep results) already contains this.

The measurement is then:

    inflation = (val - held_out) for the SELECTED run
              - mean over ALL runs of (val - held_out)

The second term is the baseline offset between the two sets, which exists for every run and has
nothing to do with selection. Whatever the winner has *beyond* that offset was manufactured by the act
of choosing. An unselected run shows none of it, which is what makes the quantity interpretable
without needing a control group.

## Three questions to ask, in order of value

1. **"For a typical sweep, how many configs do you compare before picking one?"**
   This is selection intensity and it drives the whole effect. On the benchmark we measured, inflation
   roughly doubled between 64 and 1,024 candidates.
2. **"How big is the set you select on, and do you ever evaluate the same config twice?"**
   Set size gives the noise scale. Repeated evaluation of the same config is the stronger version of
   the measurement, because two evaluations of one config are exchangeable and let us separate
   evaluation noise from real differences.
3. **"Is there an eval set you look at only once, at the end?"**
   If yes, that is the second column and we are done. If no, that is itself the finding: there is no
   quantity in their pipeline that selection has not touched, and the first fix is to create one.

## What they get

A number for their own pipeline, the script that produced it so they can rerun it whenever, and a
short write-up they own. If the inflation is small, that is a clean result they can point at when
someone questions their evals. If it is large, they have a cheap fix and a better shipped model.

## The objection to pre-empt

They will not hand over eval data, and they should not have to. Offer it the other way round: send
them a short script, they run it on their own table, and they tell us only the resulting numbers.
Nothing sensitive moves. That removes the usual reason this kind of collaboration dies.

## Fallback if they have no untouched set

Ask whether any config was trained more than once with a different seed. If so, the two runs of the
same config are exchangeable, and the same measurement works with one taking the role of the held-out
set. On NAS-Bench-101 that is exactly the structure we used, and it is why the benchmark could answer
the question at all.

## What we do NOT ask for

Compute, GPU time, model access, or a joint paper commitment. The ask is one table or one script run.
Keep it that small; it is the reason it might actually happen.


---

# Post-training specifically: why it is the better setting

Learning that they do post-training rather than plain supervised fine-tuning makes this a stronger
ask, not a weaker one. Three reasons.

**The noise is larger.** Post-training quality is judged on small eval sets, often a few hundred to a
few thousand prompts, and frequently through an LLM judge. That adds a second noise source on top of
training variance: rejudge the same outputs and the verdict moves, from judge sampling, from position
bias, from ties broken differently. The quantity being maximised is therefore noisier than a
classification accuracy, and everything we measured scales with that noise.

**Selection intensity is higher.** A post-training program sweeps data mixes, checkpoints, learning
rates, epochs, preference-optimisation strength, and reward-model variants. The number of candidates
compared before something ships is in the dozens to hundreds. On the benchmark we measured, inflation
roughly doubled from 64 candidates to 1,024, so this is the regime where it is largest.

**And the replicate structure is free.** This is the part that makes the measurement cheap for them.
Two judgments of the same model outputs are exchangeable: same model, same prompts, different judge
randomness. Rejudging costs inference on outputs they already generated, no retraining at all. That
gives exactly the interchangeable pair the measurement needs, and as a by-product it separates judge
noise from real model differences, which is worth knowing on its own.

## The sharpest single question for a post-training team

**"When you pick the best checkpoint off an eval curve, how much of that peak is noise?"**

Everyone does this. You evaluate every few hundred steps, the curve wobbles, you keep the best point.
That is taking a maximum over many noisy measurements, which is the winner's curse in its purest form,
and the chosen peak is optimistic by an amount that grows with how many checkpoints were compared and
with how noisy each evaluation is. They already have the curve, so the data cost is zero.

Two ways to measure it, whichever they can do:

    (a) evaluate the chosen checkpoint and a few unchosen ones on a second eval, or rejudge them; or
    (b) split the eval prompts in half, select the peak on one half, and read the same checkpoint on
        the other half.

Both give the same quantity. Option (b) needs no new inference whatsoever if they logged per-prompt
results, which is the thing worth asking about early.

## A technical point that widens where this applies

The estimator does not require the two sets to be exchangeable, and this matters because a post-training
team usually selects on an internal dev set and reports on public benchmarks, which are plainly
different distributions. Subtracting the mean of (selection metric minus reported metric) taken over
ALL candidates removes whatever systematic offset exists between the two sets. Only the winner's
excess over that average survives, and that excess is selection. Exchangeability is needed only for the
simpler claim that the raw gap should be zero.

So: internal dev set for selection, public benchmark for reporting, and the measurement still works
provided they kept the scores of the candidates they rejected. Those rejected candidates are the whole
value; a table containing only the winner cannot answer anything.

## The one thing to make sure he keeps

If he takes nothing else away: **log the eval scores of the candidates you did not pick.** Most
pipelines discard them. They are what turns an unanswerable question into a two-line computation.
