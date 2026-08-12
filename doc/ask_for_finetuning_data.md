# What to ask a team that fine-tunes open-weight models

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
