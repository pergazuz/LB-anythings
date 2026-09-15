# 4. The Retrain Threshold grows with the Training Set

Date: 2026-09-15

## Status

Accepted. Supersedes the fixed-threshold rule described in the spec's story 25.

## Context

The rule was: start a Training Run whenever the Training Set's size is a multiple of
`LB_RETRAIN_EVERY` (25 by default). A fixed number of Examples, forever.

Two problems.

**A fixed step buys less and less.** Model accuracy improves as a *power law* in training set
size — measured over several orders of magnitude across machine translation, language
modelling, image processing and speech recognition, with exponents between -0.07 and -0.35
([Hestness et al., *Deep Learning Scaling is Predictable, Empirically*](https://arxiv.org/pdf/1712.00409)).
A power law is a straight line on a log-log plot, which means equal *relative* increases in
data buy roughly equal improvements, and equal *absolute* increases buy less and less. Going
from 25 Examples to 50 doubles the Training Set. Going from 1000 to 1025 grows it by 2.5% and
will not move mAP measurably — but it costs a full Training Run of GPU time either way.

This matches what active-learning practice already does: adaptive acquisition schedules use
small batches early, when each Annotation is most influential, and larger batches later to
avoid repeated retraining on near-identical data — for example 25 while under 100 labelled
samples, 50 between 100 and 300, and 75 beyond that
([Sequential and Batch Active Learning](https://www.emergentmind.com/topics/sequential-and-batch-active-learning)).

**A multiple can be missed entirely.** `size % threshold == 0` only fires on the exact
multiple. If a Training Run is already active at that moment — which is common, because an
Annotation that trips the threshold is often followed closely by more — the decision is
declined and the next opportunity is a whole threshold away. Observed live: with a threshold
of 1, an Annotation at size 22 was refused because the run launched at 21 was still going, and
that Example never triggered anything.

## Decision

Retrain when the Training Set has **grown since the last launched Training Run** by at least

```
max(LB_RETRAIN_EVERY, ceil(size_at_last_run × LB_RETRAIN_GROWTH))
```

`LB_RETRAIN_EVERY` keeps its name and becomes a floor: never retrain on fewer than this many
new Examples. `LB_RETRAIN_GROWTH` defaults to `0.10`, so once the Training Set passes ten times
the floor the proportional term takes over:

| Training Set | Next run at | Step |
|---|---|---|
| 50 | 75 | 25 (floor) |
| 250 | 275 | 25 (floor) |
| 500 | 550 | 50 |
| 1000 | 1100 | 100 |
| 5000 | 5500 | 500 |

Measuring growth against the last run rather than against a multiple fixes the second problem
for free: when a run is already active the baseline has not moved, so the next Annotation
tries again instead of waiting a whole step.

The baseline is the Training Set size **at launch**, recorded in the run record the Trainer
already keeps. Launch rather than completion, because it is written exactly once per run and
cannot re-trigger a run that is still going.

A run that *fails* reports no baseline at all, so the next Annotation is free to try again
rather than wait out a step for Examples nothing ever learned from. This is not hypothetical:
an end-to-end pass caught a Training Run killed nine seconds in by
`forrtl: error (200): program aborting due to window-CLOSE event`, a Windows console event
reaching the Intel Fortran runtime underneath torch. It did not reproduce on the next pass, and
the service handled it correctly — the run was derived as failed, `/is_training` went false and
the previous Checkpoint kept serving — but the Training Set would have been stranded until 368
Examples for a run that never happened. The retry costs at most one extra attempt per
Annotation, and Annotations are human-paced.

When there is no run record at all — a fresh install, or Examples migrated from the previous
backend — there is no baseline and the first Training Run starts as soon as `LB_MIN_EXAMPLES`
is met. This is a deliberate change: previously, migrating 333 Examples meant waiting for 350
before anything trained on them, which is exactly backwards.

## Consequences

- Retraining is frequent while the Training Set is small and each Annotation matters, and rare
  once it is large and each Annotation is a rounding error. GPU time follows value.
- A run is no longer skipped because another was in flight.
- The Trainer port gains the Training Set size at launch, in both directions: `start` takes it,
  and a new `trained_at_size` reads back what the last launch recorded.
- `LB_RETRAIN_EVERY` no longer means "every N Examples" but "at least N new Examples". Existing
  configurations keep working and retrain no more often than before.
- The rule is still pure and lives in `domain/retraining.py`; only its inputs changed.
- Setting `LB_RETRAIN_GROWTH=0` restores the old proportional-free behaviour, though still
  measured as growth-since-last-run rather than as a multiple.
