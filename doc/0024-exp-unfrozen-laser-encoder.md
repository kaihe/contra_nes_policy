# Does adapting the static image projection improve Laser boss success?

Status: Implemented

## 1. Goal

The matched static and temporal frozen encoders reach only 19% and 17% Laser-start success,
respectively. Full end-to-end fine-tuning is badly compute-imbalanced: the convolutional
encoder would perform about 55 times the arithmetic of the L temporal core per episode.

Freeze the convolutional trunk but adapt its projection from the retained `4×4×256` feature
map in matched goal-free L/D10k/C20k cells. This tests task-specific visual adaptation
without recomputing the expensive trunk or relying on the already-compressed 512-D token.

After projection tuning failed, extend the winning frozen cell to C40k as a bounded
memorization diagnostic. This asks whether the still-falling training CE can improve the
same in-distribution closed-loop probe; it is not a test of held-out generalization.

## 2. Setup

Use float16 `4×4×256` features emitted after the frozen encoder's `reduce` block, before
its `proj` and `token_ln`. The datahouse artifact uses the exact Laser D10k fingerprint
membership and uid-digest holdout. Preserve the causal shift: feature `i` predicts action
`i + 1`. The learned null goal is shared by both cells because Laser has one task goal.

Common recipe: L core with `d_model=640`, 5 layers, 10 heads; learned null goal token;
20,000 optimizer cycles; batch 32; AdamW at one `1e-4` rate for core and projection; 500-step
warmup; WSD with cooldown from 18,000 to 20,000; bf16; dropout 0.2; seed 0; no value or
auxiliary head. The trunk through `reduce` remains frozen in both cells.

| run | encoder | data | cycles | state | dir |
|---|---|---:|---:|---|---|
| `L-D10k-C20k-laser-null-goal-proj-frozen` | frozen projection | reduced-feature D10k | 20,000 | done | `runs/laser-projection/L-D10k-C20k-laser-null-goal-proj-frozen` |
| `L-D10k-C20k-laser-null-goal-proj-tuned` | trainable `proj` + `token_ln`, LR `1e-4` | reduced-feature D10k | 20,000 | done | `runs/laser-projection/L-D10k-C20k-laser-null-goal-proj-tuned` |
| `L-D10k-C40k-laser-null-goal-proj-frozen` | frozen projection | reduced-feature D10k | 40,000 | done | `runs/laser-projection/L-D10k-C40k-laser-null-goal-proj-frozen` |

The reduced cache is approximately 8.4 GB for 1.02M frames. Run a 200-step smoke test before
the full cells and retain checkpoints at 2,000, 5,000, 10,000, and 20,000 cycles.
The C40k cell resumes the extendable WSD trunk at step 18,000, before any cooldown update,
then remains at `1e-4` through step 36,000 and cools to zero by step 40,000. It saves the
new step-36,000 trunk and final checkpoint.

## 3. Evaluation metrics

| metric | frozen static baseline | unfrozen candidate | source |
|---|---:|---:|---|
| final Laser success | 25/100, 25% [17.5, 34.3] | 19/100, 19% [12.5, 27.8] | evaluation 0030 |
| death / timeout | 74 / 1 | 81 / 0 | evaluation 0030 |
| boss seen | 100% | 100% | evaluation 0030 |
| mean boss damage | 47.1% | 43.8% | evaluation 0030 |
| action entropy | 2.304 bits | 2.212 bits | evaluation 0030 |
| final validation CE | 2.2097 | 2.1729 | policy `metrics.csv` |
| median step time, batch 32 | 53.32 ms | 55.98 ms | 40-step `tools/bench_step.py`, 15 warmup, 2 workers |
| peak GPU memory | 2.38 GiB | 2.53 GiB | same smoke benchmark |

Closed-loop evaluation uses the evaluation 0029 protocol on `policy-final.pt`: Laser start,
100 rollouts, temperature 1.0, seed 0, twice the expert budget, bf16, and batch 8. Report
the Wilson 95% interval, death/timeout counts, boss-seen rate, damage fraction, and entropy.

Closed-loop results and checkpoint provenance are recorded in evaluation
[0030](../../contra_nes_evaluation/doc/0030-result-laser-projection-adaptation.md).
The matched C40k follow-up is recorded in evaluation
[0031](../../contra_nes_evaluation/doc/0031-result-laser-projection-c40k.md).

| frozen-projection compute | C20k | C40k | source |
|---|---:|---:|---|
| final Laser success | 25/100, 25% [17.5, 34.3] | 27/100, 27% [19.3, 36.4] | evaluation 0031 |
| death / timeout | 74 / 1 | 73 / 0 | evaluation 0031 |
| mean boss damage | 47.1% | 48.1% | evaluation 0031 |
| action entropy | 2.304 bits | 2.293 bits | evaluation 0031 |
| mean train CE over final 2k steps | 0.2378 | 0.1961 | policy `metrics.csv` |
| final validation CE | 2.2097 | 2.5365 | policy `metrics.csv` |

## 4. Conclusion

Adapting `proj` and `token_ln` is not promising. It slightly lowers final validation CE
(2.2097 to 2.1729) but lowers closed-loop success by 6 points (25% to 19%); the Wilson
intervals overlap, and damage and action entropy also favor the frozen cell. Keep the image
projection frozen and do not spend more compute tuning this representation.

The next experiment should target the policy's closed-loop state distribution rather than
the encoder. Collect expert-labelled recovery trajectories from states reached by failed
Laser-policy rollouts (DAgger-style), mix them with the existing demonstrations, and compare
one frozen-encoder cell against the current 25% baseline. This directly tests compounding
error, which is consistent with low training CE and poor closed-loop success, while retaining
the cheapest and best-performing visual setup from this experiment.

The C40k extension confirms that this recipe is not simply optimization-limited. Doubling
compute lowers mean train CE over the final 2,000 steps from 0.2378 to 0.1961, while success
moves only from 25% to 27% with strongly overlapping intervals and nearly unchanged secondary
metrics. The predeclared gate therefore fails: do not continue this recipe to C80k. Move to
recovery-state data rather than spending more cycles or capacity on demonstration memorization.
