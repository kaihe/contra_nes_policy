# Predict terminal success and game progress separately

Status: Proposed

**Question.** What should the Contra policy-value network predict when successful
episodes can have substantially different accumulated `mc_search` rewards, and how
should stable emulator progress signals contribute to learning?

**Answer.** The main value head predicts the probability that the current episode
will end in task success under searched play. Success supplies target `1`; death,
timeout, or another declared failure supplies target `0`. Accumulated dense reward is
not a value target. Separate auxiliary heads predict normalized level and encounter
progress, including horizontal position and boss damage. These labels come from exact
emulator state at every sampled decision. MCTS backs up terminal success or the value
prediction at an unexpanded leaf, so its Q values retain one stable interpretation.
Progress heads shape the shared representation and provide diagnostics, but do not
replace value: being far through a level is not the same as being likely to survive.

---

## The value head predicts eventual task success

For every non-terminal sampled state, value means the probability of completing the
declared task when actions thereafter follow the current MCTS-improved policy. The
target is the final episode outcome and is copied to all decision states in that
episode:

| terminal classification | value target |
|---|---:|
| boss defeated or level completed | 1 |
| player death | 0 |
| configured decision or wall-clock timeout | 0 |
| emulator or infrastructure failure | discard episode |

Timeout counts as a policy failure only when the emulator ran correctly and reached a
declared task limit. A corrupt restore, worker crash, or invalid observation must not
silently become a negative training example.

The output is constrained to the zero-to-one range and trained with binary
cross-entropy. Report calibration as well as classification quality: validation loss,
Brier score, predicted-success buckets versus realized win rate, and raw-policy win
rate. Mean predicted value alone is not evidence of improvement.

This target deliberately ignores variation in button costs, elapsed decisions, enemy
damage, and other dense rewards among wins. If faster or cheaper victories later become
an explicit objective, they require a separately named cost or return head rather than
changing the meaning of value.

## MCTS backs up success in one consistent unit

A terminal successful leaf evaluates to `1`; a terminal failed leaf evaluates to `0`.
An expanded non-terminal leaf evaluates to the network's success prediction. Backup
propagates that scalar without sign changes because Contra is single-player. Root visit
counts remain the improved policy target.

The cumulative `mc_search` reward must not be added to this backup. Adding damage,
movement, and button rewards to a probability would make Q values neither expected
success nor expected return. The existing dense reward remains useful for episode
diagnostics and for comparing behavior, but the first implementation of this design
does not let it alter PUCT backup or the value label.

If sparse success proves insufficient for search, a later experiment may add bounded
progress guidance as a separately configured selection term. It must remain outside Q,
be reported independently, and be ablated against success-only search. This preserves
the interpretation and calibration of the value head.

## Progress heads predict observable task completion signals

Auxiliary progress labels are decoded from the exact emulator state stored at each
training sample. They supervise the shared CNN/GPT representation but are never used as
network inputs. The initial labels are:

| head | normalized target | applicability |
|---|---|---|
| horizontal progress | current traversable x divided by task-end x | scrolling levels |
| boss damage | one minus current boss HP divided by initial boss HP | active boss fights |
| player survival margin | current HP or remaining hits divided by task maximum | tasks exposing a reliable counter |
| encounter phase | declared pre-boss, boss-active, or completed phase | tasks with a tested decoder |

Each sample carries an applicability mask. A Laser boss sample, for example, trains
boss damage but does not invent horizontal progress when x does not describe encounter
completion. Targets are clipped to the zero-to-one range after decoder validation.
Categorical phase uses cross-entropy; continuous progress heads use a robust regression
loss.

These heads describe state, not future outcome. A nearly defeated boss can coexist with
an unavoidable player death, so progress predictions never substitute for the value
head. At terminal failure, labels retain their final observed progress instead of being
reset to zero. Death reveals the outcome label; it does not erase how far the episode
had progressed.

## Joint training keeps policy, success, and state losses auditable

Each root sample contains image/action context, legal-action mask, normalized MCTS visit
target, terminal outcome, progress labels and masks, and the existing motion, weapon,
and rapid-fire labels. The joint objective contains policy imitation, terminal-success
value prediction, and weighted auxiliary losses. Loss weights are fixed in run
configuration and logged with every checkpoint.

The CNN, GPT, and action head continue to initialize from the selected GPT-policy
checkpoint. The success-value and new progress heads initialize randomly. Existing
motion, weapon, and rapid-fire heads remain auxiliary; their initialization follows
the checkpoint compatibility rules in 0029.

Promotion is based primarily on held-out raw-policy win rate, with search win rate
reported separately. Value calibration and progress errors are gates against a broken
representation, not substitutes for closed-loop success. Training and evaluation must
label the policy being measured as either raw network actions or MCTS-enhanced actions.

---

## Provenance and auditability

| claim | source |
|---|---|
| accumulated dense reward currently supervises value and enters MCTS backup | `src/contra_policy/alphazero.py`; `src/contra_policy/mcts/core.py`; [0029](0029-design-alphazero-contra.md) |
| fixed Laser episodes expose terminal success and decoded boss state | `src/contra_policy/mcts/laser.py` |
| the policy network already has joint value, motion, weapon, and rapid-fire outputs | `src/contra_policy/model.py` |
| root visits supervise the policy in the current iterative loop | `src/contra_policy/alphazero.py`; `src/contra_policy/train_alphazero.py` |

