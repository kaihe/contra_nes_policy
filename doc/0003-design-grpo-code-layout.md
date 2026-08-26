# Organise the GRPO stack around rollout generation

Status: Accepted
Supersedes: —
Depends on: [0002](0002-design-gpt-policy.md) — the policy GRPO fine-tunes.

**Question.** The previous PPO stack was 1,610 lines and was deleted. Before writing
another, which existing project's *organisation* should we borrow — LLaMA-Factory,
slime, or Stable-Baselines3?

**Answer.** **slime's shape, at a third of its machinery.** Our expensive, stateful
component is rollout generation, and slime is the only one of the three organised
around that as the primary axis. SB3's class hierarchy encodes assumptions we already
measured as wrong for us; LLaMA-Factory's stage dispatch assumes BC and GRPO share a
data path, and ours differ completely — shards versus emulator.

**Confidence note.** Stable-Baselines3 is installed here and its layout below was read
from source. LLaMA-Factory and slime are **not** available on this machine; those two
sections are from memory and should be checked against the repos before anything is
built on them — particularly slime, where my knowledge is thinner.

---

## 1. What GRPO actually needs

Comparing organisations is only meaningful against the components they have to hold.
For GRPO on this task:

| component | what makes it awkward |
|---|---|
| **task sampler** | draws a task; GRPO needs the *same* task G times |
| **rollout generation** | one emulator per process, savestate restore per decision, ~12.7% of wall but 100% of the statefulness |
| **group advantage** | `(r_i − mean(r_group)) / std(r_group)` — no critic, no GAE, no bootstrapping |
| **reference KL** | a frozen copy of the BC policy; measured need (item regressed 76.5% → 71.1% without it) |
| **batch assembly** | variable-length episodes → padded batches, already solved by `pad_episodes` |
| **the update** | clipped ratio on 21-way logits, masked to real steps |

Note what is *absent* versus the deleted PPO: no value head, no GAE, no critic warmup,
no `explained_variance` to babysit. GRPO is a smaller thing to organise.

## 2. Stable-Baselines3 — algorithm as a class hierarchy

*Read from the installed package.*

```
common/base_class.py           BaseAlgorithm
common/on_policy_algorithm.py  OnPolicyAlgorithm: collect_rollouts() / train()
common/buffers.py              RolloutBuffer — the contract between them
common/policies.py             ActorCriticPolicy
common/callbacks.py            eval, checkpoint, logging as callbacks
ppo/ppo.py                     PPO(OnPolicyAlgorithm) — just train()
```

The organising idea: **an algorithm is a subclass, and the buffer is the interface.**
`collect_rollouts` fills it, `train` drains it, callbacks observe.

Our tree would become:

```
src/contra_policy/rl/
  base.py        OnPolicyAlgorithm: collect_rollouts() / train()
  buffers.py     GroupRolloutBuffer  (G episodes per task, group-normalised advantages)
  grpo.py        GRPO(OnPolicyAlgorithm)
  callbacks.py   checkpoint, per-family metrics, host-RAM preflight
```

**Buys.** A familiar, well-tested decomposition. The buffer-as-contract is genuinely
the right seam — it is where `build_chunk` lived in the old stack. Callbacks are a
clean home for the per-family Wilson metrics and the memory guard.

**Costs.** We measured `OnPolicyAlgorithm`'s contract as wrong for us in
[0002](0002-design-gpt-policy.md) §5: it collects `while n_steps < n_rollout_steps` and
bootstraps at the window boundary, where we need **complete unbootstrapped episodes**
with a per-task budget. Inheriting the shape means inheriting a `collect_rollouts`
signature built around a fixed step count. And the hierarchy pays for generality —
`BaseAlgorithm` exists to serve DQN, SAC and TD3 too; we have one algorithm.

## 3. LLaMA-Factory — method as a swappable stage

*From memory; not verifiable here.*

The organising idea: **one entry point, a config selects the stage**, and stages share
model/data/trainer plumbing. Roughly `llamafactory-cli train cfg.yaml` with
`stage: sft | ppo | dpo | kto`, each dispatching to a trainer subclass over a common
dataset registry.

Our tree would become:

```
src/contra_policy/
  train.py       one entry; `stage: bc | grpo`
  stages/bc.py
  stages/grpo.py
  data/          registry: shard episodes, emulator rollouts
  config/*.yaml
```

**Buys.** One config schema and one command for everything. Checkpoint, logging and
metric code is written once. Genuinely attractive if BC and GRPO are two objectives
over the same batches.

**Costs.** They are not. BC reads **whole episodes from tar shards**; GRPO **generates
episodes from an emulator**, G at a time from one savestate, and cannot be shuffled or
pre-fetched. Sharing a `data/` registry across those means an abstraction whose two
implementations have nothing in common but the output type. Stage dispatch also tends
to accumulate `if stage ==` branches through shared code, which is exactly the
`index_bias`-style bookkeeping [0002](0002-design-gpt-policy.md) deleted.

## 4. slime — generation and training as separate systems

*From memory, and the thinnest of the three — verify before building.*

As I understand it, slime is organised around **decoupling rollout generation from the
training backend**, with a buffer between them and the rollout function as the primary
extension point. Generation may be asynchronous and is treated as a first-class
subsystem rather than a method on the trainer.

Our tree would become:

```
src/contra_policy/rl/
  rollout.py     emulator, slots, savestate restore; yields complete Episodes
  tasks.py       task catalog + the G-sampler GRPO needs
  buffer.py      Episodes -> group-normalised advantages -> padded batches
  grpo.py        the objective: clipped ratio + reference KL
  trainer.py     the loop; owns neither generation nor the objective
```

**Buys.** It matches where our difficulty actually is. Generation is the stateful,
awkward part — one emulator per process, `PR_SET_PDEATHSIG` worker reaping, savestate
restore per decision — and this shape gives it a boundary instead of a method. It also
makes the G-rollouts-per-task structure explicit at the sampler, which is GRPO's whole
premise. And it lets the same buffer be fed from disk, which is how a rejection-sampling
or replay experiment would reuse the stack without touching the trainer.

Notably, **the deleted PPO stack had already converged on this shape**
(`rollout.py` / `tasks.py` / `trajectory.py` / `workers.py` / `trainer.py`) without
naming it. That is weak evidence it fits the problem.

**Costs.** slime's real machinery — async generation, a separate inference engine,
multi-node data flow — solves problems we do not have. We measured the emulator at
12.7% of wall and rejected the vLLM-style split on those grounds
([0002](0002-design-gpt-policy.md) §5). Borrowing the *shape* is right; borrowing the
apparatus would be the same mistake as adopting SB3.

## 5. Recommendation

**slime's decomposition, SB3's buffer discipline, LLaMA-Factory's config discipline.**
They are not exclusive — only the top-level axis is.

```
src/contra_policy/rl/
  tasks.py       task catalog; samples one task G times           <- GRPO's premise
  rollout.py     emulator -> complete Episodes                    <- the stateful part
  buffer.py      Episodes -> advantages -> padded batches         <- the contract
  grpo.py        clipped ratio + reference KL                     <- the objective
  trainer.py     the loop, metrics, checkpoints
  config_grpo.yaml
```

- from **slime**: generation is a subsystem with a boundary, not a trainer method
- from **SB3**: the buffer is the contract, and it is the only thing the objective sees
- from **LLaMA-Factory**: every knob in one YAML, with the reasoning written beside it

**Carry forward from the deleted stack** (it was not all wrong): `PR_SET_PDEATHSIG`
worker reaping, the budget formula's bit-parity with `contra_nes_evaluation`,
per-family Wilson-interval metrics, and the host-RAM preflight.

**Do not carry forward:** `build_chunk`, `iter_chunks`, memory carry, the `first` flag,
GAE, the value head, critic warmup. All either scaffolding for a windowed model that no
longer exists, or critic machinery GRPO removes.

## 6. Policy-ratio and resumption invariants

GRPO may update only from a behaviour density it can reproduce. Rollout and update
therefore run the trainable policy with stochastic layers disabled; `no_grad` alone is
insufficient because it does not disable dropout. Before the first optimizer step on a
fresh collection, recomputed action probabilities must give `ratio_mean = 1` and
behaviour KL zero within numerical tolerance.

The behaviour KL uses the non-negative k3 estimator for `KL(pi_old || pi_new)`. With
`d = log(pi_old) - log(pi_new)`, its sampled form is `exp(-d) - 1 + d`. Reversing the
exponent estimates a different quantity and makes `target_kl` stop the wrong updates.

An exact resume restores five kinds of state together: policy weights, optimizer state,
task/group sampling streams, action-sampling generator, and trainer minibatch RNG. The
frozen reference remains the original BC checkpoint rather than moving to the resumed
GRPO policy. A checkpoint missing any required continuation state fails loudly.

The rollout actor shares the policy's prefix construction rather than assuming a visual
goal or equal encoder/core widths. This keeps null-goal policies and learned `in_proj`
layers bit-compatible between sequential rollout and full-episode training.

## 7. Risks

| risk | gate |
|---|---|
| my slime description is wrong, and its shape does not fit | read the repo before step 1; the recommendation survives if only the *decomposition* holds |
| G rollouts per task multiplies generation cost by G | measure decisions/s at G=4,8 before choosing; emulator was 12.7% of wall, so G=8 makes it ~50% |
| group advantage is degenerate when all G fail | boss succeeds ~3.5%, so most boss groups are all-zero and contribute no gradient — this is the central open question for GRPO here |

That last row is the one to think hardest about. It is the same thinness that made
rejection sampling unattractive, and GRPO does not automatically escape it.

---

## Appendix — provenance

| claim | source |
|---|---|
| SB3 layout and `collect_rollouts`/`train` contract | read from the installed package, `common/on_policy_algorithm.py` |
| LLaMA-Factory, slime | **from memory — not installed here, not verified** |
| emulator 12.7% of wall | `tools/profile_collect.py`, [0002](0002-design-gpt-policy.md) §1 |
| item regression without reference KL | 500-update PPO run, 76.5% → 71.1% over the first and last 100 updates |
| boss 3.5% success | `contra_nes_evaluation/doc/0005-gpt-bc.md` |
| deleted stack's module layout | commit `b452713` |
