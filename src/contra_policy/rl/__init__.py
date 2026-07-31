"""Recurrent PPO fine-tuning for the Contra cross-view policy.

The behaviour-cloning path (``train.py`` / :mod:`contra_policy.lit`) is untouched;
this package is a second, independent entry point (``train_rl.py``). Its job is the
one thing BC cannot do: every recorded trace is a *win*, so no failure state is ever
demonstrated and the policy never learns to recover from one. RL supplies exactly
that, by letting the policy visit its own states and scoring the outcome.

What lives where, and why the boundaries are the way they are:

============================  ==========================================================
module                        role
============================  ==========================================================
:mod:`~contra_policy.rl.tasks`      training-split task catalog, cross-view prompt join,
                                    the family/label sampling mixture
:mod:`~contra_policy.rl.rollout`    one emulator, N slots, complete recurrent episodes
:mod:`~contra_policy.rl.trajectory` the stored episode, returns/GAE, PPO sequence batching
:mod:`~contra_policy.rl.ppo`        the clipped objective and the auxiliary losses
:mod:`~contra_policy.rl.workers`    multi-process collection (one emulator per process)
:mod:`~contra_policy.rl.checkpoint` resumable state + evaluator-compatible weights
:mod:`~contra_policy.rl.trainer`    the loop: warmup → collect → update → log → save
============================  ==========================================================

Three rules this package is built around.

**Nothing about the game is defined here.** Task completion is
``TaskMaker.goal_reached``, failure is ``env.event``'s ``die``, the task format is
``task_maker.base.load_task``, the per-frame goal position is
``task_maker.export_hf._goal_points`` — all imported from ``contra_nes_data``. A
second definition of "did the task succeed" would make the reward quietly disagree
with the evaluation harness, which is the one failure mode that cannot be caught by
watching a training curve.

**RAM is a source of reward and of training labels, never of policy input.** The
policy sees the same four things it saw in BC — pixels, the goal prompt, the
interaction id, and the (always-unknown) previous-action token.

**Nothing from ``contra_nes_evaluation`` is imported.** That repo must stay
deletable. The rollout loop here is therefore a second implementation of the same
stepping order, and ``tools/parity_vs_evaluation.py`` exists to prove the two agree
on real tasks rather than to assume it.
"""

from __future__ import annotations
