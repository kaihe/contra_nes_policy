"""Closed-loop episode collection: one emulator, N slots, complete recurrent episodes.

One decision step is: show the policy the agent's current screen, sample an action,
hold that action's buttons for ``seg.skip`` NES frames, then ask the task's own
``goal_reached`` predicate about the ``(before, after)`` RAM pair. That is the same
step the evaluation harness runs, and it has to stay the same step — a reward that
disagrees with the metric optimises the wrong thing without ever looking wrong.

Every game-side definition is imported, never restated:

============================  ==============================================
what                          from
============================  ==============================================
task completion               ``TaskMaker.goal_reached`` (per family)
episode failure               ``env.event.make_terminal_events`` → ``die``
task format / save-state      ``task_maker.base.load_task``
per-frame goal position       ``task_maker.export_hf._goal_points``
emulator                      ``util.replay.make_env`` / ``rewind_state``
action table                  :mod:`contra_policy.action_space`
============================  ==============================================

Three constraints shape the implementation, all of them the same ones
``contra_nes_evaluation`` hit:

**One emulator per process.** ``stable_retro`` refuses a second instance, so slots
cannot each own an env — they own a *save-state*, restored into a single env before
their step and stored back after it. That costs ~3% against the decision step itself
and buys a genuinely batched GPU forward across slots.
:func:`claim_emulator` makes the constraint an assertion rather than a comment.

**A freshly restored save-state has not rendered a frame.** Its framebuffer is
uniform grey, so there is no screen to show the policy at ``t=0``. :meth:`_peek`
steps the emulator briefly, keeps the frame, and restores the state it started from,
so the policy acts from the task's exact starting position with the game clock
untouched.

**Success is checked before death.** On the step that clears a boss the player can
also register a hit; completing the task is the outcome that matters, and the
evaluation harness resolves the tie the same way.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from contra_policy.action_space import ACTION_NAMES, actions_np
from contra_policy.rl.tasks import GoalPrompt, RLTask, TaskCatalog, TaskSampler, resize_to_input
from contra_policy.rl.trajectory import Episode

IDLE_ACTION = ACTION_NAMES.index("_")

#: Set once a process opens an emulator. stable_retro allows exactly one instance per
#: process; a second `make_env()` fails deep inside the emulator with an error that
#: does not mention the real cause, so the constraint is asserted here instead.
_EMULATOR_OWNER: Optional[str] = None


def claim_emulator(owner: str) -> None:
    """Record that ``owner`` holds this process's single emulator, or raise."""
    global _EMULATOR_OWNER
    if _EMULATOR_OWNER is not None:
        raise RuntimeError(
            f"this process already owns an emulator ({_EMULATOR_OWNER}); stable_retro "
            f"allows exactly one instance per process, so {owner} must run in a worker "
            f"process of its own")
    _EMULATOR_OWNER = owner


def release_emulator() -> None:
    global _EMULATOR_OWNER
    _EMULATOR_OWNER = None


def classify_step(goal_reached: bool, died: bool, steps: int, budget: int,
                  stop_on_death: bool = True) -> Tuple[bool, str, bool]:
    """Resolve one step's outcome. Returns ``(done, outcome, died)``.

    Pulled out of :meth:`EpisodeCollector._step` because the *order* of these three
    checks is the entire semantics of the reward, and it is worth testing without an
    emulator in the room:

    1. **Success first.** On the step that clears a boss the player can also register
       a hit. Completing the task is the outcome that matters, and the evaluation
       harness resolves the tie the same way — scoring that step as a death would
       train the policy away from the winning move.
    2. **Death second**, and only when ``stop_on_death`` (the evaluated default).
    3. **Budget last**, and only if neither of the above fired: exhausting the task's
       real evaluation budget is a terminal failure worth exactly 0.

    All three are true terminals. None of them bootstraps.
    """
    if goal_reached:
        return True, "success", died
    if died and stop_on_death:
        return True, "death", True
    if steps >= budget:
        return True, "timeout", died
    return False, "", died


def budget_for(seg, budget_mult: float, min_budget: int) -> int:
    """The task's decision-step budget — the evaluation harness's formula.

    ``contra_nes_evaluation`` uses ``max(min_budget, ceil(budget_mult * expert_steps))``
    with ``budget_mult=2.0`` and ``min_budget=24``. Both are config keys here and both
    must keep matching, because the budget is what makes ``timeout`` mean the same
    thing in training and in the report.
    """
    return max(min_budget, int(math.ceil(budget_mult * len(seg.actions))))


# ── the batched actor ─────────────────────────────────────────────────────────

@dataclass
class RolloutObservation:
    """One synchronous step across all slots. Inactive slots carry zeros.

    Keeping the leading dimension fixed at the slot count is what lets the recurrent
    memory stay a single tensor per block instead of a ragged per-slot structure.
    """

    image: np.ndarray          # (B, S, S, 3) uint8
    goal_image: np.ndarray     # (B, S, S, 3) uint8
    goal_mask: np.ndarray      # (B, S, S) uint8
    interaction: np.ndarray    # (B,) int64
    prev_action: np.ndarray    # (B,) int64
    active: np.ndarray         # (B,) bool


class RolloutActor:
    """Steps :class:`CrossViewContraRocket` one decision at a time, batched over slots.

    Memory handling is the part that is easy to get quietly wrong. The recurrent state
    is a flat list of tensors sharing a leading batch dimension, so recycling slot *i*
    onto a new episode means writing a fresh ``initial_state`` row into position *i* of
    every one of them — :meth:`reset`. Miss it and the new episode attends to the
    previous one's context, which shows up only as a mildly implausible success rate.

    ``prev_action_dropout`` is always zero: the previous-action token is always the
    learned "unknown" embedding, which is the configuration the 72.8% baseline was
    measured in and the one PPO must therefore optimise.
    """

    def __init__(self, model, batch_size: int, *, device: torch.device,
                 temperature: float = 1.0, seed: int = 0, precision: str = "bf16"):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.temperature = float(temperature)
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                               "fp32": None}[precision]
        if device.type != "cuda":
            self.autocast_dtype = None
        self.generator = torch.Generator(device=device).manual_seed(int(seed))
        self.memory: List[torch.Tensor] = [
            s.to(device) for s in model.recurrent.initial_state(batch_size)]

    def reset(self, slots: Sequence[int]) -> None:
        if not len(slots):
            return
        fresh = self.model.recurrent.initial_state(len(slots))
        idx = torch.as_tensor(list(slots), dtype=torch.long, device=self.device)
        for tensor, new in zip(self.memory, fresh):
            tensor[idx] = new.to(device=tensor.device, dtype=tensor.dtype)

    def reset_all(self) -> None:
        self.reset(range(self.batch_size))

    def _model_input(self, obs: RolloutObservation) -> Dict:
        def to_dev(arr, dtype):
            return torch.from_numpy(np.ascontiguousarray(arr)).to(self.device, dtype)

        b = len(obs.active)
        return {
            "image": to_dev(obs.image, torch.uint8).unsqueeze(1),      # (B, 1, S, S, 3)
            "cross_view": {
                "cross_view_image": to_dev(obs.goal_image, torch.uint8),
                "cross_view_obj_mask": to_dev(obs.goal_mask, torch.uint8),
                "cross_view_obj_id": to_dev(obs.interaction, torch.long).unsqueeze(1),
            },
            "prev_action": to_dev(obs.prev_action, torch.long).unsqueeze(1),
            "prev_action_dropout": torch.zeros((b, 1), dtype=torch.float32,
                                               device=self.device),
        }

    @torch.no_grad()
    def act(self, obs: RolloutObservation) -> Dict[str, np.ndarray]:
        """Sample one action per slot; also report its log-prob and the critic.

        The log-prob is the density of the **sampling** distribution, i.e. it carries
        the temperature. That makes it the true behaviour-policy density, which is
        what a PPO ratio has to be taken against. It also means ``temperature != 1``
        is not a valid PPO configuration — the objective evaluates the untempered
        policy — and :class:`~contra_policy.rl.trainer.RLTrainer` refuses it.
        ``temperature = 0`` (greedy) is for deterministic replay, not for training.
        """
        ctx = (torch.autocast("cuda", dtype=self.autocast_dtype)
               if self.autocast_dtype is not None else _null_context())
        with ctx:
            latents, self.memory = self.model(self._model_input(obs), self.memory)

        logits = latents["pi_logits"][:, -1, :].float()
        value = latents["vpred"][:, -1, 0].float()
        if self.temperature <= 0:
            actions = logits.argmax(dim=-1)
            sampling_logits = logits
        else:
            sampling_logits = logits / self.temperature
            probs = torch.softmax(sampling_logits, dim=-1)
            actions = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
        logprob = torch.log_softmax(sampling_logits, dim=-1).gather(
            -1, actions.unsqueeze(-1)).squeeze(-1)
        return {
            "action": actions.cpu().numpy().astype(np.int64),
            "logprob": logprob.cpu().numpy().astype(np.float32),
            "value": value.cpu().numpy().astype(np.float32),
            "point": latents["point"][:, -1, :].float().cpu().numpy(),
            "exist": torch.sigmoid(latents["exist"][:, -1, 0].float()).cpu().numpy(),
        }

    def state(self) -> dict:
        return {"generator": self.generator.get_state()}

    def load_state(self, state: dict) -> None:
        self.generator.set_state(state["generator"].to("cpu")
                                 if hasattr(state["generator"], "to")
                                 else state["generator"])


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ── slots ─────────────────────────────────────────────────────────────────────

@dataclass
class _Slot:
    """A rollout in flight, held as a save-state rather than an emulator."""

    task: RLTask
    seg: object
    maker: object
    prompt: GoalPrompt
    budget: int
    state: bytes = b""
    frame: Optional[np.ndarray] = None    # (S, S, 3) uint8 at network resolution
    prev_ram: Optional[np.ndarray] = None
    prev_action: int = IDLE_ACTION
    steps: int = 0
    done: bool = False
    outcome: str = "timeout"
    died: bool = False
    #: True once the episode has been artificially cut and is waiting for one more
    #: forward pass purely to read V(s_T) off the critic.
    awaiting_bootstrap: bool = False
    bootstrap_value: float = 0.0
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    prev_actions: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    goal_points: List[List[List[int]]] = field(default_factory=list)
    goal_visible: List[bool] = field(default_factory=list)


# ── the collector ─────────────────────────────────────────────────────────────

class EpisodeCollector:
    """Runs whole episodes through one emulator and returns complete trajectories.

    ``collect`` keeps starting episodes until **both** targets are met — at least
    ``min_episodes`` started and at least ``min_steps`` transitions taken — then
    drains the slots still in flight. PPO therefore always updates from a batch of
    many finished episodes rather than from one episode at a time, which with a binary
    episodic return is the difference between an advantage estimate and a coin flip.
    """

    def __init__(self, model, catalog: TaskCatalog, sampler: TaskSampler, *,
                 batch_size: int = 16, budget_mult: float = 2.0, min_budget: int = 24,
                 image_size: int = 256, device: torch.device = torch.device("cpu"),
                 temperature: float = 1.0, precision: str = "bf16", seed: int = 0,
                 reward: Optional[Dict[str, float]] = None,
                 max_episode_steps: int = 0, stop_on_death: bool = True,
                 collect_goal_points: bool = True, owner: str = "EpisodeCollector"):
        catalog.assert_split("train")
        self.catalog = catalog
        self.sampler = sampler
        self.batch_size = batch_size
        self.budget_mult = budget_mult
        self.min_budget = min_budget
        self.image_size = image_size
        self.max_episode_steps = int(max_episode_steps)
        self.stop_on_death = stop_on_death
        self.collect_goal_points = collect_goal_points
        self.reward = {"success": 1.0, "death": 0.0, "timeout": 0.0, "step": 0.0,
                       "truncated": 0.0, **(reward or {})}
        self.actor = RolloutActor(model, batch_size, device=device,
                                  temperature=temperature, seed=seed, precision=precision)
        self._vectors = actions_np(np.uint8)
        self._die = None
        self._makers: Dict[str, object] = {}
        self._env = None
        self._owner = owner

    # -- emulator lifecycle ------------------------------------------------

    def open(self) -> None:
        if self._env is not None:
            return
        from env.event import make_terminal_events
        from util.replay import make_env

        claim_emulator(self._owner)
        self._env = make_env()
        for ev in make_terminal_events():
            if ev.tag == "die":
                self._die = ev
                break
        if self._die is None:
            raise RuntimeError("env.event.make_terminal_events() no longer defines 'die'")

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            release_emulator()

    def __enter__(self) -> "EpisodeCollector":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _maker(self, family: str):
        """The ``TaskMaker`` whose ``goal_reached`` defines completion for this family."""
        if family not in self._makers:
            from task_maker.kill_boss import KillBossMaker
            from task_maker.kill_enemy import KillEnemyMaker
            from task_maker.pick_item import PickItemMaker
            from task_maker.traverse import TraverseMaker

            makers = {"kill": KillEnemyMaker, "item": PickItemMaker,
                      "traverse": TraverseMaker, "boss": KillBossMaker}
            if family not in makers:
                raise KeyError(f"no task maker for family {family!r}")
            self._makers[family] = makers[family]()
        return self._makers[family]

    # -- collection --------------------------------------------------------

    def collect(self, min_steps: int, min_episodes: int,
                tasks: Optional[Sequence[RLTask]] = None) -> List[Episode]:
        """Roll out until both targets are met, then drain. Returns finished episodes.

        ``tasks``, when given, replaces the sampler with a fixed queue — that is the
        path the frozen-weights parity check and the deterministic tests use, and it
        is the only way an episode start is not drawn from :class:`TaskSampler`.
        """
        self.open()
        queue = list(tasks) if tasks is not None else None
        slots: List[Optional[_Slot]] = [None] * self.batch_size
        out: List[Episode] = []
        started = steps = 0

        def want_more() -> bool:
            if queue is not None:
                return bool(queue)
            return started < min_episodes or steps < min_steps

        try:
            while True:
                fresh = []
                for i in range(self.batch_size):
                    if slots[i] is None and want_more():
                        task = queue.pop(0) if queue is not None else self.sampler.sample()
                        slots[i] = self._start(task)
                        started += 1
                        fresh.append(i)
                if fresh:
                    # Must happen before the next act(): a recycled slot still holds the
                    # previous episode's attention memory until it is overwritten.
                    self.actor.reset(fresh)
                if all(s is None for s in slots):
                    break

                obs = self._observe(slots)
                step = self.actor.act(obs)
                for i, slot in enumerate(slots):
                    if slot is None:
                        continue
                    if slot.awaiting_bootstrap:
                        slot.bootstrap_value = float(step["value"][i])
                        out.append(self._finish(slot))
                        slots[i] = None
                        continue
                    self._record(slot, int(step["action"][i]), float(step["logprob"][i]),
                                 float(step["value"][i]))
                    self._step(slot, int(step["action"][i]))
                    steps += 1
                    if slot.done:
                        out.append(self._finish(slot))
                        slots[i] = None
                    elif self.max_episode_steps and slot.steps >= self.max_episode_steps:
                        # An *artificial* cut: the episode is neither solved nor lost, so
                        # its tail value has to come from the critic. One more forward
                        # pass reads it, then the slot is freed.
                        slot.outcome = "truncated"
                        slot.awaiting_bootstrap = True
        finally:
            pass
        return out

    # -- slot lifecycle ----------------------------------------------------

    def _start(self, task: RLTask) -> _Slot:
        from util.replay import rewind_state

        if task.split != "train":
            raise RuntimeError(f"task {task.uid!r} is in the {task.split!r} split; "
                               f"RL workers may only see training tasks")
        seg = self.catalog.segment(task)
        slot = _Slot(task=task, seg=seg, maker=self._maker(task.family),
                     prompt=self.catalog.prompt(task),
                     budget=budget_for(seg, self.budget_mult, self.min_budget))
        rewind_state(self._env, seg.initial_state)
        slot.state = self._env.em.get_state()
        slot.prev_ram = self._env.unwrapped.get_ram().copy()
        slot.prev_action = IDLE_ACTION
        slot.frame = self._peek(slot)
        return slot

    def _peek(self, slot: _Slot) -> np.ndarray:
        """Render a frame for a state without advancing that state's clock."""
        from util.replay import rewind_state

        idle = self._vectors[IDLE_ACTION]
        for _ in range(slot.seg.skip):
            self._env.step(idle)
        screen = np.ascontiguousarray(self._env.unwrapped.get_screen())
        frame = resize_to_input(screen, self.image_size)
        rewind_state(self._env, slot.state)
        return np.ascontiguousarray(frame)

    def _observe(self, slots: Sequence[Optional[_Slot]]) -> RolloutObservation:
        b, s = self.batch_size, self.image_size
        image = np.zeros((b, s, s, 3), dtype=np.uint8)
        goal_image = np.zeros((b, s, s, 3), dtype=np.uint8)
        goal_mask = np.zeros((b, s, s), dtype=np.uint8)
        interaction = np.zeros(b, dtype=np.int64)
        prev_action = np.zeros(b, dtype=np.int64)
        active = np.zeros(b, dtype=bool)
        for i, slot in enumerate(slots):
            if slot is None:
                continue
            active[i] = True
            image[i] = slot.frame
            goal_image[i] = slot.prompt.image
            goal_mask[i] = slot.prompt.mask
            interaction[i] = slot.prompt.interaction
            prev_action[i] = slot.prev_action
        return RolloutObservation(image=image, goal_image=goal_image, goal_mask=goal_mask,
                                  interaction=interaction, prev_action=prev_action,
                                  active=active)

    def _record(self, slot: _Slot, action: int, logprob: float, value: float) -> None:
        slot.obs.append(slot.frame)
        slot.actions.append(action)
        slot.prev_actions.append(slot.prev_action)
        slot.logprobs.append(logprob)
        slot.values.append(value)
        if self.collect_goal_points:
            from task_maker.export_hf import _goal_points

            pts, visible = _goal_points(slot.seg, slot.prev_ram)
            slot.goal_points.append([[int(x), int(y)] for x, y in pts])
            slot.goal_visible.append(bool(visible))

    def _step(self, slot: _Slot, action: int) -> None:
        from util.replay import rewind_state

        rewind_state(self._env, slot.state)
        vector = self._vectors[action]
        for _ in range(slot.seg.skip):
            self._env.step(vector)
        cur = self._env.unwrapped.get_ram().copy()
        slot.steps += 1

        # Both predicates are always evaluated (they are cheap RAM reads) so that
        # `died` is recorded even on the step the task is completed; `classify_step`
        # owns the ordering.
        reached = bool(slot.maker.goal_reached(slot.seg, slot.prev_ram, cur))
        died_now = slot.died or bool(self._die.trigger(slot.prev_ram, cur))
        slot.done, outcome, slot.died = classify_step(
            reached, died_now, slot.steps, slot.budget, self.stop_on_death)
        if slot.done:
            slot.outcome = outcome

        if not slot.done:
            slot.frame = np.ascontiguousarray(
                resize_to_input(np.ascontiguousarray(self._env.unwrapped.get_screen()),
                                self.image_size))
            slot.state = self._env.em.get_state()
            slot.prev_ram = cur
            slot.prev_action = action

    def _finish(self, slot: _Slot) -> Episode:
        n = len(slot.actions)
        rewards = np.full(n, self.reward["step"], dtype=np.float32)
        if n:
            rewards[-1] += float(self.reward[slot.outcome])
        return Episode(
            family=slot.task.family, label=slot.task.label, uid=slot.task.uid,
            interaction=slot.prompt.interaction,
            goal_image=slot.prompt.image, goal_mask=slot.prompt.mask,
            obs=np.stack(slot.obs) if n else np.zeros((0, self.image_size,
                                                       self.image_size, 3), np.uint8),
            actions=np.asarray(slot.actions, dtype=np.int64),
            prev_actions=np.asarray(slot.prev_actions, dtype=np.int64),
            logprobs=np.asarray(slot.logprobs, dtype=np.float32),
            values=np.asarray(slot.values, dtype=np.float32),
            rewards=rewards,
            outcome=slot.outcome,
            terminal=slot.outcome != "truncated",
            bootstrap_value=float(slot.bootstrap_value),
            budget=slot.budget, expert_steps=len(slot.seg.actions), died=slot.died,
            goal_points=slot.goal_points,
            goal_visible=(np.asarray(slot.goal_visible, dtype=bool)
                          if self.collect_goal_points else None),
        )
