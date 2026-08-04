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
from contra_policy.rl.buffer import Episode

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


class TokenHistoryActor:
    """Steps :class:`~contra_policy.model.ContraPolicy` one decision at a time.

    The policy has no recurrent state, so "memory" is an explicit per-slot list of frame
    tokens: encode the new frame once, append, and re-run the causal core over
    ``[interaction, goal, frames…]``. Frame tokens are cached — the encoder never re-runs
    on history — but the core does see the whole prefix every step, which is exactly the
    conditioning it was trained under.

    Recycling slot *i* means **clearing its history**, or the new episode attends to the
    previous one's frames. That is the same hazard the recurrent version had, in a form
    that is at least visible: a list you can see the length of.

    Mirrors ``contra_eval.policies.CheckpointPolicy`` deliberately. Training-time and
    evaluation-time stepping must agree, and two independent implementations of "run the
    core over a ragged batch of histories" would be two chances to disagree.
    """

    def __init__(self, model, batch_size: int, *, device: torch.device,
                 temperature: float = 1.0, seed: int = 0, precision: str = "bf16",
                 attention_layout: str = "padded"):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.temperature = float(temperature)
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                               "fp32": None}[precision]
        if device.type != "cuda":
            self.autocast_dtype = None
        self.generator = torch.Generator(device=device).manual_seed(int(seed))
        self.d = model.encoder.cfg.hiddim
        self.prefix = 2
        self.attention_layout = attention_layout
        self.frames: List[Optional[torch.Tensor]] = [None] * batch_size
        self.goal: List[Optional[torch.Tensor]] = [None] * batch_size
        self.inter: List[Optional[torch.Tensor]] = [None] * batch_size

    def reset(self, slots: Sequence[int]) -> None:
        for i in slots:
            self.frames[i] = self.goal[i] = self.inter[i] = None

    def reset_all(self) -> None:
        self.reset(range(self.batch_size))

    @torch.no_grad()
    def begin(self, slot: int, goal_image: np.ndarray, interaction: int) -> None:
        """Encode the episode-constant prefix once, at episode start."""
        ctx = (torch.autocast("cuda", dtype=self.autocast_dtype)
               if self.autocast_dtype is not None else _null_context())
        g = torch.from_numpy(goal_image).to(self.device).unsqueeze(0)
        with ctx:
            self.goal[slot] = self.model.encoder.encode(g)                      # (1, d)
            self.inter[slot] = self.model.interaction(
                torch.tensor([interaction + 1], device=self.device))            # (1, d)
        self.frames[slot] = None

    @torch.no_grad()
    def act(self, obs: "RolloutObservation") -> Dict[str, np.ndarray]:
        """One decision per active slot. Returns ``action`` and its ``logprob``.

        No value: GRPO has no critic. The advantage arrives later, from the group.
        """
        ctx = (torch.autocast("cuda", dtype=self.autocast_dtype)
               if self.autocast_dtype is not None else _null_context())
        active = [i for i, im in enumerate(obs.image) if im is not None]
        with ctx:
            imgs = torch.from_numpy(np.stack([obs.image[i] for i in active])).to(self.device)
            new = self.model.encoder.encode(imgs)                               # (n, d)
            for k, i in enumerate(active):
                tok = new[k:k + 1]
                self.frames[i] = (tok if self.frames[i] is None
                                  else torch.cat([self.frames[i], tok], 0))
            logits = self._core_over_histories(active)

        logits = logits.float()
        if self.temperature <= 0:
            action = logits.argmax(-1)
            logp = torch.log_softmax(logits, -1).gather(-1, action[:, None]).squeeze(-1)
        else:
            scaled = logits / self.temperature
            probs = torch.softmax(scaled, -1)
            action = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
            # The log-prob of the *sampling* distribution: that is the behaviour density
            # a policy-gradient ratio must be taken against.
            logp = torch.log_softmax(scaled, -1).gather(-1, action[:, None]).squeeze(-1)

        out_a = np.zeros(self.batch_size, dtype=np.int64)
        out_l = np.zeros(self.batch_size, dtype=np.float32)
        out_a[active] = action.cpu().numpy()
        out_l[active] = logp.cpu().numpy()
        return {"action": out_a, "logprob": out_l}

    def _core_over_histories(self, active: Sequence[int]) -> torch.Tensor:
        """Left-align each slot's history into a padded batch and read its last step."""
        if self.attention_layout == "varlen":
            segments = [torch.cat([self.inter[i], self.goal[i], self.frames[i]], dim=0)
                        for i in active]
            lengths = torch.tensor([x.shape[0] for x in segments], device=self.device,
                                   dtype=torch.int32)
            if int(lengths.max()) > self.model.context:
                raise RuntimeError("rollout history exceeds policy context")
            cu = torch.cat([torch.zeros(1, device=self.device, dtype=torch.int32),
                            lengths.cumsum(0).to(torch.int32)])
            h = self.model._run_core_varlen(
                torch.cat(segments, dim=0), cu, int(lengths.max()))
            return self.model.pi_head(h[cu[1:].long() - 1])

        lens = [int(self.frames[i].shape[0]) for i in active]
        max_t = max(lens)
        L = self.prefix + max_t
        if L > self.model.context:
            raise RuntimeError(
                f"episode reached {max_t} frames + {self.prefix} prefix = {L}, over "
                f"context {self.model.context}. Raise policy.core.context and retrain; "
                f"task budgets reach 1038.")
        b = len(active)
        x = torch.zeros(b, L, self.d, device=self.device, dtype=self.frames[active[0]].dtype)
        attn = torch.zeros(b, 1, L, L, dtype=torch.bool, device=self.device)
        last = torch.zeros(b, dtype=torch.long, device=self.device)
        idx = torch.arange(L, device=self.device)
        for k, i in enumerate(active):
            t = lens[k]
            x[k, 0:1] = self.inter[i]
            x[k, 1:2] = self.goal[i]
            x[k, self.prefix:self.prefix + t] = self.frames[i]
            n = self.prefix + t
            attn[k, 0, :n, :n] = idx[:n, None] >= idx[None, :n]
            last[k] = n - 1
        # Route through the policy execution wrapper so rollout and optimiser updates
        # test the same compiled core. The raw registered core still owns checkpoints.
        h = self.model._run_core(x, attn_mask=attn)
        h_last = h.gather(1, last.view(b, 1, 1).expand(b, 1, self.d)).squeeze(1)
        return self.model.pi_head(h_last)

    def state(self) -> dict:
        return {"generator": self.generator.get_state()}

    def load_state(self, state: dict) -> None:
        self.generator.set_state(state["generator"])


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
    #: Boss HP for the graded failure reward (doc/0005 §2). ``-1`` means the task's
    #: maker exposes no progress signal.
    #:
    #: The anchor is ``hp_ref``, the task's ``boss_hp_start`` metadata — a **task-level
    #: constant**, not the value at step 0 and not this episode's peak.
    #:
    #: A boss task begins at the reveal, and the boss spawns in stages: the data repo
    #: measured ``16 -> 48 -> ~64`` (`contra_nes_data/doc/0001-boss-search-curriculum.md`).
    #: So HP reads 0 initially, then climbs. Anchoring on step 0 scores every episode as
    #: having dealt no damage; anchoring on the *episode's own* peak is worse than that
    #: — a rollout that dies mid-spawn normalises 8 damage against 16 rather than 63 and
    #: scores 0.5 where a rollout that survived to full reveal and did the same damage
    #: scores 0.13. That rewards dying early, which is precisely the failure mode
    #: (deaths at 27% of the expert's episode).
    hp_ref: int = -1
    hp_peak: int = -1
    hp_last: int = -1


# ── the collector ─────────────────────────────────────────────────────────────
    #: Which group this rollout belongs to — set by `collect_groups`, and used by
    #: the buffer to baseline it against its siblings. Nothing else reads it.
    group_id: int = 0


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
                 collect_goal_points: bool = True, owner: str = "EpisodeCollector",
                 attention_layout: str = "padded"):
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
                       "truncated": 0.0, "progress_coef": 0.0, **(reward or {})}
        self.actor = TokenHistoryActor(model, batch_size, device=device,
                                  temperature=temperature, seed=seed, precision=precision,
                                  attention_layout=attention_layout)
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

    def collect_groups(self, groups: Sequence[Sequence["RLTask"]],
                       base_gid: int = 0) -> List[Episode]:
        """Roll out every task in every group, and return the finished episodes.

        The unit is the *group*, not a step budget: GRPO needs all G rollouts of a task
        so their returns can baseline each other. A partially-collected group is useless,
        so this drains rather than stopping on a step count.

        Slots are filled from a flat queue across groups, so a group's members run
        concurrently in different slots rather than serially — each still restores the
        same savestate, and the emulator is stepped per slot anyway.

        ``base_gid`` offsets the group ids this call hands out. A caller that invokes
        this repeatedly and pools the results — :meth:`GRPOTrainer.collect_filtered`
        does — **must** advance it, or ids collide across calls and episodes of
        different tasks end up sharing a group. That silently destroys GRPO's premise:
        the baseline stops being same-task. Pinned by
        ``tests/test_rollout_groups.py::test_group_ids_are_unique_across_calls``.
        """
        self.open()
        queue = [(gid, t) for gid, g in enumerate(groups, start=base_gid) for t in g]
        slots: List[Optional[_Slot]] = [None] * self.batch_size
        out: List[Episode] = []

        while True:
            fresh = []
            for i in range(self.batch_size):
                if slots[i] is None and queue:
                    gid, task = queue.pop(0)
                    slots[i] = self._start(task)
                    slots[i].group_id = gid
                    self.actor.begin(i, slots[i].prompt.image, slots[i].prompt.interaction)
                    fresh.append(i)
            if fresh:
                self.actor.reset(fresh)
                for i in fresh:
                    self.actor.begin(i, slots[i].prompt.image,
                                     slots[i].prompt.interaction)
            if all(s is None for s in slots):
                break

            obs = self._observe(slots)
            step = self.actor.act(obs)
            for i, slot in enumerate(slots):
                if slot is None:
                    continue
                self._record(slot, int(step["action"][i]), float(step["logprob"][i]))
                self._step(slot, int(step["action"][i]))
                if slot.done:
                    out.append(self._finish(slot))
                    slots[i] = None
        return out

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
        slot.hp_peak = slot.hp_last = self._progress(slot, slot.prev_ram)
        if slot.hp_peak >= 0:
            # `boss_hp_start` is the data repo's own anchor: the maximum boss HP over
            # the source trace, i.e. the value at full reveal. Shared by every rollout
            # of the task, so members of a group are always compared on one scale.
            meta = getattr(seg, "meta", None) or {}
            slot.hp_ref = int(meta.get("boss_hp_start", -1) or -1)
        slot.frame = self._peek(slot)
        return slot

    @staticmethod
    def _progress(slot: _Slot, ram) -> int:
        """Interpreted progress for this task's family, or -1 if it has none.

        Read through the maker rather than from RAM: `KillBossMaker.boss_hp` is the data
        repo's published accessor, so no ``ADDR_*`` knowledge enters this repo
        (`kaihe/contra_nes_data#2`). Families whose maker exposes nothing keep the
        binary reward untouched.
        """
        fn = getattr(slot.maker, "boss_hp", None)
        return int(fn(ram)) if fn is not None else -1

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

    def _record(self, slot: _Slot, action: int, logprob: float) -> None:
        slot.obs.append(slot.frame)
        slot.actions.append(action)
        slot.prev_actions.append(slot.prev_action)
        slot.logprobs.append(logprob)
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
        if slot.hp_peak >= 0:
            # Every step, not just the last: `prev_ram` is deliberately not advanced on
            # the step that ends the episode, so reading it at `_finish` would miss the
            # damage dealt by the final action — and the peak would be missed entirely.
            slot.hp_last = self._progress(slot, cur)
            slot.hp_peak = max(slot.hp_peak, slot.hp_last)

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

    def _reward_for(self, slot: _Slot) -> float:
        """Terminal reward: ``reward[outcome]``, plus graded credit on failure.

        **Still one scalar per episode.** ``progress_coef`` grades *failures* by the
        fraction of boss HP removed, so a group whose members all fail — 43-75% of boss
        groups at G=8 — has real advantage spread instead of eight identical zeros
        (doc/0005 §2). It adds no per-step credit: the buffer still broadcasts one
        number across the episode.

        **Damage is measured against the task's ``boss_hp_start``**, the data repo's
        full-reveal anchor, falling back to this episode's peak only when the metadata
        is absent. The boss spawns in stages (16 -> 48 -> ~64), so neither step 0 nor
        the episode's own peak is a sound denominator — see `_Slot.hp_ref`. An episode
        that dies before the boss spawns at all has no damage and scores zero, which is
        the right answer: it made no progress.

        **The ranges must stay disjoint.** At ``progress_coef <= 0.5`` a failure scores
        at most 0.5 against a success's 1.0, so no amount of damage can outrank a win
        and the objective is still "kill the boss". Pinned by
        ``tests/test_reward.py::test_no_failure_can_outscore_a_success``.

        Deliberately *not* a symmetric step penalty. That would score a fast death above
        a long survival, and the boss failure mode is dying early — at 27% of the
        expert's episode, measured over 69 deaths. See doc/0005 §8.
        """
        base = float(self.reward.get(slot.outcome, 0.0))
        alpha = float(self.reward.get("progress_coef", 0.0))
        if slot.outcome == "success" or alpha <= 0:
            return base
        ref = slot.hp_ref if slot.hp_ref > 0 else slot.hp_peak
        if ref <= 0:
            return base
        removed = min(ref, max(0, ref - max(0, slot.hp_last)))
        return base + alpha * (removed / ref)

    def _finish(self, slot: _Slot) -> Episode:
        """A completed rollout, as the buffer wants it."""
        n = len(slot.actions)
        return Episode(
            task_uid=slot.task.uid,
            family=slot.task.family,
            group_id=slot.group_id,
            frames=(np.stack(slot.obs) if n else
                    np.zeros((0, self.image_size, self.image_size, 3), np.uint8)),
            goal_image=slot.prompt.image,
            interaction=slot.prompt.interaction,
            actions=np.asarray(slot.actions, dtype=np.int64),
            logprobs=np.asarray(slot.logprobs, dtype=np.float32),
            reward=self._reward_for(slot),
            outcome=slot.outcome,
            task_label=slot.task.label,
        )
