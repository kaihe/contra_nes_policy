# contra_nes_policy — project instructions

## Chat responses

Do not use LaTeX formulas in chat responses. Express formulas with plain language,
readable pseudocode, or fenced code blocks instead.

## Experiment documents
Every `doc/NNNN-*.md` written from now on follows `.claude/skills/experiment-writeup/`:
one writeup per set of experiments, exactly four sections (goal, setup, evaluation
metrics, conclusion), and **the conclusion is drafted by the user, never by an
assistant** — leave it `_Pending_` and ask. This replaces the generic doc template in
the `contra-nes-workflow` skill for this repo. Docs 0001–0014 predate it and stay as
they are.

Closed-loop evaluation handoffs request **`policy-final.pt` only** by default. Do not ask
the evaluation repo to probe intermediate or pre-cooldown checkpoints unless the user
explicitly requests them.

## Cross-repo work
This repo is one of three siblings (`contra_nes_data`, this repo,
`contra_nes_evaluation`). Coordinate across them via **GitHub issues** on the
**target** repo (skill: `contra-nes-handoff` — user skill, available to Claude /
Codex / Grok). Do not leave long handoffs only in chat.

- Need new export fields, tasks, or traces → issue on `kaihe/contra_nes_data`
- Need harness / metric changes → issue on `kaihe/contra_nes_evaluation`
- Shards or contracts ready for training here → expect / create issues labeled
  `handoff` on **this** repo; pickup with `gh issue list -R kaihe/contra_nes_policy --label handoff --state open`
