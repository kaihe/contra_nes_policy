# contra_nes_policy — project instructions

## Cross-repo work
This repo is one of three siblings (`contra_nes_data`, this repo,
`contra_nes_evaluation`). Coordinate across them via **GitHub issues** on the
**target** repo (skill: `contra-nes-handoff` — user skill, available to Claude /
Codex / Grok). Do not leave long handoffs only in chat.

- Need new export fields, tasks, or traces → issue on `kaihe/contra_nes_data`
- Need harness / metric changes → issue on `kaihe/contra_nes_evaluation`
- Shards or contracts ready for training here → expect / create issues labeled
  `handoff` on **this** repo; pickup with `gh issue list -R kaihe/contra_nes_policy --label handoff --state open`
