---
name: experiment-writeup
description: >
  How to write an experiment document in contra_nes_policy: one writeup per set of
  experiments, exactly four sections (goal, setup, evaluation metrics, conclusion), and
  the conclusion is drafted by the user, never by the assistant. Use when creating or
  updating any doc/NNNN-*.md in this repo, when a run finishes and its numbers need
  recording, or when asked to "write it up". Overrides the generic doc template in
  contra-nes-workflow for this repo.
---

# Experiment writeups — `contra_nes_policy`

One document per **set of experiments**. Not per run, not per revision, not per idea.

A "set" is the group of runs that answer one question together — a size ladder, a data
axis, a sweep. If two groups answer two questions, they are two documents, even when they
share a config. If one group needs a follow-up run to finish its answer, that run belongs
in the *same* document.

**Exactly four sections. Nothing else.**

```markdown
# <the question this set of experiments asks>

## 1. Goal

## 2. Setup

## 3. Evaluation metrics

## 4. Conclusion
```

No status header, no "what was rejected", no risks table, no predictions section, no
provenance appendix, no follow-up section. Those belong in the generic template in
`contra-nes-workflow`, which **does not apply to this repo's experiment docs**. If a
rationale is worth keeping and does not fit these four sections, it goes in a code comment,
a docstring, or a commit message — not into a fifth heading.

---

## 1. Goal

What question this set of experiments answers, and why it is being asked now. Name the
decision that will be made differently depending on the answer.

**Hard limit: two short paragraphs, three or four lines each.** The first states the
problem; the second states what this set of experiments is trying to find, and the decision
that turns on it. Supporting evidence appears as a clause with a number, never as its own
paragraph — everything else is already written in the document it came from, so link to it
instead of summarising it. If the section needs a third paragraph, the extra material
belongs in §2. Docs 0013–0017 predate this limit and are not retrofitted.

If the goal changes mid-experiment, rewrite this section — do not append. A goal that
quietly grew to fit the results is the failure this section exists to prevent.

## 2. Setup

What was run. **List every run, even when there are many** — a reader must be able to tell
which numbers came from which run without opening `runs/`.

Use a table, one row per run, with the columns that actually differ between rows plus the
run directory. Put what is *common* to all runs in a line above the table rather than
repeating it in every row:

```markdown
Common to all cells: boss-only, batch 16, 20,000 steps, AdamW, WSD, bf16, dropout 0.2,
frozen stage-A encoder, `config_bc_scaling.yaml`, seed 0.

| run | d_model | n_layer | params | steps | dir |
|---|---:|---:|---:|---:|---|
| M | 512 | 4 | 12.86M | 20,000 | `runs/scaling/m-d13-s0` |
| L | 640 | 5 | 25.13M | 20,000 | `runs/scaling/l-d13-s0` |
```

Also state the data and the checkpoint policy: which release and prefix, which validation
artifact, and at which steps checkpoints were kept. If a planned run did not happen, say so
in a row rather than dropping it — a missing cell is information.

### Run names

**`<model>-D<data>-C<cycles>`** — the three axes every experiment here varies, in that
order, so a run directory says what it is without opening a config:

| run | means |
|---|---|
| `M-D20k-C40k` | M core, the 20k release at its full prefix, 40,000 training cycles |
| `M-D20k.e8-C40k` | the same, on the 8-shard prefix of that release |
| `XXL-D10k-C20k` | XXL core, the 10k release D13 prefix, 20,000 cycles |

The data token names the **release**, because two releases can hold the same number of
episodes and still not be the same data — `boss-spread-20k-v1` shares no episode and no
validation shard with `boss-spread-10k-v1`. A sub-prefix of a release gets `.eN`, where N
is the shard count the manifest uses; a run at the full prefix omits it.

Anything that is currently the default stays out of the name — seed 0, WSD schedule. A run
that departs from a default appends it: `-s1` for a second seed, `-cos` for the retired
cosine schedule. Legacy directories predating this scheme are not renamed when another repo
already cites their paths; give the new-style name in the table and the real path in the
`dir` column.

Reporting tools must key off `resolved_config.yaml`, never off the directory name — a name
is a label somebody typed, and section 3 is read as evidence.

## 3. Evaluation metrics

The numbers, and where each came from. Tables only; no interpretation here — interpretation
is section 4, and section 4 is not yours to write.

For every metric, say the command or run directory that produced it, inline. A number
nobody can recheck six weeks later is not a measurement:

```markdown
| cell | train CE | val CE (final) | source |
|---|---:|---:|---|
| M | 0.1819 | 1.3627 | `tools/scaling_report.py runs/scaling` |
```

Closed-loop success rates come from `contra_nes_evaluation` — cite its doc and run
directory, and carry its label (train-state probe vs held-out) unchanged. Include the
noise floor (n, and the interval) wherever a rate is quoted; two rates without it invite a
comparison the data cannot support.

## 4. Conclusion — **the user writes this**

**Hard rule: the assistant never drafts this section.** Not a first pass, not a "suggested
wording", not a bulleted summary in the doc for the user to edit. This is the section where
the experiment's meaning is decided, and that decision is the user's.

The sequence:

1. Collect the metrics into section 3.
2. Leave section 4 holding exactly one line:
   `_Pending — metrics collected, awaiting discussion._`
   (Before metrics exist, the line is `_Pending — experiment not yet run._`)
3. **Tell the user the metrics are in and ask for the conclusion.** In chat, the assistant
   may lay out what it observes in the numbers — that is what the discussion is for.
   Observations in chat are fine; observations written into section 4 are not.
4. The user drafts the conclusion.
5. The assistant transcribes it into section 4. It may fix grammar, formatting and
   markdown. It may not add a claim, soften one, extend one to a case the user did not
   mention, or append its own reasoning underneath.

If the user's draft conflicts with a number in section 3, say so once, plainly, and let the
user decide. Do not silently reconcile the two.

A document whose section 4 still says `_Pending_` is a normal, correct state. Leaving it
pending is always better than filling it in.

---

## Housekeeping outside the four sections

Two repo mechanics still apply, because they live outside the document body:

- **Filename**: `doc/NNNN-topic.md`, sequential, never dated (`contra-nes-workflow`).
- **Index**: one line in `doc/README.md`, with the conclusion state in the status column —
  `Pending` until section 4 is written, then a few words of the user's conclusion.

Everything else in `contra-nes-workflow` — branches, commit messages, PRs, cross-repo
handoff issues, where a fact belongs (docstring vs comment vs commit vs doc) — is unchanged
and still applies here.
