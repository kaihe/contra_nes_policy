# contra_nes_policy

[ROCKET-2](https://arxiv.org/abs/2503.02505)'s cross-view goal-aligned visuomotor
policy, re-implemented for Contra (NES) on the task shards produced by
[`contra_nes_data`](../contra_nes_data).

The reference repo lives at `study/0714-rocket/resources/ROCKET-2/`. This repo keeps
its structure file-for-file (`model.py` / `cross_view_dataset.py` / `train.py`) and
replaces the parts that are Minecraft- or MineStudio-specific.

## What the policy does

A goal is given as a **cross-view prompt**: a reference frame with the target entity
marked by a coloured Gaussian blob, plus an id for the kind of interaction intended.
The policy watches the agent's own view, aligns it against that prompt, and emits
NES buttons. Five interactions come out of the shards: `kill`, `pick`, `avoid`,
`traverse`, `boss`.

Per timestep the model lays out `[view tokens…, interaction, prev_action]` and runs a
VPT transformer-XL over the flattened sequence with carried memory. The policy and
value heads read the last token; an auxiliary head reads the interaction token and
predicts **where the goal entity is in the current frame** (`exist` / `point` /
`bbox`). That aux head is what forces the prompt to actually be used — without it the
BC loss is happily minimised by replaying the modal trajectory, which for Contra is
"hold right and fire".

## Layout

| file | role |
|---|---|
| `src/contra_policy/model.py` | `CrossViewContraRocket` — the policy |
| `src/contra_policy/dataset.py` | tar-offset index + windowed map-style `Dataset` |
| `src/contra_policy/loss.py` | behaviour cloning + the cross-view grounding aux head |
| `src/contra_policy/lit.py` | `LightningModule` (replaces `MineLightning`) |
| `src/contra_policy/encoder.py` | the frozen Contra vision backbone |
| `src/contra_policy/action_space.py` | the 21 discrete NES actions |
| `src/contra_policy/goal.py` | PPU→screen geometry, goal masks, aux targets |
| `src/contra_policy/vpt/` | vendored VPT transformer-XL core |

## How it differs from ROCKET-2

**Vision backbone.** ROCKET-2 freezes a DINO ViT-B/16 pretrained on photographs. That
prior is wrong for 8-bit NES frames whose task-critical entities are 2–4px sprites, so
the view backbone is instead the Contra autoencoder already pretrained in
`contra_agent/dreamer/train_ae.py` — a DreamerV3-style ConvEncoder trained with a
four-class entity-occupancy aux loss precisely so small sprites survive into the
embedding. It is loaded from `ae_pretrained.pt` and frozen exactly as upstream freezes
its ViT. The one adaptation is reading the conv trunk's `(B, 1024, 4, 4)` feature map
as a 16-token grid instead of ViT patch tokens.

**Action head.** ROCKET-2 inherits MineStudio's hierarchical head (button groups +
camera bins). All 777k recorded Contra steps fall inside the 21-action set from
`contra_nes_data/src/agent/baseline.yaml` (verified by scanning every shard: zero
out-of-vocabulary vectors), so `pi_head` is a single 21-way categorical.

**Goal mask.** Upstream ships a segmentation mask. Contra's shards carry goal
*points*, so the mask is regenerated at load time from `goal_points` — which keeps
`sigma` a training knob and hands the mask backbone a clean single-channel input
rather than asking it to recover the blob from the tinted RGB. `bbox` targets, which
upstream derives from mask extent, are the union of sprite-sized boxes around each
centroid (for the boss this spans all live components, which is the useful signal).

**Cross view is per-episode.** ROCKET-2 *samples* a cross-view frame per window,
because a Minecraft episode contains many interactions. A Contra task episode has
exactly one goal, so the dataset emits it once per window with no time axis and the
model expands the encoded tokens. This is asserted equivalent to the per-timestep form
(`test_per_window_goal_matches_per_timestep_goal`) and roughly halves backbone compute.
The model still accepts upstream's per-timestep shape.

**Missing upstream pieces.** `loss.py` is imported by ROCKET-2's `train.py` but absent
from the open-source drop, and `minestudio.offline` is not installed; both are written
here against the same weights (`point 0.1 / bbox 0.1 / exist 0.01`).

## Setup

```sh
pip install -e .
```

Expects the shards at `~/code/contra_nes_data/game_trace/hf/{kill,item,traverse,boss}-00000.tar`
and the pretrained encoder at `~/code/contra_agent/tmp/dreamer/ae_pretrained.pt`
(both paths are config keys). Without the encoder checkpoint the view backbone trains
from scratch.

## Train

```sh
python train.py                             # defaults, sized for a 16GB GPU
python train.py --config-name config_xl     # paper-scale: hiddim 1024, 128-step windows
python train.py batch_size=8 model.timesteps=16
```

The first run builds a shard index into `cache/` (~2s, cached by shard size+mtime).
Runs land in `runs/<date>/<time>/` with `weights/`, `checkpoints/` and CSV metrics.

Measured on a 16GB laptop RTX 4090 at the defaults (`batch_size=16`, `timesteps=32`):
**5.4 GB VRAM, 0.38 s/step, ~1.7 GB host RAM in flight** — GPU-bound, with the loader
delivering 72 windows/s against a demand of 42.

### Watch the host RAM, not the VRAM

This is the binding constraint and the easy way to take the machine down. The loader
keeps `num_workers × prefetch_factor × batch_size` windows resident and `pin_memory`
doubles it; at `timesteps=128` a window is 25 MB, so a careless `num_workers=12` is
tens of GB. The datamodule prints its estimate at startup — check that line before
raising `num_workers`:

```
[dataset] 6.6 MB/window → ~1.7 GB host RAM in flight
```

The validation loader deliberately uses fewer workers and `persistent_workers=False`:
it runs ~50 batches every 1000 steps, so persistent val workers would idle >95% of the
time while holding prefetch buffers, doubling the loader's resident set and evicting
the shards from page cache that the training loader depends on.

### Windowed decoding

`export_hf.py` writes the observation video all-intra (PNG-in-MKV), so every frame is
a keyframe and a seek lands exactly. The dataset relies on that to decode *only* the
32 frames a window needs. Decoding whole episodes instead costs 4.8x more frames per
epoch than the epoch uses (7.5x on `boss`, 5.0x on `traverse`) — measured at 2.8x
slower per window, which is enough to starve the GPU. If you ever re-export with a
non-intra codec, `_frames` falls back to a full decode automatically and this gets
slow again.

## Metrics to watch

`bc_acc` is a weak signal — 72% of all recorded steps are the single action `R`, so it
reaches ~0.8 almost immediately from the action prior alone. **`point_err_px`** (mean
error of the predicted goal location, in 240×224 screen pixels) is the number that
says whether cross-view grounding is working. A 60-step smoke run moves it 80 → 37 px.

## Test

```sh
python -m pytest tests/ -q
```

21 tests over the real shards, skipping cleanly if the data is absent. They target the
silent failure modes rather than crashes: unshifted PPU coordinates, aux targets
regressed on frames where the goal is off-screen (59% of `traverse` frames), padding
leaking into a loss mean, and the per-window/per-timestep goal equivalence.
