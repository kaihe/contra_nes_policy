# A goal-agnostic encoder: one function for every image

Status: Proposed
Supersedes: 0001 §2 (the design). 0001's measurements, rejected alternatives and the
`point_err_px` finding all stand.

**Question.** 0001's encoder is goal-*conditioned*: the frame encode is modulated by a
goal token via FiLM so it can answer "where is the goal in this frame". That requires a
goal-only conv trunk, a goal projection path and a conditioning layer. Is any of it
necessary?

**Answer.** No. The encoder becomes a single symmetric function applied to any image —
agent frame or goal frame alike:

```python
encode(image) -> token, (entity_heatmap, [reconstruction])
```

Goal matching moves to the policy's temporal transformer, where a frame token attends
against a goal token. That is a stronger mechanism than FiLM broadcasting 256 channel
scales, and it is where `model.py` already computes the goal heatmap today.

---

## 1. Why

**The goal image is self-describing.** `goal.png` is "reference image + coloured blob"
and the blob is drawn into the RGB — sampled at the goal points, pixels read
**(225, 110, 18)** against an image mean of (56, 70, 14). So the goal frame already
carries "which entity is the target" in its pixels. It needs no mask channel, no mask
trunk, and no special path: it is an image with the answer painted on.

**Goal conditioning was the weaker half of the design.** FiLM produces one scale and
one shift per channel, broadcast over the whole spatial grid — it can say "attend to
turret-ish features", never "look *there*". Attention between a frame token and a goal
token can express the comparison directly. 0001 put the weaker mechanism in the
encoder while the stronger one already existed downstream.

**It deletes the goal-specific machinery:**

| removed | params |
|---|---|
| `mask_backbone` — the blob's own conv trunk | 0.70M |
| `goal_reduce` + `goal_proj` — a second projection path | 2.69M |
| `film` — goal → frame conditioning | 0.26M |
| **total** | **3.65M** |

## 2. The design

One trunk, one projection, applied to whatever image it is given:

```python
encode(image)            -> token (512)
entity_head(token)       -> (4, 32, 32)   player / player_bullets / enemies / enemy_bullets
recon_head(token)        -> (3, 256, 256) optional, see §4
```

The policy then builds `[interaction, goal_token, img_token × N]`, and the goal heatmap
head sits on the **temporal** output — where it is in `model.py` today
(`aux_vis_head(z[:, :, index_bias, :])`), not in the encoder.

Both heads still decode **from the token**, not from the conv map. That is what forces
spatial structure through the 512-d bottleneck, and 0001's test asserting no gradient
path around it carries over unchanged.

## 3. What moves to the policy

| responsibility | 0001 | 0002 |
|---|---|---|
| entity occupancy | encoder | encoder |
| reconstruction | — | encoder (optional) |
| "where is the goal in this frame" | encoder, via FiLM | **policy, via attention** |
| `point_err_px` / `exist` readout | encoder | **policy** |

The consequence to be deliberate about: **stage A loses its direct measurement.**
0001's gate — `peak_hit` 0.999 and `pck16` 1.000 on boss — was direct evidence that one
token holds goal-relevant structure. Under 0002 that is unmeasurable until stage B,
where a failure would be entangled with the retokenisation, the BC retrain and the
unfrozen trunk at once.

Mitigation: entity dice is still a per-frame, per-class measurement of exactly the
spatial content the goal head was reading. If `enemies` and `enemy_bullets` stay near
0.96 / 0.91, the token has not lost the sprites — only the goal-specific readout.

## 4. Open question: is reconstruction worth 28M parameters?

`ConvDecoder` at this config is **27.97M** — larger than the whole current encoder
(20.3M), roughly tripling the model.

The precedent is thinner than it appears. `contra_agent/dreamer/train_ae.py` trains
recon + entity together, but its docstring's claim is that a **recon-*only*** encoder
"goes entity-blind" — an argument for *adding* the entity head, not evidence that
reconstruction is needed once you have one. Its decoder also had a second job there
(the world model needs to render), which does not apply here.

So: **implement behind a flag, default off, and settle it by ablation** — entity-only
vs entity+recon, compared on entity dice and on stage-B completion. Without that,
28M parameters go to rendering background tiles on faith.

## 5. Risks, and the metric that gates each

| risk | why plausible | gate |
|---|---|---|
| a goal-agnostic token loses goal-relevant structure | the goal head is no longer asking for it | entity dice ≥ 0.95 `enemies`, ≥ 0.90 `enemy_bullets` (0001 baseline) |
| attention is a worse goal matcher than FiLM in practice | untested; the argument is theoretical | stage-B completion vs the 72.8% BC baseline |
| reconstruction is wasted capacity | 28M params, weak precedent | ablation in §4 — no dice improvement ⇒ drop it |
| stage A can no longer detect a broken token | the direct gate is gone | entity dice is the proxy; accept reduced diagnostic power |

## 6. Sequencing

1. **Strip the goal path** from `contra_encoder`: delete `mask_backbone`,
   `goal_reduce`, `goal_proj`, `film`; `encode()` takes one image.
2. **Retrain stage A** on entity occupancy alone. Gate: entity dice vs 0001's numbers.
3. **Reconstruction ablation** (§4), flag-gated.
4. **Stage B** — retokenise `contra_policy.model` to `[interact, goal, img × N]`, delete
   the `prev_action` machinery, restore the goal heatmap head on the temporal output.
   This is now where goal grounding is measured for the first time under this design.

Step 2 discards the 0001 checkpoint: the head structure and the goal path both change,
so nothing transfers.

---

## Appendix — provenance

| claim | source |
|---|---|
| blob is drawn into `goal.png` | pixel sample at `ppu_to_norm(goal_points)`, RGB (225,110,18) vs image mean (56,70,14) |
| removable parameter counts | `sum(p.numel())` per submodule on `EncoderConfig(entity_classes=4)` |
| `ConvDecoder` 27.97M | `dreamer.models.ConvDecoder(256, depth=32, feat_dim=1024)` |
| entity dice baseline | 0001, `encoder-final.pt` scored on 200 val batches |
| recon-only "entity-blind" | `contra_agent/dreamer/train_ae.py` module docstring |
