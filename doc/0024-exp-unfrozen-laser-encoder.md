# Does fine-tuning the static image encoder improve Laser boss success?

## 1. Goal

The matched static and temporal frozen encoders reach only 19% and 17% Laser-start success,
respectively, despite training the L core on Laser-only D10k data. The frozen visual
representation is the remaining untested component in this recipe.

Remove the uninformative goal image and compare frozen versus fine-tuned static encoders in
matched L/D10k/C20k cells, giving the trainable encoder a lower learning rate. This decides
whether later single-task Laser experiments should train vision end to end.

## 2. Setup

Use the raw RGB Laser frame release in the datahouse and the same uid-digest holdout as
experiments 0020, 0021, and 0023. Carve validation from the full 10,293-episode frame store
first, then train on the prefix matching the frozen D10k baseline. Preserve the frame/action
causal shift: frame `i` predicts action `i + 1`.

Common recipe: L core with `d_model=640`, 5 layers, 10 heads; learned null goal token;
20,000 optimizer cycles; effective batch 32; AdamW; core LR `1e-4`; 500-step warmup; WSD
with cooldown from 18,000 to 20,000; bf16; dropout 0.2; seed 0; no value or auxiliary head.

| run | encoder | data | cycles | state | dir |
|---|---|---:|---:|---|---|
| `L-D10k-C20k-laser-null-goal-frozen` | frozen static `f36041bc…1923c` | matched D10k frame release | 20,000 | planned control | `runs/laser-unfrozen/L-D10k-C20k-laser-null-goal-frozen` |
| `L-D10k-C20k-laser-null-goal-unfrozen` | trainable static, encoder LR `1e-5` | matched D10k frame release | 20,000 | planned candidate | `runs/laser-unfrozen/L-D10k-C20k-laser-null-goal-unfrozen` |

Run 200–500 smoke-test steps before the full cell and retain checkpoints at 2,000, 5,000,
10,000, and 20,000 cycles. If VRAM requires a smaller physical batch, use gradient
accumulation to keep effective exposure at 32 episodes per optimizer cycle.

## 3. Evaluation metrics

| metric | frozen static baseline | unfrozen candidate | source |
|---|---:|---:|---|
| final Laser success | pending | pending | matched goal-free evaluation |
| final validation CE | pending | pending | policy metrics / `tools/scaling_report.py` |
| throughput | token-cached baseline | pending | smoke-run `metrics.csv` and wall clock |
| peak GPU memory | token-cached baseline | pending | smoke-run CUDA measurement |

Closed-loop evaluation uses the evaluation 0029 protocol on `policy-final.pt`: Laser start,
100 rollouts, temperature 1.0, seed 0, twice the expert budget, bf16, and batch 8. Report
the Wilson 95% interval, death/timeout counts, boss-seen rate, damage fraction, and entropy.

## 4. Conclusion

_Pending — experiment not yet run._
