# Does fine-tuning the static image encoder improve Laser boss success?

## 1. Goal

The matched static and temporal frozen encoders reach only 19% and 17% Laser-start success,
respectively, despite training the L core on Laser-only D10k data. The frozen visual
representation is the remaining untested component in this recipe.

Fine-tune the existing pretrained static encoder with the matched L/D10k/C20k policy while
giving it a lower learning rate. This decides whether later Laser experiments should train
vision end to end or retain precomputed encoder tokens.

## 2. Setup

Use the raw RGB Laser frame release in the datahouse and the same uid-digest holdout as
experiments 0020, 0021, and 0023. Carve validation from the full 10,293-episode frame store
first, then train on the prefix matching the frozen D10k baseline. Preserve the frame/action
causal shift: frame `i` predicts action `i + 1`.

Common recipe: L core with `d_model=640`, 5 layers, 10 heads; 20,000 optimizer cycles;
effective batch 32; AdamW; core LR `1e-4`; encoder LR `1e-5`; 500-step warmup; WSD with
cooldown from 18,000 to 20,000; bf16; dropout 0.2; seed 0; no value or auxiliary head.

| run | encoder | data | cycles | state | dir |
|---|---|---:|---:|---|---|
| `L-D10k-C20k-laser-lr1e-4` | frozen static `f36041bc…1923c` | 9,771 | 20,000 | existing baseline | `runs/laser/L-D10k-C20k-laser-lr1e-4` |
| `L-D10k-C20k-laser-unfrozen` | trainable static, LR `1e-5` | matched D10k prefix | 20,000 | planned | `runs/laser-unfrozen/L-D10k-C20k-laser-unfrozen` |

Run 200–500 smoke-test steps before the full cell and retain checkpoints at 2,000, 5,000,
10,000, and 20,000 cycles. If VRAM requires a smaller physical batch, use gradient
accumulation to keep effective exposure at 32 episodes per optimizer cycle.

## 3. Evaluation metrics

| metric | frozen static baseline | unfrozen candidate | source |
|---|---:|---:|---|
| final Laser success | 19/100, 19% [12.5, 27.8] | pending | evaluation 0029 and matched candidate evaluation |
| final validation CE | 2.1465 | pending | policy metrics / `tools/scaling_report.py` |
| throughput | token-cached baseline | pending | smoke-run `metrics.csv` and wall clock |
| peak GPU memory | token-cached baseline | pending | smoke-run CUDA measurement |

Closed-loop evaluation uses the evaluation 0029 protocol on `policy-final.pt`: Laser start,
100 rollouts, temperature 1.0, seed 0, twice the expert budget, bf16, and batch 8. Report
the Wilson 95% interval, death/timeout counts, boss-seen rate, damage fraction, and entropy.

## 4. Conclusion

_Pending — experiment not yet run._
