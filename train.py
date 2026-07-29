"""Train the Contra cross-view ROCKET-2 policy.

Same shape as ``ROCKET-2/train.py`` — hydra config, a LightningModule, a datamodule,
checkpoint + LR-monitor callbacks — with the minestudio imports replaced by this
repo's own (:mod:`contra_policy.lit`, :mod:`contra_policy.loss`).

    python train.py                          # defaults: config.yaml, fits a 16GB GPU
    python train.py --config-name config_xl  # the paper-scale config
    python train.py batch_size=2 devices=1 model.timesteps=16

Logging defaults to a CSV logger under the hydra run dir; ``logger=wandb`` switches
to Weights & Biases if it is installed.
"""

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from contra_policy.action_space import ACTION_NAMES
from contra_policy.dataset import FAMILIES, ContraDataModule
from contra_policy.lit import ContraRocketLightning
from contra_policy.loss import action_class_weights
from contra_policy.model import CrossViewContraRocket


class WeightsOnlyCheckpoint(ModelCheckpoint):
    """A weights-only snapshot that can coexist with the resumable checkpoint.

    Lightning refuses two ``ModelCheckpoint``s whose ``state_key`` matches, and the
    key is derived from the monitor/interval settings — which are identical here on
    purpose, since both should fire on the same cadence. (ROCKET-2 sidesteps this by
    setting one interval to ``save_freq + 1``, which silently desynchronises the two.)
    """

    @property
    def state_key(self) -> str:
        return "weights_only_snapshot"


def build_logger(args):
    if args.logger == "wandb":
        from lightning.pytorch.loggers import WandbLogger
        return WandbLogger(project=args.project)
    return CSVLogger(save_dir=".", name="logs")


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(args):
    torch.set_float32_matmul_precision("high")   # 4090 tensor cores for the fp32 ops
    # Seeds model init, dataloader shuffling and worker RNGs. Without this, `seed`
    # only reached the train/val split and reruns were not comparable.
    L.seed_everything(args.seed, workers=True)

    policy = CrossViewContraRocket(**OmegaConf.to_container(args.model, resolve=True))
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
    print(f"[train] policy: {trainable/1e6:.1f}M trainable + {frozen/1e6:.1f}M frozen · "
          f"{policy.num_step_tokens} tokens/timestep")

    # The datamodule comes first: the class-weighted BC loss is built from the training
    # action histogram, which the shard index already carries.
    data = ContraDataModule(
        shard_dir=args.shard_dir,
        configs=list(args.configs),
        win_len=args.model.timesteps,
        image_size=args.model.image_size,
        sigma_px=args.sigma_px,
        prev_action_keep_prob=args.prev_action_keep_prob,
        aux_size=args.model.aux_size,
        carry_memory=args.carry_memory,
        family_balance_alpha=args.family_balance_alpha,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    class_weights = action_class_weights(data.action_counts, args.action_loss_alpha)
    if args.action_loss_alpha:
        top = torch.topk(class_weights, 5)
        print("[train] action class weights (top 5): "
              + " ".join(f"{ACTION_NAMES[i]}={w:.2f}"
                         for w, i in zip(top.values.tolist(), top.indices.tolist())))

    lightning_module = ContraRocketLightning(
        policy=policy,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        bc_weight=args.bc_weight,
        heatmap_weight=args.heatmap_weight,
        heatmap_pos_weight=args.heatmap_pos_weight,
        label_smoothing=args.label_smoothing,
        class_weights=class_weights,
        families=FAMILIES,
        carry_memory=args.carry_memory,
        hyperparameters=OmegaConf.to_container(args, resolve=True),
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        # save_last matters: at save_freq=2000 a 7959-step run's last snapshot was
        # step 6000 — 75% of training — and that is the checkpoint the 0728 evaluation
        # scored. `last.ckpt` always holds the final weights.
        WeightsOnlyCheckpoint(dirpath="./weights", filename="weight-{epoch}-{step}",
                              save_top_k=-1, every_n_train_steps=args.save_freq,
                              save_last=True, save_weights_only=True),
        ModelCheckpoint(dirpath="./checkpoints", filename="ckpt-{epoch}-{step}",
                        save_top_k=1, every_n_train_steps=args.save_freq,
                        save_last=True, save_weights_only=False),
    ]

    L.Trainer(
        logger=build_logger(args),
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
    ).fit(
        model=lightning_module,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
        ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    main()
