#!/usr/bin/env python
"""Build a frozen-encoder token cache for a scaling-release prefix.

    python tools/build_token_cache.py \
        --manifest ~/code/contra_nes_data/game_trace/releases/boss-spread-20k-v1/manifest.json \
        --shard-count 27 \
        --validation-sha256 84cc5c50…6f9bfc \
        --encoder runs/encoder/2026-07-31/18-00-11/checkpoints/encoder-final.pt \
        --out cache/tokens/spread20k-d27

**Pass the release's longest prefix.** Prefixes are nested, so that one cache serves every
smaller data cell — training selects a cell by uid, not by cache. `--split val` caches the
release's held-out shard instead. See `doc/0015-exp-scaling-data.md` §2.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from contra_encoder.net import load_pretrained_encoder                  # noqa: E402
from contra_policy.dataset import load_or_build_index, scaling_release  # noqa: E402
from contra_policy.token_cache import build_token_cache                 # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="release manifest.json")
    ap.add_argument("--shard-count", type=int, required=True,
                    help="which nested train prefix; normally the release's longest "
                         "(10k: 13, 20k: 27) so one cache serves every data cell")
    ap.add_argument("--encoder", required=True, help="stage-A encoder checkpoint")
    ap.add_argument("--out", required=True, help="cache directory to create")
    ap.add_argument("--split", choices=("train", "val"), default="train")
    # Not defaulted: `scaling_release` verifies the held-out shard's bytes against this,
    # and the whole point is that the contract comes from the caller, not the file being
    # checked. boss-spread-10k-v1 is 29fd4017…cc9ae0; mixed-v2 is 131835e3…52ecad.
    ap.add_argument("--validation-sha256", required=True,
                    help="expected SHA-256 of the release's validation shard")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=256,
                    help="frames per encoder forward; bounds the activation peak")
    ap.add_argument("--cache-dir", default="cache", help="where the shard index is cached")
    args = ap.parse_args()

    release = scaling_release(args.manifest, args.shard_count, args.validation_sha256)
    shards = release["val"] if args.split == "val" else release["train"]
    index = load_or_build_index(shards, cache_dir=args.cache_dir)
    print(f"[build] {args.split} D{args.shard_count}: {len(shards)} shard(s), "
          f"{len(index)} episodes, {sum(e['length'] for e in index)} frames")

    encoder = load_pretrained_encoder(args.encoder, freeze=True)
    t0 = time.time()
    build_token_cache(index, encoder, args.out, encoder_ckpt=args.encoder,
                      image_size=args.image_size, device=args.device, chunk=args.chunk)
    print(f"[build] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
