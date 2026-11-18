"""Python-MLX V0 rollout in float32 end to end (DiT, refiner, KV cache) — the oracle for the Swift fp32 rollout lane.

    MLX_ENABLE_TF32=0 python -I pymlx_rollout_fp32.py --spike-dir <dir holding ff_mlx.py> --dit <fp32 dit> \
        --refiner .. --decoder .. --inputs <inputs npz> --frames N --out <x.safetensors>

Both sides in fp32 on the same backend should agree over every frame; a bf16 run can't separate a logic error in
later frames from rounding compounding through the autoregressive sampler.
"""
import argparse
import sys

import mlx.core as mx
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    for k in ("spike-dir", "dit", "refiner", "decoder", "inputs", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="scale the prompt embedding by (1 + eps): a sensitivity probe for the oracle itself")
    a = ap.parse_args()
    sys.path.insert(0, a.spike_dir)
    import ff_mlx as F

    f32 = mx.float32
    orig = F.allocate_kv
    F.allocate_kv = lambda: orig(f32)
    dit = F.DiT(mx.load(a.dit), dtype=f32)
    ref = F.Refiner(mx.load(a.refiner), dtype=f32)
    dec = F.PixelDecoder(mx.load(a.decoder))
    z = np.load(a.inputs)
    prompt = mx.array(z["prompt_embeds"].astype(np.float32))[None] * (1.0 + a.perturb)
    noise = [mx.array(z["noise"][0, f].astype(np.float32))[None] for f in range(F.NUM_FRAMES)]
    trans = {(int(k.split("_")[1]), int(k.split("_")[2])): mx.array(z[k].astype(np.float32))
             for k in z.files if k.startswith("eps_")}
    F.NUM_FRAMES, full = a.frames, F.NUM_FRAMES
    lat, feats, mods, _ = F.rollout(dit, ref, prompt, noise, trans, log=print)
    F.NUM_FRAMES = full
    px = dec(feats, mods)[:, 3:]
    pixels = ((mx.clip(px[0], -1, 1) + 1) * 127.5).round().astype(mx.uint8).transpose(0, 2, 3, 1)
    mx.save_safetensors(a.out, {"latents": lat[0].astype(f32), "pixels": pixels})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
