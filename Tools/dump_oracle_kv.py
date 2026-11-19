"""The fp32 Python-MLX oracle rollout (pymlx_rollout_fp32.py's exact computation) plus a dump of its KV cache — the
reference for `ff-smoke onestep`, the teacher-forced per-frame gate.

    MLX_ENABLE_TF32=0 python -I dump_oracle_kv.py --spike-dir <dir holding ff_mlx.py> --dit <fp32 dit> \
        --refiner .. --decoder .. --inputs <inputs npz> --frames 14 --out onestep_kv_corgi_mlx0324.safetensors

Writes latents [frames, 16, 60, 104] and k<i>/v<i> [1, 12, frames·1560, 128] for the 30 layers, float32. Each slot is
final once its frame's context write runs, so the end-of-rollout cache is the oracle's state at every frame. ~8 GB for
14 frames: too big for git; it lives beside the other gates (see README). The latents equal the first `frames` frames
of pymlx_rollout_fp32.py's output with the same mlx (checked bit-exact on mlx 0.31.2 and 0.32.4).
"""
import argparse, sys
import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser()
for k in ("spike-dir", "dit", "refiner", "decoder", "inputs", "out"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--frames", type=int, default=14)
a = ap.parse_args()
sys.path.insert(0, a.spike_dir)
import ff_mlx as F

f32 = mx.float32
orig = F.allocate_kv
held = {}
def alloc():
    held["kv"] = orig(f32)
    return held["kv"]
F.allocate_kv = alloc
dit = F.DiT(mx.load(a.dit), dtype=f32)
ref = F.Refiner(mx.load(a.refiner), dtype=f32)
z = np.load(a.inputs)
prompt = mx.array(z["prompt_embeds"].astype(np.float32))[None]
noise = [mx.array(z["noise"][0, f].astype(np.float32))[None] for f in range(F.NUM_FRAMES)]
trans = {(int(k.split("_")[1]), int(k.split("_")[2])): mx.array(z[k].astype(np.float32))
         for k in z.files if k.startswith("eps_")}
F.NUM_FRAMES = a.frames
lat, feats, mods, _ = F.rollout(dit, ref, prompt, noise, trans, log=print)
n = a.frames * F.FRAME_TOKENS
out = {"latents": lat[0].astype(f32)}
for i, (kc, vc) in enumerate(held["kv"]):
    out[f"k{i}"] = kc[:, :, :n, :]
    out[f"v{i}"] = vc[:, :, :n, :]
mx.save_safetensors(a.out, out)
print("wrote", a.out, mx.__version__)
