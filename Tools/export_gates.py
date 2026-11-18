"""Convert the V0 spike's numpy artefacts into safetensors the Swift gates read.

    python -I export_gates.py --spike /Volumes/DEV_ARCHIVE/fastframes-s-measure --out <gates dir>

Writes (all float32 unless noted):
  inputs_<name>.safetensors   prompt_embeds [512, 4096], noise [21, 16, 60, 104], eps_<frame>_<step> [16, 60, 104]
  lanes_fp32attn.safetensors  upstream fp32 component outputs (torch side of spike/lanes.py, attention kept fp32)
  pymlx_bf16_corgi.safetensors  Python-MLX bf16 full rollout: latents [21, 16, 60, 104], pixels uint8 [81, 480, 832, 3]
  ref3_corgi.safetensors      upstream CPU bf16, first 3 latent frames: latents, pixels uint8 [9, 480, 832, 3]
"""
import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np


def save(path, arrays):
    mx.save_safetensors(str(path), {k: mx.array(np.ascontiguousarray(v)) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays)[:4], "...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spike", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s, out = Path(a.spike), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("corgi", "car"):
        z = np.load(s / "out" / f"inputs_{name}.npz")
        arrs = {"prompt_embeds": z["prompt_embeds"].astype(np.float32),
                "noise": z["noise"][0].astype(np.float32)}
        for k in z.files:
            if k.startswith("eps_"):
                arrs[k] = z[k][0].astype(np.float32)
        save(out / f"inputs_{name}.safetensors", arrs)
    lanes = np.load(s / "out" / "lanes_fp32attn.npz")
    save(out / "lanes_fp32attn.safetensors", {k: lanes[k].astype(np.float32) for k in lanes.files})
    m = s / "out" / "mlx-bf16"
    save(out / "pymlx_bf16_corgi.safetensors", {
        "latents": np.load(m / "latents_mlx.npy")[0].astype(np.float32),
        "pixels": np.load(m / "pixels_mlx.npy").astype(np.uint8)})
    r = s / "out" / "ref3"
    save(out / "ref3_corgi.safetensors", {
        "latents": np.load(r / "latents_ref.npy")[0].astype(np.float32),
        "pixels": np.load(r / "pixels_ref.npy").astype(np.uint8)})


if __name__ == "__main__":
    main()
