"""Causal Forcing++ framewise-2step (`zhuhz22/Causal-Forcing`, `causal-forcing++/framewise-2step.pt`, Apache-2.0)
-> `dit.safetensors`: the `generator_ema` weights in bf16 with upstream FastFrames' key names — the layout
mlx-community/Causal-Forcing-PP-framewise-2step-bf16 publishes and FastFramesMLX's CausalDiT reads.

    python -I convert_dit.py --pt <framewise-2step.pt> --out <dir>

- `generator_ema` (the EMA weights FastFrames' `checkpoints.load_state_dict` selects), FSDP / `model.` prefixes stripped
- fp32 -> bf16 (round to nearest even): FastFrames runs the DiT in bf16 (`build_wan(..., dtype=bfloat16)`)
- tensors otherwise untouched: `patch_embedding.weight` stays a Conv3d [1536, 16, 1, 2, 2]
- writes `config.json` with the source revision, sha256s and tensor count

The pickle is loaded with `weights_only=True` (no code execution).
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

PREFIXES = ("model._fsdp_wrapped_module.", "_fsdp_wrapped_module.", "model.")


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default="2f8eb8bb6eeb1238da9d13e5420d342a74d634a6")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sd = torch.load(a.pt, map_location="cpu", mmap=True, weights_only=True)["generator_ema"]
    tensors = {}
    for k, v in sd.items():
        for p in PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
                break
        tensors[k] = v.to(torch.bfloat16).contiguous()
    assert len(tensors) == 825, len(tensors)
    dst = out / "dit.safetensors"
    save_file(tensors, str(dst), metadata={"format": "pt"})
    cfg = {
        "model": "Causal Forcing++ framewise-2step (generator_ema) — the FastFrames-S DiT",
        "architecture": "Wan2.1-T2V-1.3B, frame-causal (dim 1536, ffn 8960, 12 heads, 30 layers, in/out 16)",
        "dtype": "bfloat16",
        "key_names": "upstream (apple/ml-fastframes CausalWanModel / Wan2.1)",
        "num_tensors": len(tensors),
        "num_params": int(sum(t.numel() for t in tensors.values())),
        "source": {
            "repo": "zhuhz22/Causal-Forcing",
            "revision": a.revision,
            "file": "causal-forcing++/framewise-2step.pt",
            "sha256": sha256(a.pt),
            "state_dict": "generator_ema",
            "license": "apache-2.0",
        },
        "dit_safetensors_sha256": sha256(dst),
        "converter": "mlx-fastframes-swift Tools/convert_dit.py",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
