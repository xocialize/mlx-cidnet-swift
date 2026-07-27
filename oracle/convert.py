"""P3 weight conversion: Fediory/HVI-CIDNet-* -> safetensors in MLX NHWC layout.

Upstream already publishes `model.safetensors` (PyTorchModelHubMixin), so this is a relayout, not a
format change. One transform:

    conv weight   (O, I, kH, kW) -> (O, kH, kW, I)
                  depthwise (C, 1, 3, 3) -> (C, 3, 3, 1), which is what MLX Conv2d(groups: C) wants

Everything else — LayerNorm weight/bias, PReLU slope, the per-head CAB `temperature`, and the
learned `trans.density_k` — is rank ≤ 3 and passes through untouched.

`density_k` is LEARNED and differs per checkpoint (Generalization 0.98, LOLv1-wperc 1.13,
SICE 0.40 against an init of 0.2), so it must ride in the weights and never be hardcoded.

Run:  .venv/bin/python convert.py [repo-id]
"""
import json
import os
import sys

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file, save_file

REPO = sys.argv[1] if len(sys.argv) > 1 else "Fediory/HVI-CIDNet-Generalization"
OUT = os.path.join("converted", REPO.split("/")[-1])
os.makedirs(OUT, exist_ok=True)

src = hf_hub_download(REPO, "model.safetensors", cache_dir="hf-cache")
sd = load_file(src)
print(f"=== {REPO} ===")
print(f"  source: {len(sd)} tensors")

converted = {}
stats = {"conv": 0, "depthwise": 0, "passthrough": 0}

for k, v in sd.items():
    a = np.asarray(v, dtype=np.float32)
    if a.ndim == 4:
        depthwise = a.shape[1] == 1 and a.shape[0] > 1
        a = np.transpose(a, (0, 2, 3, 1))
        stats["depthwise" if depthwise else "conv"] += 1
    else:
        stats["passthrough"] += 1
    converted[k] = np.ascontiguousarray(a)

print("  transforms:")
for kind, n in stats.items():
    print(f"     {kind:12s}: {n}")
total = sum(int(np.prod(v.shape)) for v in converted.values())
print(f"  params: {total:,}  ({total * 4 / 1e6:.2f} MB fp32)")

k_key = [k for k in converted if "density_k" in k]
print(f"  density_k key(s): {k_key} -> {[float(converted[k].ravel()[0]) for k in k_key]}")

meta = {
    "format": "pt",
    "source": REPO,
    "license": "MIT",
    "layout": "MLX NHWC; conv (O,kH,kW,I)",
    "params": str(total),
}
save_file(converted, os.path.join(OUT, "model.safetensors"), metadata=meta)

with open(os.path.join(OUT, "CONVERSION.json"), "w") as f:
    json.dump({"repo": REPO, "transforms": stats, "params": total}, f, indent=2)

sz = os.path.getsize(os.path.join(OUT, "model.safetensors")) / 1e6
print(f"  written: {OUT}/model.safetensors  ({sz:.2f} MB)")
