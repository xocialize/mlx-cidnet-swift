"""Publish the converted HVI-CIDNet weights to mlx-community.

Three of the authors' ten checkpoints — the ones that differ enough to be a product choice:

  Generalization -> mlx-community/HVI-CIDNet-Generalization-fp32   (default)
  LOLv1-wperc    -> mlx-community/HVI-CIDNet-LOLv1-fp32
  SICE           -> mlx-community/HVI-CIDNet-SICE-fp32

Run:  .venv/bin/python publish.py [--dry-run]
"""
import os
import sys

from huggingface_hub import HfApi, add_collection_item, create_collection

API = HfApi()
HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = {
    "HVI-CIDNet-Generalization": (
        "HVI-CIDNet-Generalization-fp32", 0.9835,
        "**Recommended default.** Trained for cross-dataset generalization. Does the least damage "
        "when the exposure gate mis-fires (23.37 dB on an already-correct exposure, vs 20.99 / "
        "16.10 for the others)."),
    "HVI-CIDNet-LOLv1-wperc": (
        "HVI-CIDNet-LOLv1-fp32", 1.1255,
        "The headline LOL-v1 model (perceptual loss). Strongest lift on *severely* underexposed "
        "input (+14.50 dB at 0.15x exposure) and the best published LOL numbers."),
    "HVI-CIDNet-SICE": (
        "HVI-CIDNet-SICE-fp32", 0.4003,
        "Trained on SICE (multi-exposure). The most aggressive auto-exposure normalizer — its "
        "output mean is near-constant across a 12x input range, so it will DARKEN a bright image "
        "by design."),
}


def card(repo, k, blurb):
    return f"""---
library_name: mlx
license: mit
license_link: https://github.com/Fediory/HVI-CIDNet/blob/master/LICENSE
base_model: Fediory/HVI-CIDNet
pipeline_tag: image-to-image
tags:
  - mlx
  - low-light-enhancement
  - exposure-correction
  - image-enhancement
  - hvi-cidnet
---

# mlx-community/{repo}

[HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet) low-light / exposure correction, converted to
**Apple MLX** for Apple-Silicon inference via the
[`mlx-cidnet-swift`](https://github.com/xocialize/mlx-cidnet-swift) Swift package.

Yan et al., *HVI: A New Color Space for Low-light Image Enhancement*, **CVPR 2025** — 1st place,
NTIRE 2025 Low-Light Enhancement Challenge. 1,975,569 parameters (7.9 MB). Underexposed image in →
re-exposed image at the same resolution out.

{blurb}

## Use with mlx-cidnet-swift

```swift
import CIDNetMLXCore

let model = CIDNet()
try model.loadWeights(from: weightsURL)   // model.safetensors from this repo
let brightened = model(imageNHWC)          // NHWC RGB in [0,1]
```

Or as an MLXEngine `imageRelight` ModelPackage (`MLXCIDNet.CIDNetRelightPackage`), which resolves
this repo via the Hub and applies the exposure gate below automatically.

## ⚠️ Gate this model on input exposure — it is not safe to apply unconditionally

Every published checkpoint drives its output toward a **target mean luma regardless of input**. On
an image that is already correctly exposed the model therefore *degrades* it. Measured against a
correctly-exposed reference:

| checkpoint | applied to an already-correct exposure |
|---|---|
| Generalization | 23.37 dB |
| LOLv1-wperc | 20.99 dB |
| SICE | **16.10 dB** |

SICE's output mean stays within 0.35–0.40 across a **12× input range** — it is an auto-exposure
normalizer, not a shadow lift.

Estimate input luma and **bypass above a threshold**; expose a strength blend for partial
application. `mlx-cidnet-swift` does both. Note also that Generalization scores **−0.52 dB — worse
than doing nothing** — at 0.70× exposure, which is the *moderate* underexposure regime that
LOL-trained models are documented to handle badly.

## Conversion

MLX **NHWC** layout: 191 tensors → 81 conv + 61 depthwise transposed `(O,I,kH,kW) → (O,kH,kW,I)`,
49 passthrough. Upstream already publishes safetensors, so this is a relayout, not a format change.

`trans.density_k` is a **learned** parameter and differs per checkpoint (this one: **{k}**; init is
0.2). It rides in the weights — do not hardcode it.

## Parity

Gated against the PyTorch oracle on the CPU stream, fp32, judged on relative error:

- **key contract** — 191 tensors / 1,975,569 params / 0 missing / 0 unused, strict load
- **HVI colour transform** — **bit-exact** (worst 6.6e-07), including the degenerate colours and
  the yellow/cyan channel *ties* that decide the masking priority
- **primitives** — PReLU exactly 0.00e+00; the align-corners resamplers ~5e-07
- **full model** — cosine **1.00000000** at 64² / 128² / 256², and on a severely underexposed tile

Weights: MIT (Fediory/HVI-CIDNet, published first-party by the authors). Port code: MIT.
"""


def main():
    dry = "--dry-run" in sys.argv
    published = []
    for src, (repo, k, blurb) in VARIANTS.items():
        rid = f"mlx-community/{repo}"
        weights = os.path.join(HERE, "converted", src, "model.safetensors")
        if not os.path.exists(weights):
            print(f"[skip] {rid}: missing {weights}")
            continue
        if dry:
            print(f"[dry-run] would upload {weights} -> {rid}")
            continue
        print(f"[publish] {rid} ({os.path.getsize(weights)/1e6:.2f} MB) …")
        API.create_repo(rid, repo_type="model", exist_ok=True)
        API.upload_file(path_or_fileobj=weights, path_in_repo="model.safetensors", repo_id=rid)
        API.upload_file(path_or_fileobj=card(repo, k, blurb).encode(),
                        path_in_repo="README.md", repo_id=rid)
        published.append((rid, blurb))
        print(f"[publish]   ok → https://huggingface.co/{rid}")

    if published:
        col = create_collection(
            title="HVI-CIDNet (MLX)", namespace="mlx-community", exists_ok=True,
            # HF caps this at 150 characters.
            description="Apple-MLX HVI-CIDNet exposure correction (CVPR 2025, MIT). "
                        "Gate on input exposure: these re-expose toward a target luma.")
        for rid, note in published:
            try:
                add_collection_item(col.slug, item_id=rid, item_type="model", note=note[:500])
            except Exception as e:
                print(f"[collection]   {rid}: {e}")
        print(f"[collection] ok → https://huggingface.co/collections/{col.slug}")


if __name__ == "__main__":
    main()
