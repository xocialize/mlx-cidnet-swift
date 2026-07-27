"""P3 ship-blocker: MEASURE the "fails on normally-exposed images" claim instead of assuming it.

PORT-QUEUE cites arXiv 2511.15496 — LOL-trained low-light models catastrophically fail on
*moderately* underexposed images, several SOTA methods scoring below doing nothing, because LOL has
one severely-underexposed image per scene so models learn a near-fixed gain.

That drives a mandatory luma gate. But two published checkpoints are trained differently
(Generalization, and SICE which is a MULTI-EXPOSURE dataset), so the size of the problem — and
therefore how much the gate has to carry — is an empirical question per checkpoint.

Method: take a well-exposed reference image, synthesize an exposure ladder by scaling it, run each
checkpoint, and measure how far the output drifts from the correctly-exposed reference. A model
that behaves would leave an already-correct image roughly alone.

Run:  .venv/bin/python exposure_probe.py
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "upstream")
from net.CIDNet import CIDNet  # noqa: E402

torch.set_grad_enabled(False)

CKPTS = [
    "Fediory/HVI-CIDNet-LOLv1-wperc",       # the headline LOL-v1 model — expected worst offender
    "Fediory/HVI-CIDNet-Generalization",
    "Fediory/HVI-CIDNet-SICE",              # multi-exposure training set
]

# A well-exposed synthetic scene: mid-grey mean, full tonal range, some structure.
g = np.random.default_rng(4242)
yy, xx = np.mgrid[0:256, 0:256].astype(np.float32) / 255.0
base = np.stack([
    0.5 + 0.32 * np.sin(xx * 9) * np.cos(yy * 7),
    0.5 + 0.32 * np.cos(xx * 6 + yy * 5),
    0.5 + 0.28 * np.sin((xx + yy) * 11),
])[None]
base = np.clip(base + 0.02 * g.standard_normal(base.shape, dtype=np.float32), 0, 1).astype(np.float32)
ref = torch.from_numpy(np.ascontiguousarray(base))

# Exposure ladder. 1.0 = correctly exposed; below that is progressively darker.
SCALES = [1.0, 0.7, 0.5, 0.3, 0.15, 0.08]


def psnr(a, b):
    mse = float(torch.mean((a.clamp(0, 1) - b.clamp(0, 1)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


print("Each row: input at a given exposure -> model output, scored against the CORRECTLY EXPOSED")
print("reference. Higher is better. The question for the gate is the TOP row (scale 1.00): a model")
print("that leaves a good image alone scores high; one that 'enhances' it anyway scores low.\n")

for repo in CKPTS:
    m = CIDNet.from_pretrained(repo).eval()
    print(f"=== {repo.split('/')[-1]}   (density_k = {m.trans.density_k.item():.4f}) ===")
    print("   scale  in_mean  out_mean   PSNR(in,ref)   PSNR(out,ref)   delta")
    for s in SCALES:
        x = (ref * s).clamp(0, 1)
        y = m(x)
        p_in, p_out = psnr(x, ref), psnr(y, ref)
        flag = ""
        if s == 1.0 and p_out < 25:
            flag = "  <-- DEGRADES an already-good image"
        elif p_out < p_in:
            flag = "  <-- WORSE than doing nothing"
        print(f"   {s:5.2f}  {float(x.mean()):7.4f}  {float(y.mean()):8.4f}   "
              f"{p_in:11.2f}   {p_out:13.2f}   {p_out - p_in:+6.2f}{flag}")
    print()
