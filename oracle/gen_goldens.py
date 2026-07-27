"""P3 oracle — per-sub-op goldens for the HVI-CIDNet Swift port.

fp32, CPU-torch, numpy-seeded, C-contiguous, PyTorch NCHW. The Swift gate transposes to NHWC,
runs, transposes back, compares.

The HVI colour transform is the piece that must be bit-exact, so it gets the most coverage:
forward, inverse, round-trip, and the degenerate cases (pure grey, pure black, saturated primaries)
where the masked-assignment ordering and the eps guards actually bite.

Run:  .venv/bin/python gen_goldens.py [repo-id]
Out:  goldens/*.npy  +  goldens/MANIFEST.txt
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "upstream")
from net.CIDNet import CIDNet  # noqa: E402

torch.set_grad_enabled(False)

REPO = sys.argv[1] if len(sys.argv) > 1 else "Fediory/HVI-CIDNet-Generalization"
OUT = "goldens"
os.makedirs(OUT, exist_ok=True)

model = CIDNet.from_pretrained(REPO)
model.eval()

manifest = []


def save(name, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    np.save(os.path.join(OUT, name + ".npy"), a)
    manifest.append(f"{name + '.npy':44s} {str(a.shape):26s} "
                    f"min={a.min():+.6f} max={a.max():+.6f} mean={a.mean():+.6f}")
    print(f"  saved {name}.npy  {a.shape}")


def dump(name, t):
    save(name, t.detach().cpu().numpy())


def seeded(seed, *shape):
    g = np.random.default_rng(seed)
    return torch.from_numpy(g.standard_normal(shape, dtype=np.float32))


def seeded01(seed, *shape):
    """Image-like input in [0,1] — HVIT assumes this range."""
    g = np.random.default_rng(seed)
    return torch.from_numpy(g.random(shape, dtype=np.float32))


print(f"\n=== checkpoint: {REPO} ===")
print(f"  density_k = {model.trans.density_k.item():.8f}  (LEARNED, differs per checkpoint)")
save("density_k", model.trans.density_k.detach().numpy())

# ---------------------------------------------------------------- HVI transform
print("\n=== 1. HVIT (RGB -> HVI) ===")
x = seeded01(4001, 1, 3, 64, 64)
dump("hvit_in", x)
dump("hvit_out", model.trans.HVIT(x))

print("\n=== 2. HVIT degenerate cases (where the masking and eps guards bite) ===")
# Grey (min == max, so `hue[min==value] = 0`), pure black (value == 0 → saturation forced 0),
# pure white, and each saturated primary — one per row of an 8-wide strip.
edge = torch.zeros(1, 3, 1, 8)
edge[0, :, 0, 0] = torch.tensor([0.0, 0.0, 0.0])   # black
edge[0, :, 0, 1] = torch.tensor([1.0, 1.0, 1.0])   # white
edge[0, :, 0, 2] = torch.tensor([0.5, 0.5, 0.5])   # grey
edge[0, :, 0, 3] = torch.tensor([1.0, 0.0, 0.0])   # red   (r is max)
edge[0, :, 0, 4] = torch.tensor([0.0, 1.0, 0.0])   # green (g is max)
edge[0, :, 0, 5] = torch.tensor([0.0, 0.0, 1.0])   # blue  (b is max)
edge[0, :, 0, 6] = torch.tensor([1.0, 1.0, 0.0])   # yellow — TIE between r and g
edge[0, :, 0, 7] = torch.tensor([0.0, 1.0, 1.0])   # cyan   — TIE between g and b
dump("hvit_edge_in", edge)
dump("hvit_edge_out", model.trans.HVIT(edge))

print("\n=== 3. PHVIT (HVI -> RGB) ===")
# PHVIT reads self.this_k, which HVIT sets. Call HVIT first so the golden reflects the real
# (stateful) upstream path; the Swift port uses density_k directly, which is equivalent.
_ = model.trans.HVIT(x)
hvi_in = model.trans.HVIT(x)
dump("phvit_in", hvi_in)
dump("phvit_out", model.trans.PHVIT(hvi_in))

print("\n=== 4. Round-trip HVIT -> PHVIT (should recover the input closely) ===")
rt = model.trans.PHVIT(model.trans.HVIT(x))
dump("roundtrip_out", rt)
print(f"     max |rt - in| = {(rt - x).abs().max():.6e}")

# ---------------------------------------------------------------- primitives
print("\n=== 5. LayerNorm (channels_first, eps=1e-6) ===")
ln = model.HV_LCA1.norm
xl = seeded(4002, 1, 36, 32, 32)
dump("layernorm_in", xl)
dump("layernorm_out", ln(xl))

print("\n=== 6. PReLU (single learned slope) ===")
pr = model.HVE_block1.prelu
xp = seeded(4003, 1, 36, 16, 16)
dump("prelu_in", xp)
dump("prelu_out", pr(xp))
save("prelu_weight", pr.weight.detach().numpy())

print("\n=== 7. NormDownsample / NormUpsample ===")
# NOTE: these use nn.UpsamplingBilinear2d == Upsample(bilinear, align_corners=TRUE).
# That differs from the align_corners=False used elsewhere in the queue's ports.
xd = seeded(4004, 1, 36, 32, 32)
dump("down_in", xd)
dump("down_out", model.HVE_block1(xd))

xu = seeded(4005, 1, 144, 8, 8)
yu = seeded(4006, 1, 72, 16, 16)
dump("up_x", xu)
dump("up_y", yu)
dump("up_out", model.HVD_block3(xu, yu))

# ---------------------------------------------------------------- blocks
print("\n=== 8. CAB (channel-dim cross attention) ===")
cab = model.HV_LCA1.ffn
xc = seeded(4007, 1, 36, 16, 16)
yc = seeded(4008, 1, 36, 16, 16)
dump("cab_x", xc)
dump("cab_y", yc)
dump("cab_out", cab(xc, yc))

print("\n=== 9. IEL ===")
iel = model.HV_LCA1.gdfn
xi = seeded(4009, 1, 36, 16, 16)
dump("iel_in", xi)
dump("iel_out", iel(xi))

print("\n=== 10. HV_LCA vs I_LCA (they differ: I_LCA keeps a residual on gdfn, HV_LCA does not) ===")
xa = seeded(4010, 1, 36, 16, 16)
ya = seeded(4011, 1, 36, 16, 16)
dump("lca_x", xa)
dump("lca_y", ya)
dump("hv_lca_out", model.HV_LCA1(xa, ya))
dump("i_lca_out", model.I_LCA1(xa, ya))

# ---------------------------------------------------------------- full model
print("\n=== 11. Full model ===")
for size in (64, 128, 256):
    xi = torch.from_numpy(
        np.random.default_rng(5000 + size).random((1, 3, size, size), dtype=np.float32))
    dump(f"full_{size}_in", xi)
    dump(f"full_{size}_out", model(xi))

print("\n=== 12. Full model on a DARK structured image (the real use case) ===")
g = np.random.default_rng(6001)
yy, xx = np.mgrid[0:256, 0:256].astype(np.float32) / 255.0
img = np.stack([
    0.5 + 0.35 * np.sin(xx * 9) * np.cos(yy * 7),
    0.5 + 0.35 * np.cos(xx * 6 + yy * 5),
    0.5 + 0.30 * np.sin((xx + yy) * 11),
])[None]
img = np.clip(img + 0.02 * g.standard_normal(img.shape, dtype=np.float32), 0, 1)
dark = (img * 0.12).astype(np.float32)          # severely underexposed
xi = torch.from_numpy(np.ascontiguousarray(dark))
dump("full_dark256_in", xi)
dump("full_dark256_out", model(xi))

with open(os.path.join(OUT, "MANIFEST.txt"), "w") as f:
    f.write("HVI-CIDNet PyTorch goldens — fp32, CPU, PyTorch NCHW, C-contiguous.\n")
    f.write(f"checkpoint: {REPO}\n")
    f.write(f"density_k : {model.trans.density_k.item():.8f}  (learned, per-checkpoint)\n")
    f.write("constructor: CIDNet()  channels=[36,36,72,144] heads=[1,2,4,8] norm=False\n")
    f.write("input contract: RGB [0,1]; no size multiple required (3 bilinear /2 stages).\n\n")
    f.write("\n".join(manifest) + "\n")

print(f"\n✅ {len(manifest)} goldens written to {OUT}/")
