# mlx-cidnet-swift — port status

**Work order:** P3 in `mlxengine-todo/PORT-QUEUE.md` — HVI-CIDNet low-light / exposure correction.

Upstream: [`Fediory/HVI-CIDNet`](https://github.com/Fediory/HVI-CIDNet) — **MIT**, Copyright (c) 2024
Fediory Feng. Yan et al., *HVI: A New Color Space for Low-light Image Enhancement*, **CVPR 2025**;
1st place, NTIRE 2025 Low-Light Enhancement.

---

## Stage 0 — upstream verification ✅ PASSED (2026-07-27)

| Fact | Verified value |
|---|---|
| License | **MIT**, and **zero** non-commercial mentions anywhere in the repo |
| Weights | **Ten** first-party checkpoints under the `Fediory` HF org, each publishing `model.safetensors` + `config.json` via `PyTorchModelHubMixin` |
| Parameters | **1,975,569** — 7.90 MB fp32 / 3.95 MB fp16 |
| Constructor | `channels=[36,36,72,144]`, `heads=[1,2,4,8]`, `norm=False` |
| Load + forward | verified on Generalization, LOLv1-wperc and SICE |

> ⚠️ **Weight availability was probed BEFORE planning.** P1 in the same queue was abandoned after its
> weights turned out to be permanently gone. The queue's ✅ marks verify *licences*, not that a
> checkpoint still downloads.

### `density_k` is learned and per-checkpoint

Generalization **0.9835** · LOLv1-wperc **1.1255** · SICE **0.4003** — against an init of 0.2. It
rides in the weights and must never be hardcoded.

### Corrections to the queue's description, read from source

- Padding is **`ReplicationPad2d`**, not reflection.
- Resampling is **`nn.UpsamplingBilinear2d`** = bilinear with **`alignCorners: true`** — the opposite
  of the sibling FFTformer port, and an easy thing to copy across wrongly.

### Upstream quirks the port must respect

1. **`I_LCA5` is dead code.** `forward` computes `i_dec2 = I_LCA5(...)` and then overwrites `i_dec2`
   with `ID_block2(...)` before any read. **Verified by ablation:** zeroing `I_LCA5`'s weights
   changes the output by exactly **0.0**, while `HV_LCA5`, `I_LCA2` and `I_LCA6` each move it by
   ~0.5–1.0. Declared for strict loading, never evaluated.
2. **`RGB_HVI` is stateful:** `HVIT` writes `self.this_k` and `PHVIT` reads it back, so calling the
   inverse alone silently uses `this_k = 0`. This port reads the parameter directly in both
   directions — identical on the intended path, correct on the one upstream leaves broken.
3. **Hue is built by sequential masked assignment**, later writes winning, so the `where` chain must
   run grey > red > green > blue. The `% 6` applies to the **red branch only**, and it is Python
   floored modulo (`-0.5 % 6 == 5.5`), implemented as `x - 6·floor(x/6)`.
4. **`HV_LCA` and `I_LCA` differ by one line** — `I_LCA` keeps a residual around `gdfn`, `HV_LCA`
   replaces. Not a typo upstream.
5. The encoder deliberately downsamples the **pre-LCA** tensors (`i_enc3 = IE_block3(i_enc2)`); the
   LCA outputs survive only through the jump connections.

---

## Stage 1 — parity ✅ **ALL GATES GREEN, FIRST ATTEMPT**

`swift run cidnet-gate --all <goldens> <weights>`, CPU stream, judged on **relative** error.

| Gate | Result | Worst |
|---|---|---|
| **S0** key contract | ✅ 191 tensors, 1,975,569 params, strict `verify: .all` clean — and passes for **all three** checkpoints | exact |
| **HVI** colour transform | ✅ 4/4 — **bit-exact** | 6.6e-07 |
| **S1** primitives | ✅ 4/4 — PReLU exactly **0.00e+00** | 5.5e-07 |
| **S2** blocks | ✅ 4/4 | 3.4e-07 |
| **S3** full model | ✅ 4/4 at cosine **1.00000000**, incl. a severely underexposed tile | 1.3e-06 |

The HVI gate covers the degenerate colours where the masking actually decides the answer — black,
white, grey, the three primaries, and **yellow/cyan, which are exact ties** between two channels and
therefore test the priority ordering directly.

Full-model agreement at ~1e-06 is far tighter than the sibling FFTformer port's 3.6e-05, simply
because CIDNet is a fifth the size with fewer accumulation steps.

Conversion (`oracle/convert.py`): 191 tensors → 81 conv + 61 depthwise transposed to NHWC, 49
passthrough. Upstream already ships safetensors, so this is a relayout, not a format change.

---

## 🔴 Ship-blocker: the luma gate is **mandatory for every checkpoint** — measured

The queue requires a luma gate because LOL-trained models fail on *moderately* underexposed images.
I hypothesised the Generalization and SICE checkpoints (SICE being **multi-exposure**) might not need
it. **That is disproven.** `oracle/exposure_probe.py` scales a correctly-exposed reference down a
ladder and scores each output against the correct exposure:

| checkpoint | @1.00 (already good) | notable |
|---|---|---|
| **Generalization** | **23.37 dB** | best-behaved, but at 0.70 scores **−0.52 dB — worse than doing nothing**, the paper's exact hole |
| LOLv1-wperc | 20.99 dB | brightens (0.50 → 0.58); biggest win on severe underexposure (+14.50 dB @0.15) |
| SICE | **16.10 dB** | **worst** — darkens hard (0.50 → 0.35) despite multi-exposure training |

The shared root cause: **all three drive output toward a target mean luma regardless of input.**
SICE's output mean is nearly constant (0.35–0.40) across a 12× input range — it is an auto-exposure
normalizer, so it will darken an already-bright image by design.

**Recommendation: ship Generalization as the default** — it does the least damage when the gate
mis-fires — with a mandatory luma gate and a strength slider.

⚠️ **Caveat: this ladder is synthetic linear scaling**, not real under-exposure (no sensor noise, no
clipping asymmetry). It is directional evidence that the gate is required, **not** a calibrated
threshold. Real exposure fixtures are still needed to set the actual cut.

🔑 Free gate input for iPhone video, found while running V8: `com.apple.quicktime.scene-illuminance`
is written **per frame** (~25.9 Hz) — a capture-time illuminance measurement, better evidence than
estimating luma from processed pixels.

---

## Remaining

- [ ] Capability decision (queue stop-and-ask): new `imageRelight` vs a second package on
      `imageRestore`. The queue leans new capability — it wants a strength parameter and composes
      differently from restoration.
- [ ] Luma gate implementation + real exposure fixtures to calibrate the threshold.
- [ ] `ModelPackage` wrapper, conformance gates, footprint measurement.
- [ ] Publish converted weights to `mlx-community`; registry row.

## Reproduce

```bash
cd oracle
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch numpy einops huggingface_hub safetensors
git clone --depth 1 https://github.com/Fediory/HVI-CIDNet.git upstream
.venv/bin/python gen_goldens.py            # 35 goldens
.venv/bin/python convert.py Fediory/HVI-CIDNet-Generalization
.venv/bin/python deadcode_probe.py         # the I_LCA5 ablation
.venv/bin/python exposure_probe.py         # the ship-blocker measurement
cd .. && swift run cidnet-gate --all oracle/goldens oracle/converted/HVI-CIDNet-Generalization/model.safetensors
```

`oracle/upstream`, `oracle/.venv`, `oracle/converted`, `oracle/hf-cache` are generated — not committed.
