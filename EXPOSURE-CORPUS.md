# Exposure corpus — what to shoot, and why

Two things about this package are currently **placeholders**, and both need real captures:

1. **The luma-gate threshold** (`CIDNetConfiguration.lumaGateThreshold`, default `0.35`). Derived from
   a synthetic exposure ladder. It is evidence the gate is *required*; it is not a calibrated cut.
2. **Whether the model actually helps** when it does fire, on real photographs rather than on
   post-hoc darkened ones.

Good news up front: **this corpus is far easier to collect than the deblur one.** Motion blur cannot
be captured twice, which is why `mlx-fftformer-swift`'s
[REAL-BLUR-VALIDATION.md](../mlx-fftformer-swift/REAL-BLUR-VALIDATION.md) needs tripod-paired
handheld shots and a homography. **Underexposure is reproducible** — the same static scene can be
photographed at several exposures from a locked-off camera, so the well-exposed frame *is* pixel-
aligned ground truth for the darker ones. No alignment, no warping, no residual.

---

## 🔴 The trap: do not darken images in software

The current threshold is uncalibrated precisely because it came from multiplying a bright image by a
constant. **That is not what an underexposed photograph looks like.** Real sensor underexposure
brings, all of which post-hoc scaling erases:

- **read noise and shot noise** that do *not* scale with the signal — the dark frame is noisier
  relative to its content, which is most of what the model is actually fighting;
- **quantisation collapse at the low end** — an 8-bit JPEG that recorded shadows in 6 code values
  cannot be recovered to smooth gradient, and scaling a bright image never produces that;
- **black-level offset and sensor floor**, so real shadows sit slightly above zero with a colour cast
  rather than at a clean scaled value;
- **the camera's own tone curve and noise reduction**, already applied.

A model evaluated on scaled images will look better than it is. Shoot the dark frames dark.

---

## The fixture classes

### A · Exposure brackets — "does it help, and by how much"

Locked-off camera, **static scene**, only exposure changes. The correctly-exposed frame is ground
truth; the darker frames are model inputs.

- Camera on a tripod or a solid surface; **nothing in the frame may move** (no foliage in wind, no
  traffic, no people, no shifting daylight).
- Shoot a **reference** at correct exposure, then **−1, −2, −3 EV**.
- Vary exposure with **shutter speed at fixed ISO** where you can. That keeps the noise character
  consistent across the bracket so you are measuring exposure, not ISO.
- Then shoot a **second set on auto-ISO** — because that is what phones actually do, and high-ISO
  dark frames are the realistic input.
- **20–30 scenes**, spanning: interiors under mixed light, outdoor dusk/night, backlit subjects,
  faces (skin tone drift is the most visible failure), and fine shadow texture (fabric, foliage,
  hair) where invented detail shows.

### B · 🔑 "Correctly dark" images — the class that breaks a naive gate

**This is the most important class in the corpus, and the least obvious.**

The gate currently decides on **mean luma alone**, and mean luma cannot distinguish *underexposed*
from *deliberately, correctly dark*. A night scene, a low-key portrait, a silhouette, a candlelit
interior, a dark-room product shot — all legitimately low-luma, all **correct as shot**. A pure
mean-luma gate will fire on every one of them, and the model — which drives output toward a target
mean luma regardless of input — will brighten them into something wrong.

Shoot **20–30 images that are dark on purpose and correct as they are**:

- night exteriors with intentional deep shadow
- low-key / Rembrandt portraits
- silhouettes against a bright window or sky
- candlelit or single-source interiors
- concert / stage lighting
- dark-background product or still-life

These have **no ground truth and need none** — the input *is* the correct answer, so the measurement
is simply "how far did the model move it," and the target is *not far at all*, or better, bypassed.

**If this class overlaps in mean luma with class A's underexposed frames — which it almost certainly
will — then mean luma is insufficient and the gate needs a better signal.** Candidates to evaluate
once the corpus exists: shadow-clipping fraction, histogram shape (a correctly-dark image is
*intentionally* bimodal; an underexposed one is compressed toward the floor), highlight headroom, or
the per-frame `com.apple.quicktime.scene-illuminance` metadata for iPhone video (found during V8 —
a capture-time illuminance reading, which is *scene* brightness rather than *image* brightness, and
so may separate the two classes directly).

That is the single most valuable thing this corpus can tell us.

### C · Normal exposures — the false-positive floor

**20–30 ordinary, correctly-exposed photographs.** Everyday snapshots, mixed subjects. These
establish how often the gate fires when it should not, and confirm the measured degradation
(23.37 dB for the default checkpoint) is what real photos actually see.

---

## Capture notes

- **Shoot RAW+JPEG if you can.** The JPEG is the realistic input; the RAW lets us confirm what was
  actually recoverable, which separates "the model hallucinated detail" from "the detail was there".
- Keep EXIF intact — ISO, shutter, and aperture are how we reconstruct the true EV offset. Do not
  route the files through anything that re-encodes.
- Transfer via **AirDrop or Image Capture**, not a Photos export. (Same lesson as V8: exports
  re-encode and strip metadata.)
- Mixed devices are fine and mildly useful — a phone pipeline and a camera pipeline underexpose
  differently.

---

## What we will compute

| Class | Metric | Passing looks like |
|---|---|---|
| A · brackets | PSNR / SSIM of (dark → model) vs the correct-exposure frame, and of (dark) vs it | model beats the untouched input, per scene, with the gap widening as EV drops |
| B · correctly dark | Δ against the input itself; and the bypass rate | **bypassed**, or moved very little. Any large brightening here is a failure |
| C · normal | bypass rate; PSNR of output vs input when it does fire | bypass rate ≈ 100% |

The **threshold** falls out of A + B together: plot mean luma for both classes and pick the cut that
best separates "needs help" from "correct as shot". If they do not separate, that is the finding, and
the gate grows a second signal.

Report per-scene deltas, not just an average — a model that wins on average while wrecking every
low-key portrait has not passed.

---

## Minimum viable set

If the full corpus is too much, this still answers the most important question:

- **10 exposure brackets** (class A, reference + −2 EV)
- **15 correctly-dark images** (class B)
- **10 normal images** (class C)

That is enough to calibrate a threshold and to find out whether mean luma can do the job at all.
