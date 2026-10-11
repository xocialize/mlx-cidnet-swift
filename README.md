# mlx-cidnet-swift

**HVI-CIDNet** exposure / low-light correction on Apple Silicon — an MLX-Swift port, and the first
MLXEngine `imageRelight` package.

Upstream: [`Fediory/HVI-CIDNet`](https://github.com/Fediory/HVI-CIDNet) — **MIT**, no non-commercial
clause anywhere. Yan et al., *HVI: A New Color Space for Low-light Image Enhancement*, **CVPR 2025**;
1st place, NTIRE 2025. 1,975,569 parameters (7.9 MB).

Weights: [`mlx-community/HVI-CIDNet-*-fp32`](https://huggingface.co/collections/mlx-community/hvi-cidnet-mlx-6a67a33c0faa782edbc6752e).

**Engine-materialized weights read in place since 0.1.1 (AB-A-0112).** MLXEngine downloads the declared
`weightSources` into its model store before `load()`; `load()` reads that copy (flat or snapshot layout) instead of
fetching a second one through swift-transformers' `HubApi`. It also reads `<root>/models/<org>/<name>/`, where earlier
versions downloaded, so existing installs aren't fetched again. `HubApi` remains only the backstop outside the engine
(no store, or nothing materialized); an explicit `weightsURL` bypasses all of it.

**Any image size since 0.1.2 (AB-T-0223).** The UNet halves the resolution three times, so the network needs H and W
divisible by 8. The forward reflect-pads the bottom and right edges up to the next multiple of 8, as upstream's eval
does, and crops the output back. Before 0.1.2 a size like 570×320 hit an MLX concatenate `fatalError`, which took the
host process down with it.

## Products

| Product | Depends on | Purpose |
|---|---|---|
| `CIDNetMLXCore` | MLX only | the model + the HVI colour transform. Standalone-usable. |
| `MLXCIDNet` | + MLXToolKit | the `imageRelight` `ModelPackage`, incl. the exposure gate. |
| `cidnet-gate` | — | CLI parity gates. |

## ⚠️ This model must be gated on input exposure

Every published checkpoint drives its output toward a **target mean luma regardless of input**, so on
an already-correctly-exposed image it *degrades* the picture:

| checkpoint | applied to a correct exposure |
|---|---|
| Generalization (default) | 23.37 dB |
| LOLv1-wperc | 20.99 dB |
| SICE | **16.10 dB** |

`MLXCIDNet` handles this: a Rec.709 luma estimate runs **before** any model work, bypasses above a
threshold, and reports `bypassed: true`. `strength` (0…1) blends toward the input.

That behaviour is exactly why `imageRelight` is a separate capability rather than a second
`imageRestore` package — a planner told to "restore" an image must not silently re-expose it.

⚠️ The default threshold (0.35) is a **placeholder** from a synthetic exposure ladder, not a
calibrated value. See [PORT-STATUS.md](PORT-STATUS.md).

## Gates

```bash
swift run cidnet-gate --s0  <weights>
swift run cidnet-gate --all <goldens> <weights>
swift test
```

All green: key contract 191 tensors, HVI transform **bit-exact**, full model at cosine 1.00000000,
conformance 11/11.

## License

Port code MIT. Upstream model and weights MIT — see [NOTICE](NOTICE).
