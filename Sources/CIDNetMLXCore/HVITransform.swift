//
//  HVITransform.swift
//  mlx-cidnet-swift / CIDNetMLXCore
//
//  The HVI colour space — the one genuinely novel component of HVI-CIDNet, and the only piece the
//  port queue flags as needing to be bit-exact. Ported from `net/HVI_transform.py`.
//
//  Upstream: https://github.com/Fediory/HVI-CIDNet (MIT)
//  Paper:    Yan et al., "HVI: A New Color Space for Low-light Image Enhancement", CVPR 2025.
//
//  HVI is a polar re-parameterisation of HSV whose chroma is scaled by a *learned*,
//  intensity-dependent "colour sensitivity" term, so that dark pixels — where hue is numerically
//  unstable — contribute proportionally less chroma.
//
//  Conventions: NHWC, channel axis last. Upstream is NCHW.
//

import Foundation
import MLX
import MLXNN

/// RGB ↔ HVI, with the learned density exponent `k`.
///
/// Upstream keeps `k` in a stateful field: `HVIT` writes `self.this_k = k.item()` and `PHVIT`
/// *reads it back*, so calling the inverse without the forward silently uses `this_k = 0` (making
/// `color_sensitive` identically 1). This port takes `k` from the parameter directly in both
/// directions — numerically identical on the intended path, and correct on the path upstream
/// leaves broken.
public final class RGBHVI: Module, @unchecked Sendable {

    /// Learned density exponent. **Per-checkpoint** — Generalization 0.9835, LOLv1-wperc 1.1255,
    /// SICE 0.4003, against an init of 0.2 — so it must load from the weights, never be hardcoded.
    @ParameterInfo(key: "density_k") public var densityK: MLXArray

    public override init() {
        self._densityK.wrappedValue = MLXArray([Float(0.2)])
    }

    private static let pi = Float(3.141592653589793)
    private static let eps = Float(1e-8)

    /// Python/torch floored modulo: the result takes the sign of the divisor, unlike C truncation.
    /// `(-0.5) % 6` is `5.5`, not `-0.5`, and the hue wrap depends on it.
    private static func floorMod(_ x: MLXArray, _ m: Float) -> MLXArray {
        x - m * MLX.floor(x / m)
    }

    /// RGB `(B,H,W,3)` in [0,1] → HVI `(B,H,W,3)`.
    public func forward(_ rgb: MLXArray) -> MLXArray {
        let eps = Self.eps, pi = Self.pi
        let r = rgb[.ellipsis, 0], g = rgb[.ellipsis, 1], b = rgb[.ellipsis, 2]

        let value = MLX.max(rgb, axis: -1)
        let imgMin = MLX.min(rgb, axis: -1)
        let chroma = value - imgMin + eps

        // Upstream assigns the hue branches SEQUENTIALLY, so a later write wins:
        //   1. blue-max   hue = 4 + (r-g)/chroma
        //   2. green-max  hue = 2 + (b-r)/chroma
        //   3. red-max    hue = ((g-b)/chroma) % 6
        //   4. grey       hue = 0
        // Priority is therefore grey > red > green > blue, and the `% 6` applies to the red branch
        // ONLY. Nesting the `where`s in that order reproduces it exactly.
        let hueBlue = 4.0 + (r - g) / chroma
        let hueGreen = 2.0 + (b - r) / chroma
        let hueRed = Self.floorMod((g - b) / chroma, 6)

        var hue = MLX.which(g .== value, hueGreen, hueBlue)
        hue = MLX.which(r .== value, hueRed, hue)
        hue = MLX.which(imgMin .== value, MLXArray(Float(0)), hue)
        hue = hue / 6.0

        var saturation = (value - imgMin) / (value + eps)
        saturation = MLX.which(value .== 0, MLXArray(Float(0)), saturation)

        // Colour sensitivity: (sin(v·π/2) + eps)^k. Dark pixels get a small factor, which is the
        // whole point — it damps the chroma exactly where hue is least trustworthy.
        let colorSensitive = MLX.pow(MLX.sin(value * 0.5 * pi) + eps, densityK.reshaped([1]))

        let h = colorSensitive * saturation * MLX.cos(2.0 * pi * hue)
        let v = colorSensitive * saturation * MLX.sin(2.0 * pi * hue)
        return MLX.stacked([h, v, value], axis: -1)
    }

    /// HVI `(B,H,W,3)` → RGB `(B,H,W,3)`.
    public func inverse(_ hvi: MLXArray) -> MLXArray {
        let eps = Self.eps, pi = Self.pi

        var hh = MLX.clip(hvi[.ellipsis, 0], min: -1, max: 1)
        var vv = MLX.clip(hvi[.ellipsis, 1], min: -1, max: 1)
        let intensity = MLX.clip(hvi[.ellipsis, 2], min: 0, max: 1)

        let colorSensitive = MLX.pow(MLX.sin(intensity * 0.5 * pi) + eps, densityK.reshaped([1]))
        hh = MLX.clip(hh / (colorSensitive + eps), min: -1, max: 1)
        vv = MLX.clip(vv / (colorSensitive + eps), min: -1, max: 1)

        // The `+ eps` inside atan2 is upstream's, and it matters at the origin: atan2(0,0) is 0 but
        // atan2(eps,eps) is π/4, so a fully desaturated pixel lands on a different hue. Replicated
        // deliberately — dropping it would be a silent divergence.
        var h = MLX.atan2(vv + eps, hh + eps) / (2 * pi)
        h = Self.floorMod(h, 1)

        let s = MLX.clip(MLX.sqrt(hh * hh + vv * vv + eps), min: 0, max: 1)
        let v = MLX.clip(intensity, min: 0, max: 1)

        // Standard HSV sextant reconstruction.
        let hi = MLX.floor(h * 6.0)
        let f = h * 6.0 - hi
        let p = v * (1.0 - s)
        let q = v * (1.0 - f * s)
        let t = v * (1.0 - (1.0 - f) * s)

        // Upstream starts from zeros and writes only the six sextants, so anything outside 0…5
        // stays 0. `h` is wrapped to [0,1) so `hi == 6` should be unreachable, but the zero default
        // is preserved rather than assumed away.
        let zero = MLXArray(Float(0))
        func pick(_ s0: MLXArray, _ s1: MLXArray, _ s2: MLXArray,
                  _ s3: MLXArray, _ s4: MLXArray, _ s5: MLXArray) -> MLXArray {
            var out = zero + MLX.zeros(like: v)
            out = MLX.which(hi .== 0, s0, out)
            out = MLX.which(hi .== 1, s1, out)
            out = MLX.which(hi .== 2, s2, out)
            out = MLX.which(hi .== 3, s3, out)
            out = MLX.which(hi .== 4, s4, out)
            out = MLX.which(hi .== 5, s5, out)
            return out
        }
        let r = pick(v, q, p, p, t, v)
        let g = pick(t, v, v, q, p, p)
        let b = pick(p, p, t, v, v, q)

        return MLX.stacked([r, g, b], axis: -1)
    }
}
