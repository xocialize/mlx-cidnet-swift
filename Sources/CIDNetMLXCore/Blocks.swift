//
//  Blocks.swift
//  mlx-cidnet-swift / CIDNetMLXCore
//
//  The CIDNet building blocks, mirroring `net/transformer_utils.py` and `net/LCA.py`.
//  NHWC throughout; module keys match the upstream state dict.
//

import Foundation
import MLX
import MLXNN

/// Channel-dimension LayerNorm (upstream `data_format="channels_first"`, `eps=1e-6`).
///
/// Upstream normalizes over dim=1 of NCHW — i.e. channels. In NHWC channels are already last, so
/// `MLXNN.LayerNorm` is the identical computation with no reshaping. Note eps is **1e-6** here
/// (FFTformer's channel LayerNorm used 1e-5 — same idea, different constant).
public final class ChannelLayerNorm: Module, UnaryLayer, @unchecked Sendable {
    @ParameterInfo(key: "weight") public var weight: MLXArray
    @ParameterInfo(key: "bias") public var bias: MLXArray

    public init(_ dim: Int) {
        self._weight.wrappedValue = MLXArray.ones([dim])
        self._bias.wrappedValue = MLXArray.zeros([dim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let centered = x - mean
        let variance = (centered * centered).mean(axis: -1, keepDims: true)
        return weight * (centered / MLX.sqrt(variance + 1e-6)) + bias
    }
}

/// `Conv2d → bilinear resample`, then PReLU.
///
/// ⚠️ Upstream uses `nn.UpsamplingBilinear2d`, which is `Upsample(mode: .bilinear,
/// alignCorners: TRUE)` — not the `alignCorners: false` used by the sibling FFTformer port. Both
/// the ×0.5 downsample and the ×2 upsample take the align-corners-true grid.
public final class NormDownsample: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "down") public var down: [Module]      // [Conv2d, Upsample] — Sequential
    @ModuleInfo(key: "prelu") public var prelu: PReLU

    private let resample = Upsample(scaleFactor: 0.5, mode: .linear(alignCorners: true))

    public init(_ inCh: Int, _ outCh: Int) {
        self._down.wrappedValue = [
            Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: 3, padding: 1, bias: false)
        ]
        self._prelu.wrappedValue = PReLU(count: 1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let conv = down[0] as? Conv2d else { return x }
        return prelu(resample(conv(x)))
    }
}

/// `Conv2d → ×2 bilinear → concat(skip) → 1×1 Conv2d → PReLU`.
public final class NormUpsample: Module, @unchecked Sendable {
    @ModuleInfo(key: "up_scale") public var upScale: [Module]   // [Conv2d, Upsample]
    @ModuleInfo(key: "up") public var up: Conv2d
    @ModuleInfo(key: "prelu") public var prelu: PReLU

    private let resample = Upsample(scaleFactor: 2.0, mode: .linear(alignCorners: true))

    public init(_ inCh: Int, _ outCh: Int) {
        self._upScale.wrappedValue = [
            Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: 3, padding: 1, bias: false)
        ]
        self._up.wrappedValue = Conv2d(
            inputChannels: outCh * 2, outputChannels: outCh, kernelSize: 1, bias: false)
        self._prelu.wrappedValue = PReLU(count: 1)
    }

    public func callAsFunction(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        guard let conv = upScale[0] as? Conv2d else { return x }
        let scaled = resample(conv(x))
        return prelu(up(concatenated([scaled, y], axis: -1)))
    }
}

/// Cross-Attention Block — attention over the **channel** dimension (Restormer-MDTA style), so
/// there is no windowing and the attention matrix is `(C/heads)²`, independent of resolution.
///
/// `q` comes from `x`, `k`/`v` from `y`; `q` and `k` are L2-normalized along the *spatial* axis and
/// scaled by a learned per-head temperature.
public final class CAB: Module, @unchecked Sendable {
    @ParameterInfo(key: "temperature") public var temperature: MLXArray
    @ModuleInfo(key: "q") public var q: Conv2d
    @ModuleInfo(key: "q_dwconv") public var qDW: Conv2d
    @ModuleInfo(key: "kv") public var kv: Conv2d
    @ModuleInfo(key: "kv_dwconv") public var kvDW: Conv2d
    @ModuleInfo(key: "project_out") public var projectOut: Conv2d

    private let heads: Int

    public init(dim: Int, heads: Int, bias: Bool = false) {
        self.heads = heads
        self._temperature.wrappedValue = MLXArray.ones([heads, 1, 1])
        self._q.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1, bias: bias)
        self._qDW.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3,
                                        padding: 1, groups: dim, bias: bias)
        self._kv.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim * 2, kernelSize: 1, bias: bias)
        self._kvDW.wrappedValue = Conv2d(inputChannels: dim * 2, outputChannels: dim * 2, kernelSize: 3,
                                         padding: 1, groups: dim * 2, bias: bias)
        self._projectOut.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim,
                                               kernelSize: 1, bias: bias)
    }

    /// `(B,H,W,C)` → `(B, heads, C/heads, H·W)`.
    ///
    /// Upstream's `rearrange(x, 'b (head c) h w -> b head c (h w)')` splits the CHANNEL axis into
    /// `(head, c)` with head outermost. In NHWC the channel axis is last, so the equivalent is a
    /// reshape to `(B, H·W, heads, c)` followed by a transpose — going via NCHW would be wrong,
    /// because it would interleave the head split against the spatial flattening.
    private func headSplit(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped(b, h * w, heads, c / heads).transposed(0, 2, 3, 1)
    }

    public func callAsFunction(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))

        let qh = headSplit(qDW(q(x)))
        let kvOut = kvDW(kv(y)).split(parts: 2, axis: -1)
        let kh = headSplit(kvOut[0])
        let vh = headSplit(kvOut[1])

        // L2 normalize along the spatial axis (upstream `normalize(..., dim=-1)`).
        func l2(_ t: MLXArray) -> MLXArray {
            t / MLX.sqrt(MLX.sum(t * t, axis: -1, keepDims: true) + 1e-12)
        }
        let attn = MLX.softmax(
            MLX.matmul(l2(qh), l2(kh).transposed(0, 1, 3, 2)) * temperature, axis: -1)

        let out = MLX.matmul(attn, vh)                       // (B, heads, c/heads, H·W)
        let merged = out.transposed(0, 3, 1, 2).reshaped(b, h, w, c)
        return projectOut(merged)
    }
}

/// Intensity Enhancement Layer — a gated FFN with two Tanh-residual depthwise branches.
public final class IEL: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "project_in") public var projectIn: Conv2d
    @ModuleInfo(key: "dwconv") public var dwconv: Conv2d
    @ModuleInfo(key: "dwconv1") public var dwconv1: Conv2d
    @ModuleInfo(key: "dwconv2") public var dwconv2: Conv2d
    @ModuleInfo(key: "project_out") public var projectOut: Conv2d

    public init(dim: Int, ffnExpansionFactor: Float = 2.66, bias: Bool = false) {
        let hidden = Int(Float(dim) * ffnExpansionFactor)     // int() truncation, as upstream
        self._projectIn.wrappedValue = Conv2d(inputChannels: dim, outputChannels: hidden * 2,
                                              kernelSize: 1, bias: bias)
        self._dwconv.wrappedValue = Conv2d(inputChannels: hidden * 2, outputChannels: hidden * 2,
                                           kernelSize: 3, padding: 1, groups: hidden * 2, bias: bias)
        self._dwconv1.wrappedValue = Conv2d(inputChannels: hidden, outputChannels: hidden,
                                            kernelSize: 3, padding: 1, groups: hidden, bias: bias)
        self._dwconv2.wrappedValue = Conv2d(inputChannels: hidden, outputChannels: hidden,
                                            kernelSize: 3, padding: 1, groups: hidden, bias: bias)
        self._projectOut.wrappedValue = Conv2d(inputChannels: hidden, outputChannels: dim,
                                               kernelSize: 1, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = dwconv(projectIn(x)).split(parts: 2, axis: -1)
        let x1 = MLX.tanh(dwconv1(parts[0])) + parts[0]
        let x2 = MLX.tanh(dwconv2(parts[1])) + parts[1]
        return projectOut(x1 * x2)
    }
}

/// HV-branch Lightweight Cross Attention.
///
/// ⚠️ Differs from ``ILCA`` in one line: the `gdfn` result **replaces** `x` here, whereas `I_LCA`
/// adds it as a residual. Easy to normalize away by accident; it is not a typo upstream.
public final class HVLCA: Module, @unchecked Sendable {
    @ModuleInfo(key: "gdfn") public var gdfn: IEL
    @ModuleInfo(key: "norm") public var norm: ChannelLayerNorm
    @ModuleInfo(key: "ffn") public var ffn: CAB

    public init(dim: Int, heads: Int) {
        self._gdfn.wrappedValue = IEL(dim: dim)
        self._norm.wrappedValue = ChannelLayerNorm(dim)
        self._ffn.wrappedValue = CAB(dim: dim, heads: heads)
    }

    public func callAsFunction(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        let a = x + ffn(norm(x), norm(y))
        return gdfn(norm(a))                    // NO residual — see the note above
    }
}

/// I-branch Lightweight Cross Attention.
public final class ILCA: Module, @unchecked Sendable {
    @ModuleInfo(key: "norm") public var norm: ChannelLayerNorm
    @ModuleInfo(key: "gdfn") public var gdfn: IEL
    @ModuleInfo(key: "ffn") public var ffn: CAB

    public init(dim: Int, heads: Int) {
        self._norm.wrappedValue = ChannelLayerNorm(dim)
        self._gdfn.wrappedValue = IEL(dim: dim)
        self._ffn.wrappedValue = CAB(dim: dim, heads: heads)
    }

    public func callAsFunction(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        let a = x + ffn(norm(x), norm(y))
        return a + gdfn(norm(a))                // WITH residual
    }
}
