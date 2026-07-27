//
//  CIDNet.swift
//  mlx-cidnet-swift / CIDNetMLXCore
//
//  Role: MLX-Swift port of HVI-CIDNet — low-light enhancement / exposure correction.
//        CVPR 2025; 1st place, NTIRE 2025 Low-Light Enhancement.
//
//  Upstream: https://github.com/Fediory/HVI-CIDNet (MIT code and weights)
//  Weights:  Fediory/HVI-CIDNet-* (ten first-party checkpoints, PyTorchModelHubMixin)
//
//  Dual-branch UNet: the HV (chroma) branch and the I (intensity) branch exchange information at
//  every scale through lightweight cross-attention, then the result is added to the input HVI and
//  converted back to RGB. 1,975,569 parameters — 7.90 MB fp32.
//
//  Conventions: NHWC; module keys mirror the upstream state dict exactly.
//

import Foundation
import MLX
import MLXNN

public final class CIDNet: Module, @unchecked Sendable {

    public struct Configuration: Sendable {
        public var channels = [36, 36, 72, 144]
        public var heads = [1, 2, 4, 8]
        /// Upstream default and what every released checkpoint uses. When false the
        /// `NormDownsample`/`NormUpsample` LayerNorms are not constructed at all — which is why
        /// they are absent from the state dict.
        public var useNorm = false
        public init() {}
    }

    // HV (chroma) branch
    @ModuleInfo(key: "HVE_block0") public var hveBlock0: [Module]   // [ReplicationPad2d, Conv2d]
    @ModuleInfo(key: "HVE_block1") public var hveBlock1: NormDownsample
    @ModuleInfo(key: "HVE_block2") public var hveBlock2: NormDownsample
    @ModuleInfo(key: "HVE_block3") public var hveBlock3: NormDownsample
    @ModuleInfo(key: "HVD_block3") public var hvdBlock3: NormUpsample
    @ModuleInfo(key: "HVD_block2") public var hvdBlock2: NormUpsample
    @ModuleInfo(key: "HVD_block1") public var hvdBlock1: NormUpsample
    @ModuleInfo(key: "HVD_block0") public var hvdBlock0: [Module]

    // I (intensity) branch
    @ModuleInfo(key: "IE_block0") public var ieBlock0: [Module]
    @ModuleInfo(key: "IE_block1") public var ieBlock1: NormDownsample
    @ModuleInfo(key: "IE_block2") public var ieBlock2: NormDownsample
    @ModuleInfo(key: "IE_block3") public var ieBlock3: NormDownsample
    @ModuleInfo(key: "ID_block3") public var idBlock3: NormUpsample
    @ModuleInfo(key: "ID_block2") public var idBlock2: NormUpsample
    @ModuleInfo(key: "ID_block1") public var idBlock1: NormUpsample
    @ModuleInfo(key: "ID_block0") public var idBlock0: [Module]

    @ModuleInfo(key: "HV_LCA1") public var hvLCA1: HVLCA
    @ModuleInfo(key: "HV_LCA2") public var hvLCA2: HVLCA
    @ModuleInfo(key: "HV_LCA3") public var hvLCA3: HVLCA
    @ModuleInfo(key: "HV_LCA4") public var hvLCA4: HVLCA
    @ModuleInfo(key: "HV_LCA5") public var hvLCA5: HVLCA
    @ModuleInfo(key: "HV_LCA6") public var hvLCA6: HVLCA

    @ModuleInfo(key: "I_LCA1") public var iLCA1: ILCA
    @ModuleInfo(key: "I_LCA2") public var iLCA2: ILCA
    @ModuleInfo(key: "I_LCA3") public var iLCA3: ILCA
    @ModuleInfo(key: "I_LCA4") public var iLCA4: ILCA
    /// ⚠️ **DEAD IN THE FORWARD PASS.** Upstream computes `i_dec2 = I_LCA5(...)` and then
    /// immediately overwrites `i_dec2` with `ID_block2(i_dec3, v_jump1)` before any read, so its
    /// output reaches nothing. Verified by ablation against the released checkpoint: zeroing
    /// `I_LCA5`'s weights changes the output by exactly 0.0, while `HV_LCA5`, `I_LCA2` and
    /// `I_LCA6` all move it by ~0.5–1.0. Declared so strict weight loading stays clean; never
    /// evaluated, because evaluating it would cost time and change nothing.
    @ModuleInfo(key: "I_LCA5") public var iLCA5: ILCA
    @ModuleInfo(key: "I_LCA6") public var iLCA6: ILCA

    @ModuleInfo(key: "trans") public var trans: RGBHVI

    public init(_ cfg: Configuration = Configuration()) {
        let (ch1, ch2, ch3, ch4) = (cfg.channels[0], cfg.channels[1], cfg.channels[2], cfg.channels[3])
        let (h2, h3, h4) = (cfg.heads[1], cfg.heads[2], cfg.heads[3])

        // `nn.Sequential(ReplicationPad2d(1), Conv2d(k=3, padding=0))`. Replication padding — NOT
        // reflection, and not the conv's own zero padding. Index 1 is the conv, matching the
        // upstream Sequential key path.
        func stem(_ inCh: Int, _ outCh: Int) -> [Module] {
            [Identity(),
             Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: 3, padding: 0, bias: false)]
        }

        self._hveBlock0.wrappedValue = stem(3, ch1)
        self._hveBlock1.wrappedValue = NormDownsample(ch1, ch2)
        self._hveBlock2.wrappedValue = NormDownsample(ch2, ch3)
        self._hveBlock3.wrappedValue = NormDownsample(ch3, ch4)
        self._hvdBlock3.wrappedValue = NormUpsample(ch4, ch3)
        self._hvdBlock2.wrappedValue = NormUpsample(ch3, ch2)
        self._hvdBlock1.wrappedValue = NormUpsample(ch2, ch1)
        self._hvdBlock0.wrappedValue = stem(ch1, 2)

        self._ieBlock0.wrappedValue = stem(1, ch1)
        self._ieBlock1.wrappedValue = NormDownsample(ch1, ch2)
        self._ieBlock2.wrappedValue = NormDownsample(ch2, ch3)
        self._ieBlock3.wrappedValue = NormDownsample(ch3, ch4)
        self._idBlock3.wrappedValue = NormUpsample(ch4, ch3)
        self._idBlock2.wrappedValue = NormUpsample(ch3, ch2)
        self._idBlock1.wrappedValue = NormUpsample(ch2, ch1)
        self._idBlock0.wrappedValue = stem(ch1, 1)

        self._hvLCA1.wrappedValue = HVLCA(dim: ch2, heads: h2)
        self._hvLCA2.wrappedValue = HVLCA(dim: ch3, heads: h3)
        self._hvLCA3.wrappedValue = HVLCA(dim: ch4, heads: h4)
        self._hvLCA4.wrappedValue = HVLCA(dim: ch4, heads: h4)
        self._hvLCA5.wrappedValue = HVLCA(dim: ch3, heads: h3)
        self._hvLCA6.wrappedValue = HVLCA(dim: ch2, heads: h2)

        self._iLCA1.wrappedValue = ILCA(dim: ch2, heads: h2)
        self._iLCA2.wrappedValue = ILCA(dim: ch3, heads: h3)
        self._iLCA3.wrappedValue = ILCA(dim: ch4, heads: h4)
        self._iLCA4.wrappedValue = ILCA(dim: ch4, heads: h4)
        self._iLCA5.wrappedValue = ILCA(dim: ch3, heads: h3)
        self._iLCA6.wrappedValue = ILCA(dim: ch2, heads: h2)

        self._trans.wrappedValue = RGBHVI()
    }

    /// Replication pad by 1 on all sides — MLX has no native replication pad, and this is a
    /// clamp-to-edge gather on each axis.
    private func replicationPad1(_ x: MLXArray) -> MLXArray {
        let (h, w) = (x.dim(1), x.dim(2))
        let rows = MLXArray([Int32(0)] + (0 ..< h).map { Int32($0) } + [Int32(h - 1)])
        let cols = MLXArray([Int32(0)] + (0 ..< w).map { Int32($0) } + [Int32(w - 1)])
        return x.take(rows, axis: 1).take(cols, axis: 2)
    }

    private func stemForward(_ seq: [Module], _ x: MLXArray) -> MLXArray {
        guard let conv = seq[1] as? Conv2d else { return x }
        return conv(replicationPad1(x))
    }

    /// NHWC RGB in [0,1] → enhanced NHWC RGB.
    ///
    /// The variable names track `net/CIDNet.py` line for line, including the places upstream reads
    /// a *pre*-LCA tensor into the next downsample (`i_enc3 = IE_block3(i_enc2)`, not `i_enc3`) —
    /// those are load-bearing, since the LCA output survives only through the jump connections.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hvi = trans.forward(x)
        let i = hvi[.ellipsis, 2].expandedDimensions(axis: -1)

        let iEnc0 = stemForward(ieBlock0, i)
        let iEnc1 = ieBlock1(iEnc0)
        let hv0 = stemForward(hveBlock0, hvi)
        let hv1 = hveBlock1(hv0)

        let iJump0 = iEnc0
        let hvJump0 = hv0

        var iEnc2 = iLCA1(iEnc1, hv1)
        var hv2 = hvLCA1(hv1, iEnc1)
        let vJump1 = iEnc2
        let hvJump1 = hv2
        iEnc2 = ieBlock2(iEnc2)
        hv2 = hveBlock2(hv2)

        // The LCA2 outputs feed ONLY the jump connections — upstream then downsamples the
        // *pre*-LCA tensors. Reproduced exactly.
        let vJump2 = iLCA2(iEnc2, hv2)
        let hvJump2 = hvLCA2(hv2, iEnc2)
        let iEnc3 = ieBlock3(iEnc2)
        let hv3e = hveBlock3(hv2)

        let iEnc4 = iLCA3(iEnc3, hv3e)
        let hv4a = hvLCA3(hv3e, iEnc3)

        let iDec4 = iLCA4(iEnc4, hv4a)
        let hv4 = hvLCA4(hv4a, iEnc4)

        let hv3 = hvdBlock3(hv4, hvJump2)
        let iDec3 = idBlock3(iDec4, vJump2)

        // I_LCA5 is skipped: its result is overwritten before use upstream (ablation-verified).
        let hv2b = hvLCA5(hv3, iDec3)

        let hv2c = hvdBlock2(hv2b, hvJump1)
        let iDec2 = idBlock2(iDec3, vJump1)

        let iDec1a = iLCA6(iDec2, hv2c)
        let hv1b = hvLCA6(hv2c, iDec2)

        let iDec1 = idBlock1(iDec1a, iJump0)
        let iDec0 = stemForward(idBlock0, iDec1)
        let hv1c = hvdBlock1(hv1b, hvJump0)
        let hv0b = stemForward(hvdBlock0, hv1c)

        let outputHVI = concatenated([hv0b, iDec0], axis: -1) + hvi
        return trans.inverse(outputHVI)
    }

    /// Loads converted safetensors weights under the strict verifier.
    public func loadWeights(from url: URL) throws {
        let arrays = try MLX.loadArrays(url: url)
        try update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
        eval(self)
    }
}
