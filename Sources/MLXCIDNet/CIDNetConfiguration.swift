import Foundation
import MLXToolKit

/// An HVI-CIDNet checkpoint this package can load.
///
/// The authors publish ten; these are the three that matter for a product, and they behave
/// **differently enough that the choice is a product decision, not a tuning knob** — see
/// `PORT-STATUS.md` for the measured exposure ladder.
public enum CIDNetVariant: String, Codable, Sendable, CaseIterable {
    /// Trained for cross-dataset generalization. **The default.** Does the least damage when the
    /// luma gate mis-fires (23.37 dB on an already-correct exposure, vs 20.99 / 16.10 for the
    /// others), which matters more in production than peak LOL benchmark score.
    case generalization
    /// The headline LOL-v1 model (with perceptual loss) — strongest lift on *severely*
    /// underexposed input (+14.50 dB at 0.15× exposure) and the best published LOL numbers.
    case lolv1
    /// Trained on SICE (a multi-exposure set). The most aggressive auto-exposure normalizer: its
    /// output mean is near-constant across a 12× input range, so it will **darken** a bright image
    /// by design. Useful when you want a consistent target exposure, not when you want a gentle lift.
    case sice

    public var repo: String {
        switch self {
        case .generalization: return "mlx-community/HVI-CIDNet-Generalization-fp32"
        case .lolv1: return "mlx-community/HVI-CIDNet-LOLv1-fp32"
        case .sice: return "mlx-community/HVI-CIDNet-SICE-fp32"
        }
    }

    /// 1,975,569 params at fp32 is only 7.9 MB — there is nothing to gain from a smaller dtype, and
    /// the model's job (redistributing luminance) is precision-sensitive at the low end.
    public var quant: Quant { .fp32 }
}

/// Init-time configuration for `CIDNetRelightPackage` (C9).
public struct CIDNetConfiguration: PackageConfiguration, ModelStorable {
    public var variant: CIDNetVariant

    /// Mean-luma threshold below which the model runs. Above it the request is **bypassed**.
    ///
    /// This gate is not optional polish — it is the ship-blocker. Every published checkpoint drives
    /// its output toward a target mean luma *regardless of input*, so on an already-correctly-exposed
    /// image the model degrades it. Measured at correct exposure: Generalization 23.37 dB,
    /// LOLv1 20.99 dB, SICE 16.10 dB.
    ///
    /// ⚠️ The default is a **conservative placeholder**, not a calibrated value. It was chosen from a
    /// synthetic linear-scaling exposure ladder, which lacks sensor noise and clipping asymmetry.
    /// Calibrate against real underexposed captures before trusting it — see PORT-STATUS.md.
    public var lumaGateThreshold: Float

    /// Where downloadable weights are materialized (engine-supplied). Excluded from `Codable`.
    public var modelsRootDirectory: URL?

    /// Explicit local weights file, bypassing the store — for parity work and pre-publication runs.
    public var weightsURL: URL?

    public init(variant: CIDNetVariant = .generalization,
                lumaGateThreshold: Float = 0.35,
                modelsRootDirectory: URL? = nil,
                weightsURL: URL? = nil) {
        self.variant = variant
        self.lumaGateThreshold = lumaGateThreshold
        self.modelsRootDirectory = modelsRootDirectory
        self.weightsURL = weightsURL
    }

    private enum CodingKeys: String, CodingKey {
        case variant, lumaGateThreshold
    }
}

extension CIDNetConfiguration: QuantConfigured {
    public var quant: Quant { variant.quant }
}

extension CIDNetConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "weights", repo: variant.repo, revision: nil,
                      matching: ["model.safetensors"])]
    }

    /// Honors the explicit `weightsURL` first, then the default store probe (MS-2).
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let weightsURL, FileManager.default.fileExists(atPath: weightsURL.path) { return [] }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}
