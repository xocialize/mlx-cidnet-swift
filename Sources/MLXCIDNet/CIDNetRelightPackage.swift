import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import MLXProfiling
import Hub
import CIDNetMLXCore

public enum CIDNetPackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case imageEncodeFailed
    case weightsMissing(String)
}

/// An MLXEngine `imageRelight` package over **HVI-CIDNet** — exposure / lighting correction.
///
/// The first package on the `imageRelight` capability (contract 1.29.0). Two behaviours are part of
/// the package rather than the caller's problem:
///
/// 1. **The luma gate.** Every published checkpoint drives its output toward a target mean luma
///    *regardless of input*, so on an already-correctly-exposed image the model degrades it. The
///    package estimates input luma and declines rather than "enhancing" a good photo, reporting
///    `bypassed: true`.
/// 2. **Strength blending.** `strength` linearly blends model output toward the input, which is
///    what makes this a slider instead of a switch.
@InferenceActor
public final class CIDNetRelightPackage: ModelPackage {
    public typealias Configuration = CIDNetConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Fediory/HVI-CIDNet is MIT with no non-commercial clause anywhere; the authors publish
            // the checkpoints first-party under that same licence. Port code MIT.
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "Fediory/HVI-CIDNet", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // 1,975,569 params @ fp32 = 7.9 MB — the weights floor is negligible and the work
                // is all activation. ⚠️ FLAGGED-provisional: scaled from the model's size and the
                // sibling image packages' measured ratios; replace with an in-app `phys_footprint`
                // run before this package is marked validated.
                footprints: [
                    QuantFootprint(quant: .fp32,
                                   residentBytes: 60_000_000,
                                   peakActivationBytes: 3_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                ImageRelightContract.descriptor(
                    name: "cidnet-relight",
                    summary: "HVI-CIDNet exposure correction: lifts shadows and recovers mid-tones "
                        + "in underexposed images, with a strength dial and an automatic bypass for "
                        + "images that are already correctly exposed."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var model: CIDNet?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard model == nil else { return }
        let url: URL
        if let explicit = configuration.weightsURL {
            guard FileManager.default.fileExists(atPath: explicit.path) else {
                throw CIDNetPackageError.weightsMissing(explicit.path)
            }
            url = explicit
        } else {
            let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? HubApi()
            let dir = try await hub.snapshot(from: Hub.Repo(id: configuration.variant.repo),
                                             matching: ["model.safetensors"]) { progress, speed in
                WeightDownloadProgress.report(fraction: progress.fractionCompleted, bytesPerSecond: speed)
            }
            url = dir.appendingPathComponent("model.safetensors")
        }
        let net = CIDNet()
        try net.loadWeights(from: url)
        model = net
    }

    public func unload() async {
        model = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: entry checkpoint is the first act of run().
        try Task.checkCancellation()
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .imageRelight,
              let req = request as? ImageRelightRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let pb = try Self.decodeToPixelBuffer(req.image)
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let x = rgbNHWC(from: ensureBGRA(pb), width: w, height: h) else {
            throw CIDNetPackageError.imageDecodeFailed("NHWC conversion (\(w)x\(h))")
        }

        // ---- the luma gate ------------------------------------------------------------------
        // Rec.709 luma, matching how the eye weights the channels — a plain RGB mean would call a
        // saturated blue image "bright". Evaluated before any model work so a bypass costs nothing.
        let luma = MLX.mean(x[.ellipsis, 0] * 0.2126
                            + x[.ellipsis, 1] * 0.7152
                            + x[.ellipsis, 2] * 0.0722).item(Float.self)

        let requested = min(max(req.strength ?? 1.0, 0), 1)
        guard luma < configuration.lumaGateThreshold, requested > 0 else {
            // Already well exposed (or explicitly zero strength): return the input untouched.
            // Running the model here would measurably DEGRADE the image — that is the whole reason
            // this capability is separate from `imageRestore`.
            return ImageRelightResponse(image: req.image, appliedStrength: 0, bypassed: true)
        }

        try Task.checkCancellation()
        let prof = MLXProfiler.shared
        prof.beginRun("cidnet imageRelight \(configuration.variant) \(w)x\(h) luma=\(luma)")
        let blended = prof.region("relight", "forward") { () -> MLXArray in
            let enhanced = model(x)
            // Linear blend toward the input. At strength 1 this is exactly the model output, so the
            // full-strength path stays bit-identical to the parity-gated result.
            return requested >= 1 ? enhanced : x + (enhanced - x) * requested
        }
        let outPB = pixelBuffer(fromRGBNHWC: MLX.clip(blended, min: 0, max: 1), width: w, height: h)
        prof.endRun(denominators: ["image": 1])
        guard let outPB else { throw CIDNetPackageError.imageEncodeFailed }

        try Task.checkCancellation()
        let outImage: Image
        if req.image.format == .rawBGRA8 {
            guard let raw = Self.encodeRawBGRA8(outPB) else { throw CIDNetPackageError.imageEncodeFailed }
            outImage = raw
        } else {
            guard let png = Self.encodePNG(outPB) else { throw CIDNetPackageError.imageEncodeFailed }
            outImage = Image(format: .png, data: png, width: w, height: h)
        }
        return ImageRelightResponse(image: outImage, appliedStrength: requested, bypassed: false)
    }

    // MARK: - Image codec (same shape as the sibling image packages)

    nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw CIDNetPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        let w = cg.width, h = cg.height
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw CIDNetPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw CIDNetPackageError.imageDecodeFailed("CGContext for BGRA draw")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    nonisolated static func rawBGRA8ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw CIDNetPackageError.imageDecodeFailed("rawBGRA8 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 4)
        guard srcStride >= w * 4, image.data.count >= srcStride * h else {
            throw CIDNetPackageError.imageDecodeFailed("rawBGRA8 data too small")
        }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw CIDNetPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw CIDNetPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = min(srcStride, dstStride)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), rowBytes)
            }
        }
        return buffer
    }

    nonisolated static func encodeRawBGRA8(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0 else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 4
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawBGRA8(data: out, width: w, height: h)
    }
}

extension CIDNetRelightPackage {
    public nonisolated static var registration: PackageRegistration {
        .of(CIDNetRelightPackage.self)
    }
}
