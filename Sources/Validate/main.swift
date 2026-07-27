//
//  main.swift
//  mlx-cidnet-swift / CIDNetValidate
//
//  Drives the package through the **real `MLXServeEngine`** and reports the authoritative split
//  footprint, so the manifest can stop carrying an estimate.
//
//  Why this exists rather than the gate's `--bench`: `--bench` calls the core model directly and
//  reads MLX-pool memory. Two problems. It bypasses register/prepare and the governor, and MLX-pool
//  peak **under-reads the admission basis by ~2.7×** — the BiRefNet re-baseline that forced every
//  optimizer-family manifest to be re-measured (BiRefNet best went 18.3 → 47.8 GB when measured
//  properly). The number that matters is process `phys_footprint`.
//
//  This target uses `MLXEngineTestKit.ValidationHarness`, which is the same code path and the same
//  metric the (now archived) MLXEngineImage validation app used: 150 ms `phys_footprint` sampling,
//  the resident floor read **post-load / pre-run**, and the `[label] SPLIT floor= peak= act=` line
//  the manifests were re-baselined from. `package-efficiency.md` explicitly blesses an
//  xcodebuild-or-`swift run` executable reading `phys_footprint` as equivalent to the in-app number.
//
//  ⚠️ One honest caveat: a CLI process carries no AppKit/Metal-view overhead, so absolute `floor`
//  and `peak` sit a few hundred MB below a GUI app's. That is conservative in the WRONG direction
//  for admission, so the manifest adds margin. `activation = peak − floor` — the dominant term for
//  a full-frame restorer — is process-shape independent.
//
//  Usage:  swift run cidnet-validate <weights.safetensors> [image.png] [width] [height]
//

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import MLXServeCore
import MLXEngineTestKit
import MLXCIDNet

setvbuf(stdout, nil, _IONBF, 0)

let args = Array(CommandLine.arguments.dropFirst())
guard let weightsPath = args.first else {
    print("usage: cidnet-validate <weights.safetensors> [image.png] [width] [height]")
    exit(2)
}
let imagePath = args.count > 1 ? args[1] : nil
let width = args.count > 2 ? Int(args[2]) ?? 1920 : 1920
let height = args.count > 3 ? Int(args[3]) ?? 1080 : 1080

/// A real PNG if one was supplied, else a synthetic structured image at the requested size.
/// Synthetic is fine for a MEMORY measurement — footprint depends on dimensions, not content — and
/// it keeps the envelope explicit rather than whatever a fixture happens to be.
func makeImage() -> MLXToolKit.Image {
    if let imagePath, let data = FileManager.default.contents(atPath: imagePath) {
        let src = CGImageSourceCreateWithData(data as CFData, nil)
        let cg = src.flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) }
        print("input: \(imagePath) (\(cg?.width ?? 0)x\(cg?.height ?? 0))")
        return MLXToolKit.Image(format: .png, data: data, width: cg?.width, height: cg?.height)
    }
    var bytes = [UInt8](repeating: 0, count: width * height * 4)
    for y in 0 ..< height {
        for x in 0 ..< width {
            let i = (y * width + x) * 4
            let fx = Float(x) / Float(width), fy = Float(y) / Float(height)
            // DARK on purpose: the package's luma gate bypasses anything at or above its
            // threshold (0.35 mean Rec.709 luma), so a mid-grey input would measure the bypass
            // path and report a floor with no model in it. 0.12x puts this firmly in the band
            // where the model actually runs.
            let dim: Float = 0.12
            bytes[i + 0] = UInt8(max(0, min(255, (0.5 + 0.3 * sin(fx * 11)) * dim * 255)))       // B
            bytes[i + 1] = UInt8(max(0, min(255, (0.5 + 0.3 * cos(fy * 9)) * dim * 255)))        // G
            bytes[i + 2] = UInt8(max(0, min(255, (0.5 + 0.3 * sin((fx + fy) * 7)) * dim * 255))) // R
            bytes[i + 3] = 255
        }
    }
    print("input: synthetic \(width)x\(height)")
    return MLXToolKit.Image.rawBGRA8(data: Data(bytes), width: width, height: height)
}

@MainActor
func main() async {
    // Match the shipping configuration: Forge constructs the engine `.blocking`, so a package that
    // would be refused in production must be refused here too.
    let engine = MLXServeEngine(policy: .permissiveOnly, licenseEnforcement: .blocking)

    let config = CIDNetConfiguration(weightsURL: URL(fileURLWithPath: weightsPath))
    let request = ImageRelightRequest(image: makeImage())

    do {
        let result = try await ValidationHarness.run(
            engine: engine,
            registration: CIDNetRelightPackage.registration,
            configuration: config,
            capability: .imageRelight,
            request: request,
            isolate: true,
            clearCache: { MLX.GPU.clearCache() },
            inputSummary: "\(width)x\(height)",
            heartbeatLabel: "cidnet")

        print("")
        print(result.run.splitLogLine("cidnet-\(config.variant.rawValue)"))
        print("")
        print("  DECLARE  residentBytes        = \(result.run.residentFloorBytes)")
        print("           peakActivationBytes  = \(result.run.activationBytes)")
        if result.run.retainedAfterRunBytes > 200_000_000 {
            print("  ⚠️ retains \(result.run.retainedAfterRunBytes) B after run+clearCache — "
                + "a live model holding intermediates. Belongs in the transient, not residency.")
        }
        if let out = result.response as? ImageRelightResponse {
            print("  output: \(out.image.data.count) bytes, \(out.image.width ?? 0)x\(out.image.height ?? 0)"
                + "  bypassed=\(out.bypassed) strength=\(out.appliedStrength)")
            if out.bypassed {
                print("  NOTE: the luma gate BYPASSED — the synthetic input is not dark enough. "
                    + "Re-run with a dark image to measure the model path.")
            }
        }
    } catch {
        print("❌ validation failed: \(error)")
        exit(1)
    }
}

await main()
