// OddSizeTests.swift — any H × W runs, not only multiples of 8 (AB-T-0223).
//
// The UNet halves the resolution three times and concatenates each ×2 upsample with its skip, so a
// 570×320 image hit a concatenate shape mismatch, which mlx-swift turns into `fatalError`: the whole
// host process died. The forward now reflect-pads to the next multiple of 8 and crops back.
//
// Offline and CPU-pinned: randomly initialised weights, no download, no Metal. The end-to-end case
// goes through the package's run() with a 570×321 PNG, the shape of the live repro.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import MLXToolKit
import UniformTypeIdentifiers
import XCTest
@testable import CIDNetMLXCore
@testable import MLXCIDNet

final class OddSizeTests: XCTestCase {

    // Odd in both axes, and neither a multiple of 8.
    private let (width, height) = (570, 321)

    /// Runs `body` with MLX pinned to the CPU. `withDefaultDevice` alone scopes only the Swift-side
    /// default stream; mlx core internals still use the global C++ default device, so that is set
    /// too, after latching the Swift default so the process isn't left on the CPU.
    private func withMLXCPU<R>(_ body: () async throws -> R) async rethrows -> R {
        _ = Device.defaultDevice()
        Device.setDefault(device: Device(.cpu))
        defer { Device.setDefault(device: Device(.gpu)) }
        return try await Device.withDefaultDevice(Device(.cpu), body)
    }

    // MARK: - Padding geometry

    func testAlignedExtent() {
        XCTAssertEqual(CIDNet.alignedExtent(570), 576)
        XCTAssertEqual(CIDNet.alignedExtent(321), 328)
        XCTAssertEqual(CIDNet.alignedExtent(320), 320, "an aligned extent is not padded")
        XCTAssertEqual(CIDNet.alignedExtent(1), 8)
    }

    /// torch `F.pad(..., 'reflect')`, which upstream's eval uses: mirror about the last sample
    /// without repeating it.
    func testReflectIndicesMatchTorch() {
        let indices = CIDNet.reflectIndices(570, extent: 576)
        XCTAssertEqual(Array(indices.prefix(570)), (0 ..< 570).map(Int32.init))
        XCTAssertEqual(Array(indices.suffix(6)), [568, 567, 566, 565, 564, 563])
        // Shorter than its pad, where torch refuses: fold again at the first sample.
        XCTAssertEqual(CIDNet.reflectIndices(3, extent: 8), [0, 1, 2, 1, 0, 1, 2, 1])
        XCTAssertEqual(CIDNet.reflectIndices(1, extent: 8), [Int32](repeating: 0, count: 8))
    }

    // MARK: - Core forward

    func testOddSizeForwardKeepsExtent() async {
        await withMLXCPU {
            let net = CIDNet()
            let x = MLXRandom.uniform(low: 0, high: 0.3, [1, height, width, 3])
            let out = net(x)
            eval(out)
            XCTAssertEqual(out.shape, [1, height, width, 3])
            XCTAssertFalse(MLX.isNaN(out).any().item(Bool.self), "NaN in the output")
        }
    }

    // MARK: - Package end to end

    func testPackageRelightsOddSizePNG() async throws {
        let weights = FileManager.default.temporaryDirectory
            .appending(path: "cidnet-oddsize-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: weights) }

        try await withMLXCPU {
            let random = CIDNet()
            try MLX.save(arrays: Dictionary(uniqueKeysWithValues: random.parameters().flattened()), url: weights)

            let package = CIDNetRelightPackage(configuration: CIDNetConfiguration(weightsURL: weights))
            try await package.load()
            let response = try await package.run(ImageRelightRequest(image: try darkPNG()))

            guard let relit = response as? ImageRelightResponse else {
                return XCTFail("unexpected response \(type(of: response))")
            }
            XCTAssertFalse(relit.bypassed, "a dark image must reach the model, not the luma gate's bypass")
            XCTAssertEqual(relit.image.format, .png)
            XCTAssertEqual(relit.image.width, width)
            XCTAssertEqual(relit.image.height, height)
            let decoded = try XCTUnwrap(CGImageSourceCreateWithData(relit.image.data as CFData, nil)
                .flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) })
            XCTAssertEqual(decoded.width, width)
            XCTAssertEqual(decoded.height, height)
        }
    }

    /// A dark gradient, well under the luma gate's 0.35.
    private func darkPNG() throws -> Image {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let i = (y * width + x) * 4
                pixels[i + 0] = UInt8(10 + 40 * x / width)    // R
                pixels[i + 1] = UInt8(10 + 40 * y / height)   // G
                pixels[i + 2] = 20                            // B
            }
        }
        let cg: CGImage = try pixels.withUnsafeMutableBytes { buffer in
            let ctx = try XCTUnwrap(CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
            return try XCTUnwrap(ctx.makeImage())
        }
        let out = NSMutableData()
        let dest = try XCTUnwrap(CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, cg, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        return Image(format: .png, data: out as Data, width: width, height: height)
    }
}
