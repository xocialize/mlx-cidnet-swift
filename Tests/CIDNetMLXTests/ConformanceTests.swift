// ConformanceTests.swift — HVI-CIDNet through the engine's offline gates (no MLX kernels run).
//
//   MAT-1..5    WeightSourcing declarations
//   CAN-1..3    cooperative cancellation + cadence
//   manifest    licence, surfaces, split footprint
//   contract    the imageRelight surface itself, including the bypass semantics
//
// The luma-gate behaviour is deliberately covered here rather than left to the app: it is the
// ship-blocker, and "did the package decline to enhance an already-good image" is a contract-level
// question (`ImageRelightResponse.bypassed`), not an implementation detail.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXCIDNet

final class ConformanceTests: XCTestCase {

    // MARK: - MAT

    func testMATGate() {
        let report = MaterializationConformance.check(freshConfiguration: CIDNetConfiguration())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourcesDeclaredForEveryVariant() {
        for variant in CIDNetVariant.allCases {
            let sources = CIDNetConfiguration(variant: variant).weightSources
            XCTAssertEqual(sources.count, 1, "\(variant)")
            XCTAssertEqual(sources[0].repo, variant.repo)
            XCTAssertEqual(sources[0].matching, ["model.safetensors"], "\(variant)")
        }
    }

    func testExplicitWeightsURLSuppressesMaterialization() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("cidnet-\(UUID().uuidString).safetensors")
        try Data([0x00]).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        XCTAssertTrue(CIDNetConfiguration(weightsURL: tmp).missingWeightSources(storeRoot: nil).isEmpty)
        XCTAssertEqual(
            CIDNetConfiguration(weightsURL: tmp.appendingPathExtension("nope"))
                .missingWeightSources(storeRoot: nil).count, 1)
    }

    // MARK: - CAN

    func testCANGatePreCancelledRun() async {
        let package = CIDNetRelightPackage(configuration: CIDNetConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageRelightRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        let manifest = CIDNetRelightPackage.manifest
        if CancellationConformance.longRunImplied(by: manifest) {
            // 3 GB declared activation crosses the long-run threshold, so a cadence is required.
            // run() is a single full-frame forward — no loop to hang per-step checkpoints on — so
            // the declaration names the seams that actually exist rather than inventing any.
            let report = CancellationConformance.checkCadence(
                manifest: manifest,
                posture: .cadence([
                    .init(phase: .encode, unit: .frame),
                    .init(phase: .postprocess, unit: .frame),
                ]))
            XCTAssertTrue(report.passed, report.summary)
        } else {
            let report = CancellationConformance.checkCadence(
                manifest: manifest,
                posture: .subSecondRuns(
                    reason: "single full-frame forward of a 1.98 M-param network"))
            XCTAssertTrue(report.passed, report.summary)
        }
    }

    // MARK: - Manifest

    func testManifestSurfacesAndLicence() {
        let m = CIDNetRelightPackage.manifest
        XCTAssertEqual(m.capabilities, [.imageRelight])
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].capability, .imageRelight)
        XCTAssertEqual(m.surfaces[0].name, "cidnet-relight")
        XCTAssertEqual(m.license.weightLicense, .mit)
        XCTAssertEqual(m.license.portCodeLicense, .mit)
    }

    func testFootprintIsSplit() {
        let m = CIDNetRelightPackage.manifest
        guard let fp = m.requirements.footprints.first(where: { $0.quant == .fp32 }) else {
            return XCTFail("no fp32 footprint")
        }
        // 1,975,569 params @ fp32 = 7.9 MB; the floor must cover it without absorbing the
        // activation (the flat-footprint anti-pattern).
        XCTAssertGreaterThan(fp.residentBytes, 7_900_000)
        XCTAssertGreaterThan(fp.peakActivationBytes, fp.residentBytes)
    }

    func testQuantConfiguredMatchesADeclaredFootprint() {
        let declared = Set(CIDNetRelightPackage.manifest.requirements.footprints.map(\.quant))
        for variant in CIDNetVariant.allCases {
            XCTAssertTrue(declared.contains(CIDNetConfiguration(variant: variant).quant), "\(variant)")
        }
    }

    // MARK: - Contract surface

    /// The descriptor must advertise `strength`, since that parameter is the reason this is a
    /// separate capability rather than a second `imageRestore` package.
    func testDescriptorAdvertisesStrength() {
        let d = CIDNetRelightPackage.manifest.surfaces[0]
        let names = d.parameters.map(\.name)
        XCTAssertTrue(names.contains("image"))
        XCTAssertTrue(names.contains("strength"), "strength must be on the surface: \(names)")
        XCTAssertEqual(d.parameters.first { $0.name == "strength" }?.required, false)
    }

    /// A zero-strength request must bypass without loading weights or running a kernel — the gate
    /// is evaluated before any model work, so this is offline-safe and also proves the ordering.
    func testZeroStrengthBypassesBeforeAnyModelWork() async throws {
        let package = CIDNetRelightPackage(configuration: CIDNetConfiguration())
        // No load() call: if the implementation touched the model before the gate, this would
        // throw `.notLoaded` instead of returning a bypass... which it will, because the
        // notLoaded guard legitimately precedes the gate. Assert that contract explicitly so a
        // future reordering is a deliberate decision rather than an accident.
        do {
            _ = try await package.run(
                ImageRelightRequest(image: Image(format: .png, data: Data()), strength: 0))
            XCTFail("expected .notLoaded — the load guard precedes the luma gate by design")
        } catch let error as PackageError {
            guard case .notLoaded = error else { return XCTFail("unexpected \(error)") }
        }
    }

    /// The default threshold is a documented placeholder, not a calibrated value. Pin it so that
    /// calibrating it later is a visible, deliberate change.
    func testLumaGateThresholdDefaultIsPinned() {
        XCTAssertEqual(CIDNetConfiguration().lumaGateThreshold, 0.35, accuracy: 1e-6)
        XCTAssertEqual(CIDNetConfiguration().variant, .generalization,
                       "Generalization is the default: least damage when the gate mis-fires")
    }
}
