//
//  main.swift
//  mlx-cidnet-swift / CIDNetGate
//
//  Parity gates against the PyTorch oracle. Executable, not a test target — the SPM test product's
//  metallib is unreliable for GPU work, while `swift run` is fine.
//
//  Modes:
//    --s0   <weights>              key contract
//    --hvi  <goldens> <weights>    the HVI colour transform (the bit-exact component)
//    --s1   <goldens> <weights>    primitives
//    --s2   <goldens> <weights>    blocks
//    --s3   <goldens> <weights>    full model
//    --all  <goldens> <weights>
//

import Foundation
import CIDNetMLXCore
import MLX
import MLXNN

private let _unbuffered: Void = { setvbuf(stdout, nil, _IONBF, 0) }()

func fail(_ msg: String) -> Never {
    _ = _unbuffered
    print("❌ \(msg)")
    exit(1)
}

func loadedModel(_ path: String) -> CIDNet {
    let model = CIDNet()
    do { try model.loadWeights(from: URL(fileURLWithPath: path)) }
    catch { fail("weight load failed: \(error)") }
    return model
}

func g(_ dir: String, _ name: String) -> MLXArray {
    do { return try loadNPY("\(dir)/\(name).npy") }
    catch { fail("golden \(name): \(error)") }
}

/// S0 — module tree vs checkpoint. Runs no kernel.
func gateS0(_ weightsPath: String) {
    _ = _unbuffered
    print("=== S0 · key contract ===\n")
    let model = CIDNet()
    var swiftKeys: [String: [Int]] = [:]
    var total = 0
    for (k, v) in model.parameters().flattened() { swiftKeys[k] = v.shape; total += v.size }
    print("Swift module tree : \(swiftKeys.count) tensors, \(total) params")

    guard let loaded = try? MLX.loadArrays(url: URL(fileURLWithPath: weightsPath)) else {
        fail("could not load \(weightsPath)")
    }
    let ckptTotal = loaded.values.reduce(0) { $0 + $1.size }
    print("Checkpoint        : \(loaded.count) tensors, \(ckptTotal) params\n")

    let sk = Set(swiftKeys.keys), ck = Set(loaded.keys)
    let missing = sk.subtracting(ck).sorted(), unused = ck.subtracting(sk).sorted()
    if !missing.isEmpty {
        print("MISSING (\(missing.count)):"); missing.prefix(20).forEach { print("   \($0)  \(swiftKeys[$0]!)") }
    }
    if !unused.isEmpty {
        print("UNUSED (\(unused.count)):"); unused.prefix(20).forEach { print("   \($0)  \(loaded[$0]!.shape)") }
    }
    var mismatch: [(String, [Int], [Int])] = []
    for k in sk.intersection(ck) where swiftKeys[k]! != loaded[k]!.shape {
        mismatch.append((k, swiftKeys[k]!, loaded[k]!.shape))
    }
    if !mismatch.isEmpty {
        print("SHAPE MISMATCH (\(mismatch.count)):")
        for (k, a, b) in mismatch.prefix(20) { print("   \(k)\n     swift \(a) vs ckpt \(b)") }
    }
    guard missing.isEmpty, unused.isEmpty, mismatch.isEmpty else { fail("S0 FAILED") }
    do { try model.update(parameters: ModuleParameters.unflattened(loaded), verify: .all) }
    catch { fail("S0 FAILED at update(verify: .all): \(error)") }
    print("✅ S0 PASSED — \(swiftKeys.count) tensors, \(total) params, strict update clean.")
}

/// The HVI colour transform — the component the port queue requires to be bit-exact.
func gateHVI(_ dir: String, _ weightsPath: String) -> Bool {
    print("=== HVI · colour transform (bit-exact target) ===\n")
    let r = GateReport("HVI")
    let model = loadedModel(weightsPath)
    let k = g(dir, "density_k")
    print("  density_k golden \(k.item(Float.self))  vs loaded \(model.trans.densityK.item(Float.self))\n")

    r.check("hvit", toNCHW(model.trans.forward(toNHWC(g(dir, "hvit_in")))),
            g(dir, "hvit_out"), tol: 1e-6)

    // Degenerate colours: black, white, grey, the three primaries, and two exact ties (yellow,
    // cyan) where the sequential-overwrite priority actually decides the answer.
    r.check("hvit_edge_cases", toNCHW(model.trans.forward(toNHWC(g(dir, "hvit_edge_in")))),
            g(dir, "hvit_edge_out"), tol: 1e-6)

    r.check("phvit", toNCHW(model.trans.inverse(toNHWC(g(dir, "phvit_in")))),
            g(dir, "phvit_out"), tol: 1e-6)

    let rt = model.trans.inverse(model.trans.forward(toNHWC(g(dir, "hvit_in"))))
    r.check("roundtrip", toNCHW(rt), g(dir, "roundtrip_out"), tol: 1e-6)

    return r.summarize()
}

/// Primitives.
func gateS1(_ dir: String, _ weightsPath: String) -> Bool {
    print("=== S1 · primitives ===\n")
    let r = GateReport("S1")
    let model = loadedModel(weightsPath)

    r.check("layernorm", toNCHW(model.hvLCA1.norm(toNHWC(g(dir, "layernorm_in")))),
            g(dir, "layernorm_out"), tol: 1e-6)
    r.check("prelu", toNCHW(model.hveBlock1.prelu(toNHWC(g(dir, "prelu_in")))),
            g(dir, "prelu_out"), tol: 1e-6)
    // NOTE: bilinear with alignCorners TRUE here (nn.UpsamplingBilinear2d), unlike the sibling
    // FFTformer port. Same tolerance rationale as there: interpolation weights are computed in
    // floating point, then a conv accumulates over the channels.
    r.check("downsample", toNCHW(model.hveBlock1(toNHWC(g(dir, "down_in")))),
            g(dir, "down_out"), tol: 1e-5)
    r.check("upsample", toNCHW(model.hvdBlock3(toNHWC(g(dir, "up_x")), toNHWC(g(dir, "up_y")))),
            g(dir, "up_out"), tol: 1e-5)
    return r.summarize()
}

/// Blocks.
func gateS2(_ dir: String, _ weightsPath: String) -> Bool {
    print("=== S2 · blocks ===\n")
    let r = GateReport("S2")
    let model = loadedModel(weightsPath)

    r.check("cab", toNCHW(model.hvLCA1.ffn(toNHWC(g(dir, "cab_x")), toNHWC(g(dir, "cab_y")))),
            g(dir, "cab_out"), tol: 1e-5)
    r.check("iel", toNCHW(model.hvLCA1.gdfn(toNHWC(g(dir, "iel_in")))),
            g(dir, "iel_out"), tol: 1e-5)
    let lx = toNHWC(g(dir, "lca_x")), ly = toNHWC(g(dir, "lca_y"))
    r.check("hv_lca", toNCHW(model.hvLCA1(lx, ly)), g(dir, "hv_lca_out"), tol: 1e-5)
    r.check("i_lca", toNCHW(model.iLCA1(lx, ly)), g(dir, "i_lca_out"), tol: 1e-5)
    return r.summarize()
}

/// Full model, including a severely underexposed image — the real use case.
func gateS3(_ dir: String, _ weightsPath: String) -> Bool {
    print("=== S3 · full model ===\n")
    let r = GateReport("S3")
    let model = loadedModel(weightsPath)
    for name in ["full_64", "full_128", "full_256", "full_dark256"] {
        let out = model(toNHWC(g(dir, "\(name)_in")))
        eval(out)
        r.check(name, toNCHW(out), g(dir, "\(name)_out"), tol: 1e-4)
    }
    return r.summarize()
}

// MARK: - entry

let args = Array(CommandLine.arguments.dropFirst())
guard let mode = args.first else {
    print("""
    usage:
      cidnet-gate --s0   <weights>
      cidnet-gate --hvi  <goldens> <weights>
      cidnet-gate --s1   <goldens> <weights>
      cidnet-gate --s2   <goldens> <weights>
      cidnet-gate --s3   <goldens> <weights>
      cidnet-gate --all  <goldens> <weights>
    """)
    exit(2)
}

// fp32 parity gates pin to the CPU stream — Apple-GPU fp32 accumulates enough per-op error to
// both mask real bugs and imitate them.
Device.setDefault(device: .cpu)

switch mode {
case "--s0":
    guard args.count >= 2 else { fail("--s0 needs a weights path") }
    gateS0(args[1])
case "--hvi", "--s1", "--s2", "--s3", "--all":
    guard args.count >= 3 else { fail("\(mode) needs <goldens> <weights>") }
    let (dir, w) = (args[1], args[2])
    var ok = true
    if mode == "--hvi" || mode == "--all" { ok = gateHVI(dir, w) && ok; print("") }
    if mode == "--s1" || mode == "--all" { ok = gateS1(dir, w) && ok; print("") }
    if mode == "--s2" || mode == "--all" { ok = gateS2(dir, w) && ok; print("") }
    if mode == "--s3" || mode == "--all" { ok = gateS3(dir, w) && ok }
    if !ok { exit(1) }
default:
    fail("unknown mode \(mode)")
}
