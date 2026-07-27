// swift-tools-version: 6.2
import PackageDescription

// mlx-cidnet-swift — HVI-CIDNet low-light enhancement for MLXEngine. ONE repo, TWO products:
//   • CIDNetMLXCore — engine-agnostic Swift/MLX core (no MLXToolKit dep; usable standalone)
//   • MLXCIDNet     — the MLXEngine ModelPackage over that core
// Upstream: Fediory/HVI-CIDNet (MIT code and weights). See PORT-STATUS.md.
let package = Package(
    name: "mlx-cidnet-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "CIDNetMLXCore", targets: ["CIDNetMLXCore"]),
        .library(name: "MLXCIDNet", targets: ["MLXCIDNet"]),
        .executable(name: "cidnet-gate", targets: ["CIDNetGate"]),
    ],
    dependencies: [
        // 0.38.0 = contract 1.29.0, which introduces the `imageRelight` capability.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.38.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/xocialize/mlx-profiling.git", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "CIDNetMLXCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "MLXCIDNet",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                "CIDNetMLXCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "CIDNetMLXTests",
            dependencies: [
                "CIDNetMLXCore",
                "MLXCIDNet",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            resources: [.copy("Resources/goldens")]
        ),
        // Parity gates need a real Metal context — executable lane, not the test target.
        .executableTarget(
            name: "CIDNetGate",
            dependencies: [
                "CIDNetMLXCore",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/Gate",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
