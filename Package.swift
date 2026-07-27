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
        .executable(name: "cidnet-gate", targets: ["CIDNetGate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
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
