// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ModelRuntimeChecks",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "2.29.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.29.1"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.1.6")
    ],
    targets: [.executableTarget(name: "ModelRuntimeChecks", dependencies: [
        .product(name: "MLXVLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift")
    ], path: ".", sources: ["main.swift", "MemoryEfficientIdefics3.swift"])])
