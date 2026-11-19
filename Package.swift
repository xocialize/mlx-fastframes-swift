// swift-tools-version: 6.2
import PackageDescription

// mlx-fastframes-swift — Apple FastFrames (arXiv 2611.23226; apple/ml-fastframes) autoregressive two-step
// text-to-video on MLX. FF-S: a frame-causal Wan2.1-1.3B DiT (Causal Forcing++ framewise-2step, Apache-2.0)
// plus Apple's KV refinement network and causal ConvNeXt pixel decoder (Apple AMLR — research only, Gaea plugin
// channel only per AB-D-0259; never a shipping app).
//   • FastFramesMLX — engine-agnostic core: the frame-causal DiT with a preallocated KV cache, the KV refiner, the
//     pixel decoder, the FlowMatch rollout, and umT5 prompt encoding through wan-core
//   • MLXFastFrames — the MLXEngine `textToVideo` ModelPackage (research-only licence declared; see README)
//   • ff-smoke — V2 parity gates against upstream's fp32 component outputs and the Python-MLX V0 rollout, the timed
//     full rollout, and an MP4 writer
let package = Package(
    name: "mlx-fastframes-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "FastFramesMLX", targets: ["FastFramesMLX"]),
        .library(name: "MLXFastFrames", targets: ["MLXFastFrames"]),
        .executable(name: "ff-smoke", targets: ["FastFramesSmoke"]),
    ],
    dependencies: [
        // The fleet's mlx-swift fork: 0.32.4 = upstream 0.32.3 + the multi-output-op leak fix (#501), the floor
        // mlx-engine-swift ≥ 0.74.1 requires (AB-A-0249); 0.33.1 = upstream 0.33.0 with the head-dim-split SDPA
        // routes disabled (AB-D-0264). Ceiling = wan-core's (< 0.34.0). Same package identity as ml-explore's URL,
        // and a root declaration wins, so wan-core's range resolves against the fork.
        .package(url: "https://github.com/xocialize/mlx-swift", "0.32.4" ..< "0.34.0"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.82.1"),
        .package(url: "https://github.com/xocialize/wan-core-mlx-swift.git", from: "0.6.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "FastFramesMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXFastFrames",
            dependencies: [
                "FastFramesMLX",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "FastFramesSmoke",
            dependencies: [
                "FastFramesMLX",
                "MLXFastFrames",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/Smoke",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "FastFramesMLXTests",
            dependencies: [
                "FastFramesMLX",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXFastFramesTests",
            dependencies: [
                "MLXFastFrames",
                "FastFramesMLX",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
