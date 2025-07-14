# Poseidon CUDA Implementation

## Introduction

This is a high-performance CUDA implementation of the Poseidon hash function built on top of the Sppark framework. The implementation provides a complete Poseidon hash with configurable parameters, supporting both BLS12-377 and BLS12-381 scalar fields. The project extends beyond the initial requirement of implementing just the first round, delivering a full Poseidon hash implementation with optimized GPU kernels and Rust bindings.

The implementation leverages compile-time constants and template specialization to maximize GPU performance while maintaining flexibility through configurable parameters.

## Features

- **CUDA Kernel Implementation**: Full Poseidon permutation implementation including all rounds (partial and full)
- **Rust Bindings**: Safe Rust interface (using FFI) with proper error handling
- **Multi-Curve Support**: Compatible with BLS12-377 and BLS12-381 scalar fields
- **Configurable Parameters**: Extensible configuration system for different Poseidon variants
- **Compile-Time Optimization**: Uses compile-time constants for maximum performance
- **Integration with Sppark**: Built on the Sppark framework for GPU acceleration
- **Testing**: Unit tests with ground-truth implementaion (intiial correctness verification)

## Directory Structure

```
poc/poseidon-cuda/ 
 ├── src/ 
 │ ├── lib.rs # Main Rust library with public APIs 
 │ └── ... # Additional Rust source files 
 ├── cuda/ 
 │ ├── poseidon_inf.cu # Main CUDA implementation 
 │ └── poseidon_fr_r2_c1_t8_p31_a17.cuh # Configuration header 
 ├── tests/ 
 │ └── poseidon.rs # Test suite 
 ├── Cargo.toml # Rust package configuration 
 ├── build.rs # Build script with CUDA compilation 
 └── README.md # This file
 ```

### Directory Details

- **`src/`**: Contains the Rust library implementation with safe bindings to the CUDA kernels
- **`cuda/`**: Houses the CUDA kernel implementations and configuration headers
- **`tests/`**: Comprehensive test suite ensuring correctness across different configurations
- **`Cargo.toml`**: Package configuration with feature flags for different curves and Poseidon variants
- **`build.rs`**: Build script that handles CUDA compilation and feature detection


## Technical Requirements

### Hardware Requirements

- **NVIDIA GPU**: GPU with CUDA support and Compute Capability 7.0 or higher

### Software Requirements

#### CUDA Toolkit
- **Version**: 11.4 or higher
- **Installation**: Download from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
- **Components Required**:
  - CUDA Compiler (nvcc)
  - CUDA Runtime Library

#### Rust Toolchain
- **Version**: Latest stable release
- **Installation**: Install via [rustup](https://rustup.rs/)
- **Required Components**:
  - `rustc` (Rust compiler)
  - `cargo` (Rust package manager)

#### System Dependencies
- **Operating System**: 
  - Linux: Ubuntu 18.04+
  - Windows: Not tested
  - macOS: Not supported (CUDA limitation)

- **C++ Compiler**: 
  - Linux: GCC 7+ or Clang 8+
  - Windows: MSVC 2019+ or compatible

### Verification Commands

```bash
# Check CUDA installation
nvcc --version

# Check GPU compute capability
nvidia-smi

# Check Rust installation
rustc --version
cargo --version

# Verify GPU is accessible from CUDA
deviceQuery  # (from CUDA samples)
```


## Build/Test Instructions

### Building

To build the package you need to specify two features, one for the curve used and another one for the Poseidon configuration.
Currently this package supports the following possible curve features:
- bls12_381
- bls12_381

and the following Posidon configurations:
- poseidon_fr_r2_c1_t8_p31_a17

The followings are examples how to build the package. From the `poc/poseidon-cuda/` directory, run:

```bash
# Build with BLS12-381 support
cargo build --release --features=bls12_381,poseidon_fr_r2_c1_t8_p31_a17

# Build with BLS12-377 support  
cargo build --release --features=bls12_377,poseidon_fr_r2_c1_t8_p31_a17
```

### Testing

```bash
# Test with BLS12-381
cargo test --release --features=bls12_381,poseidon_fr_r2_c1_t8_p31_a17

# Test with BLS12-377
cargo test --release --features=bls12_377,poseidon_fr_r2_c1_t8_p31_a17
```



## Poseidon Configuration System:

The implementation uses a configuration system that leverages Rust's feature flags and CUDA's compile-time constants for optimal performance.

### Current Configuration
The feature poseidon_fr_r2_c1_t8_p31_a17 represents:

- fr: Field (scalar field)
- r2: Rate elements of the state: 2
- c1: Capacity elements of the state: 1
- t8: Number of full rounds: 8
- p31: Partial rounds: 31 rounds
- a17: Alpha parameter (17)

### How It Works
1. Cargo Feature Flag: The feature is defined in Cargo.toml and enables specific code paths
2. Build Script Detection: build.rs detects the feature and passes it as a compiler define:

```
if cfg!(feature = "poseidon_fr_r2_c1_t8_p31_a17") {
    posidon_feature = "FEATURE_POSEIDON_FR_R2_C1_T8_P31_A17";
}

...

nvcc.define(posidon_feature, None);
```

3. CUDA Configuration: The header file `poc/poseidon-cuda/cuda/poseidon_fr_r2_c1_t8_p31_a17.cuh` defines the needed parameters:

```
// poc/poseidon-cuda/cuda/poseidon_fr_r2_c1_t8_p31_a17.cuh
constexpr uint32_t POSEIDON_RATE = 2;
constexpr uint32_t POSEIDON_CAPACITY = 1;
constexpr uint32_t POSEIDON_FULL_ROUNDS = 8;
constexpr uint32_t POSEIDON_PARTIAL_ROUNDS = 31;
constexpr uint32_t POSEIDON_ALPHA = 17;
```


### Benefits

- Performance: Compile-time constants enable aggressive compiler optimizations
- Memory Efficiency: No runtime parameter storage needed
- Kernel Specialization: Each configuration gets its own optimized kernel
- Type Safety: Rust's type system prevents configuration mismatches

### Extending the System

To add a new configuration (e.g., poseidon_fr_r4_c2_t12_p56_a5):

1. Add to `Cargo.toml`:

```
[features]
poseidon_fr_r4_c2_t12_p56_a5 = []
```

2. Update `build.rs`:

```
let mut posidon_feature = "";
if cfg!(feature = "poseidon_fr_r4_c2_t12_p56_a5") {
    posidon_feature = "FEATURE_POSEIDON_FR_R4_C2_T12_P56_A5";
}
```

3. Create a configuration header in `poc/poseidon-cuda/cuda/` with the new config.

4. Update `poc/poseidon-cuda/cuda/poseidon_config.cuh` to include the new config.


## Limitations

1. Single Configuration Build: Only one Poseidon configuration can be active at build time
2. Static Parameters: Parameters are fixed at compile time, requiring rebuilds for different configurations
3. Memory Layout: Current implementation assumes specific memory alignment requirements
4. Curve Dependency: Tightly coupled to specific elliptic curve implementations
5. GPU Memory: Large state sizes may be limited by GPU memory constraints
6. Constant GPU Memory: ARK and MDS paramemetrs are stored in CUDA `__constant__` memory that is limited in size (can limit state size)
7. CUDA Kernel assumes state resides within a block limiting the state width to the maximum of the block size (1024 elements).

## Performance Notes

> **_NOTE:_** Important: The current implementation is not ready for comprehensive performance evaluation due to the small state size in the default configuration.

The default configuration `poseidon_fr_r2_c1_t8_p31_a17` uses only 3 elements in the state, which presents several challenges:

- Limited Parallelism: Small state size doesn't fully utilize GPU's parallel processing capabilities
- Memory Bandwidth: Insufficient work per thread to hide memory latency
- Occupancy: Low computational intensity per thread limits GPU occupancy

For meaningful performance evaluation, future configurations should include:

- Larger State Sizes: Configurations with 8, 12, or 16 elements would better utilize GPU resources
- Batch Processing: Multiple independent hash computations processed simultaneously
- Memory Coalescing: Optimized memory access patterns for larger states (or batched execution)

### Recommended Next Steps

- Implement poseidon_fr_r4_c2_t12_p56_a5 configuration for better parallelism
- Add batch processing capabilities for multiple hash computations
- Optimize memory layout for larger state sizes
- Implement performance benchmarking suite


## Future updates and Extensions

### Near-term Enhancements
- Batch Processing: Process multiple independent states simultaneously
- Pre-allocated Memory Pool: Efficient GPU memory management for repeated operations
- Streaming Interface: Asynchronous processing with CUDA streams
- Additional Curves/Posidon-configs
- Integrate into a Sponge scheme to compolete the full hashing structure

### Advanced Features

- Multi-GPU Support: Distribute work across multiple GPUs
- Merkle Tree Integration: Optimized tree hashing with Poseidon
- Zero-Knowledge Integration: Direct integration with proof systems

### Performance Optimizations

- Kernel Fusion: If next updates includes mutiiple kernels, combine multiple operations into single kernels
- Memory Hierarchy: Optimize shared memory and cache usage
- Instruction Tuning: Target specific GPU architectures
- Occupancy Optimization: Maximize GPU utilization


## Acknowledgments

This implementation builds upon the Sppark framework developed by Supranational. 
The Poseidon hash function was originally described in the paper `"Poseidon: A New Hash Function for Zero-Knowledge Proof Systems" by Grassi et al`.

Special thanks to the Sppark team for providing the foundational GPU acceleration primitives that made this implementation possible.

## License

This implementation is licensed under the Apache License Version 2.0, consistent with the Sppark framework. See the LICENSE file for details.