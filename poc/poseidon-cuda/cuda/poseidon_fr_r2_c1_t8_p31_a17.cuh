#pragma once

// The compile-time config values of Poseidon hash kernels. To provide new configuration,
// create a new file in the cuda directory using the same name convention and then modify
// the Cargo.toml/build.rs files to include the new feature. You also need to include the
// new feature in the `poseidon_config.cuh` file.

// The rate value is the number of elements in the rate state, which is the input to the Poseidon hash.
constexpr uint32_t POSEIDON_RATE = 2;

// The capacity value is the number of elements in the capacity state, which is used to store intermediate values.
constexpr uint32_t POSEIDON_CAPACITY = 1;

// The number of full rounds in the Poseidon permutation, which is the main part of the hash function.
constexpr uint32_t POSEIDON_FULL_ROUNDS = 8;

// The number of partial rounds in the Poseidon permutation, which is used to add additional security.
constexpr uint32_t POSEIDON_PARTIAL_ROUNDS = 31;

// The exponent used in the S-Box function, which is a non-linear transformation applied during the permutation.
constexpr uint32_t POSEIDON_ALPHA = 17;
