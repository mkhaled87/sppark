#pragma once

#if defined(FEATURE_BLS12_381)
# include <ff/bls12-381-fp2.hpp>
#elif defined(FEATURE_BLS12_377)
# include <ff/bls12-377-fp2.hpp>
#else
# error "No FEATURE specified for the field type!"
#endif

#if defined(FEATURE_POSEIDON_FR_R2_C1_T8_P31_A17)
#include "poseidon_fr_r2_c1_t8_p31_a17.cuh"
#else
#error "No FEATURE specified for Poseidon config!";
#endif

// The total size of the Poseidon state, which is the sum of the rate and capacity sizes.
constexpr uint32_t POSEIDON_STATE_SIZE = POSEIDON_RATE + POSEIDON_CAPACITY;

// The total number of rounds in the Poseidon permutation, which is the sum of full and partial rounds.
constexpr uint32_t POSEIDON_TOTAL_ROUNDS = POSEIDON_FULL_ROUNDS + POSEIDON_PARTIAL_ROUNDS;

// Define constant memory for ark and mds matrices
__constant__ fr_t poseidon_ark[POSEIDON_TOTAL_ROUNDS][POSEIDON_STATE_SIZE];
__constant__ fr_t poseidon_mds[POSEIDON_STATE_SIZE][POSEIDON_STATE_SIZE];
