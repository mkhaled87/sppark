#pragma once

// Include the main Poseidon header files (poseidon_config.cuh first to handle config switiching)
#include "poseidon_config.cuh"
#include "poseidon_parameters.cuh"
#include "poseidon_kernels.cuh"

// Define the types for the values fixed at this header
using PoseidonStateType = PoseidonState<fr_t, POSEIDON_RATE, POSEIDON_CAPACITY>;
using PoseidonParametersType = PoseidonParameters<fr_t, POSEIDON_RATE, POSEIDON_CAPACITY, POSEIDON_FULL_ROUNDS, POSEIDON_PARTIAL_ROUNDS, POSEIDON_ALPHA>;

// Check PoseidonStateType size is as expected
static_assert(
  sizeof(PoseidonStateType) == (PoseidonParametersType::RateSize + PoseidonParametersType::CapacitySize) * sizeof(PoseidonParametersType::FieldType), 
  "PoseidonStateType size is not as expected");

// Check PosidonPramaetersType size is as expected
static_assert(
  sizeof(PoseidonParametersType) == sizeof(PoseidonParametersType::FieldType) *
          (PoseidonParametersType::StateSize*PoseidonParametersType::TotalRounds + 
          PoseidonParametersType::StateSize*PoseidonParametersType::StateSize), 
          "PoseidonParametersType size is not as expected");
// Define the types for the values fixed at this header
