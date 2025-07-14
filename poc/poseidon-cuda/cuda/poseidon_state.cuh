#pragma once

#include <cuda.h>

/// @struct PoseidonState - Poseidon hash state structure
///
/// @brief This structure holds the state of the Poseidon hash, including the capacity and rate states,
/// as well as the current round index. It is used to manage the internal state during the
/// calculation of the Posiedon permutation.
///
/// @tparam FieldT The field type used in the Poseidon hash, typically a finite field.
/// @tparam RateV The number of elements in the rate state.
/// @tparam CapacityV The number of elements in the capacity state.
///
template<typename FieldT, uint32_t RateV, uint32_t CapacityV>
struct PoseidonState {

    /// @brief An alias for the field type used in the Poseidon hash
    using FieldType = FieldT;

    // Operator [] to access elements in the state
    __device__ __forceinline__
    FieldType& operator[](uint32_t index) {
        
        // Demultiplex the index to access either capacity or rate state
        if (index < CapacityV) {
            return capacity_state[index];
        } else {
            return rate_state[index - CapacityV];
        }
    }

    /// @brief Capacity state for the Poseidon hash
    FieldType capacity_state[CapacityV];

    /// @brief Rate state for the Poseidon hash
    FieldType rate_state[RateV];
};