#pragma once

#include <cuda.h>

/// @struct PoseidonState - Poseidon hash state structure
///
/// @brief This structure holds the parameters for the Poseidon hash, including the capacity and rate states,
/// as well as the current round index. It is used to manage the internal state during the
/// calculation of the Poseidon permutation.
///
/// @tparam FieldT The field type used in the Poseidon hash, typically a finite field.
/// @tparam RateV The number of elements in the rate state.
/// @tparam CapacityV The number of elements in the capacity state.
/// @tparam FullRoundsV The number of full rounds in the Poseidon permutation.
/// @tparam PartialRoundsV The number of partial rounds in the Poseidon permutation.
/// @tparam AlphaV The exponent used in the S-Box function.
///
template<typename FieldT, uint32_t RateV, uint32_t CapacityV, uint32_t FullRoundsV, uint32_t PartialRoundsV, uint32_t AlphaV>
struct PoseidonParameters {

    /// @brief An alias for the field type used in the Poseidon hash
    using FieldType = FieldT;

    /// @brief The number of elements in the rate state
    static constexpr uint32_t RateSize = RateV;

    /// @brief The number of elements in the capacity state
    static constexpr uint32_t CapacitySize = CapacityV;

    /// @brief The total number of elements in the state (rate + capacity)
    static constexpr uint32_t StateSize = RateSize + CapacitySize;

    /// @brief The number of full rounds in the Poseidon permutation
    static constexpr uint32_t FullRounds = FullRoundsV;

    /// @brief The number of full rounds in the first (or last) part of the Poseidon permutation 
    static constexpr uint32_t FullRoundsOver2 = FullRounds / 2;

    /// @brief The number of partial rounds in the Poseidon permutation
    static constexpr uint32_t PartialRounds = PartialRoundsV;

    /// @brief The total number of rounds in the Poseidon permutation (full + partial)
    static constexpr uint32_t TotalRounds = FullRounds + PartialRounds;

    /// @brief The exponent used in the S-Box function
    static constexpr uint32_t Alpha = AlphaV;

    // Additive round keys
    FieldType ark[TotalRounds][StateSize];

    // Maximally Distance Separating (MDS) matrix
    FieldType mds[StateSize][StateSize];
};