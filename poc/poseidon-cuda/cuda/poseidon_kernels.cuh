#pragma once

#include <cuda.h>

#include "poseidon_state.cuh"

/// @brief CUDA kernel for the Poseidon permutation
///
/// This kernel performs the Poseidon permutation on the provided state using the specified parameters.
/// It applies the round keys, S-Box function, and MDS matrix multiplication in a parallel manner.
///
/// @tparam PoseidonStateT The type of the Poseidon state.
/// @tparam StateSizeV The size of the state (rate + capacity).
/// @tparam FullRoundsV The number of full rounds in the Poseidon permutation.
/// @tparam PartialRoundsV The number of partial rounds in the Poseidon permutation.
/// @tparam AlphaV The exponent used in the S-Box function.
///
/// @param state Pointer to the Poseidon state to be permuted.
///
template<typename PoseidonStateT, uint32_t StateSizeV, uint32_t FullRoundsV, uint32_t PartialRoundsV, uint32_t AlphaV>
__launch_bounds__(StateSizeV)
__global__ 
void CUDA_Poseidon_Permutation(PoseidonStateT* state) {
    
    // Alias for the field type used in the Poseidon hash
    using FieldType = typename PoseidonStateT::FieldType;

    // Constants for the number of rounds
    constexpr uint32_t TotalRounds = FullRoundsV + PartialRoundsV;
    constexpr uint32_t FullRoundsOver2 = FullRoundsV / 2;

    // Ensure the thread is within bounds
    if (threadIdx.x >= StateSizeV) {
        return;
    }

    // Shared memory for the state and the parameters
    __shared__ PoseidonStateT shared_state;

    // Load the state and parameters into shared memory
    shared_state[threadIdx.x] = (*state)[threadIdx.x];

    // Ensure all threads have loaded their data into shared memory
    __syncthreads();

    #pragma unroll
    for (uint32_t round = 0; round < TotalRounds; ++round) {

        // Apply the round key addition
        shared_state[threadIdx.x] += poseidon_ark[round][threadIdx.x];

        // Apply the S-Box function
        if(round < FullRoundsOver2  || round >= TotalRounds - FullRoundsOver2) {
            shared_state[threadIdx.x] ^= AlphaV;
        } else {
            if (threadIdx.x == 0) {
                shared_state[threadIdx.x] ^= AlphaV;
            }
        }

        // Sync threads to ensure all threads have completed their operations
        __syncthreads();

        // Apply the MDS matrix multiplication corrsponding to the current state element
        FieldType matrix_mult_result = FieldType{};
        matrix_mult_result.zero();
        for (uint32_t i = 0; i < StateSizeV; ++i) {
            matrix_mult_result += poseidon_mds[threadIdx.x][i] * shared_state[i];
        }

        // Sync threads to ensure all threads have completed the MDS multiplication
        __syncthreads();

        // Store the result back into the rate state
        shared_state[threadIdx.x] = matrix_mult_result;

        // Sync threads to ensure all threads have completed their operations before starting the next round
        __syncthreads();
    }

    // Write the final shared state back to global memory
    (*state)[threadIdx.x] = shared_state[threadIdx.x];
}
