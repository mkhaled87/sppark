#include <iostream>
#include <iomanip>

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

#include "poseidon.cuh"


/// @brief C++ launcher for the CUDA kernel of Poseidon permutation
///
/// @param device_id The ID of the GPU device to use.
/// @param p_inout_state Pointer to the Poseidon state to be permuted.
/// @param p_params Pointer to the Poseidon parameters.
///
/// @return RustError indicating success or failure of the operation.
SPPARK_FFI
RustError::by_value cuda_poseidon_permuration(size_t device_id, void* p_inout_state, const void* p_params) {

    // Check if the pointers are null
    if (!p_inout_state || !p_params) {

        std::cerr << "Error: Null pointer passed for state or parameters." << std::endl;
        return RustError{cudaErrorInvalidValue};
    }

    /// Cast provided objects to state/params types
    PoseidonStateType* h_state = reinterpret_cast<PoseidonStateType*>(p_inout_state);
    PoseidonParametersType* h_params = reinterpret_cast<PoseidonParametersType*>(const_cast<void*>(p_params));

    // Get targeted GPU device
    auto& gpu = select_gpu(device_id);

    try {

        // Activate current GPU device
        gpu.select();

        // Allocate device memory for state and parameters
        dev_ptr_t<PoseidonStateType> d_state{1, gpu};

        // Copy state data to device memory
        gpu.HtoD(&d_state[0], h_state, 1);

        // Copy params to constant memory
        CUDA_OK(cudaMemcpyToSymbol(poseidon_ark, h_params->ark, sizeof(PoseidonParametersType::FieldType) * PoseidonParametersType::TotalRounds * PoseidonParametersType::StateSize));
        CUDA_OK(cudaMemcpyToSymbol(poseidon_mds, h_params->mds, sizeof(PoseidonParametersType::FieldType) * PoseidonParametersType::StateSize * PoseidonParametersType::StateSize));

        // Call the CUDA Kernel CUDA_Poseidon_Permutation
        CUDA_Poseidon_Permutation
            <PoseidonStateType,
             PoseidonParametersType::StateSize,
             PoseidonParametersType::FullRounds,
             PoseidonParametersType::PartialRounds,
             PoseidonParametersType::Alpha>
            <<<1, PoseidonParametersType::StateSize, 0, gpu>>>(&d_state[0]);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA_Poseidon_Permutation CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        }

        // Copy back the result from device to host
        gpu.DtoH(h_state, &d_state[0], 1);

        // Sync the GPU to ensure all operations are complete
        gpu.sync();

    } catch (const cuda_error& e) {
        
        gpu.sync();
        return RustError{e.code(), e.what()};
    }

    // dummy return with success for now
    return RustError{cudaSuccess};
}