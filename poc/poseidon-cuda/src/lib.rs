use ark_ff::PrimeField;

pub mod util;

// Poseidon parameters type
#[repr(C)]
pub struct PoseidonParameters<Field: PrimeField, const STATE_SIZE: usize, const TOTAL_ROUNDS: usize, const ALPHA: usize> {
    /// Additive Round keys. These are added before each MDS matrix application to make it an affine shift.
    /// They are indexed by `ark[round_num][state_element_index]`
    pub ark: [[Field; STATE_SIZE]; TOTAL_ROUNDS],

    /// Maximally Distance Separating Matrix.
    pub mds: [[Field; STATE_SIZE]; STATE_SIZE],
}

impl<Field: PrimeField, const STATE_SIZE: usize, const TOTAL_ROUNDS: usize, const ALPHA: usize> Default
for PoseidonParameters<Field, STATE_SIZE, TOTAL_ROUNDS, ALPHA> {
    fn default() -> Self {
        Self { 
            ark: [[Field::zero(); STATE_SIZE]; TOTAL_ROUNDS], 
            mds: [[Field::zero(); STATE_SIZE]; STATE_SIZE] 
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PoseidonState<Field: PrimeField, const STATE_SIZE: usize> {
    pub state: [Field; STATE_SIZE],
}

impl<Field: PrimeField, const STATE_SIZE: usize> Default for PoseidonState<Field, STATE_SIZE> {
    fn default() -> Self {
        Self {
            state: [Field::zero(); STATE_SIZE]
        }
    }
}

// External symbol for the cuda_poseidon_permuration
extern "C" {
    fn cuda_poseidon_permuration(
        device_id: usize,
        inout_state: *mut core::ffi::c_void,
        params: *const core::ffi::c_void
    ) -> sppark::Error;
}

// Wrappper to call the ubnsafe function and do the checking of the error
pub fn poseidon_permuration<Field: PrimeField, const STATE_SIZE: usize, const TOTAL_ROUNDS: usize, const ALPHA: usize>(
    device_id: usize,
    inout_state: &PoseidonState<Field, STATE_SIZE>,
    params: &PoseidonParameters<Field, STATE_SIZE, TOTAL_ROUNDS, ALPHA>
) {

    let err = unsafe {
        cuda_poseidon_permuration(
            device_id,
            inout_state as *const _ as *mut core::ffi::c_void,
            params as *const _ as *const core::ffi::c_void
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

}
