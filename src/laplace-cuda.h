#ifndef LAPLACE_CUDA_H
#define LAPLACE_CUDA_H

namespace gpu {

// Jacobi iteration kernel for GPU parallelization
// Calculates the new values for the interior elements of the matrix
__global__ void jacobi(float *out_array, float *inp_array, int num_elements);
__global__ void compute_diff(float *new_solution, float *old_solution, float *diff, int num_elements);

// void check_error(float* old_solution, float* new_solution, int num_elements, float& error);
} // namespace gpu
#endif // LAPLACE_CUDA_H
