#include "laplace-cuda.h"

namespace gpu {
// Jacobi iteration kernel for GPU parallelization
// Calculates the new values for the interior elements of the matrix
__global__ void jacobi(float *out_array, float *inp_array, int num_elements) {
  // Calculate the global indices of the element to compute
  int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  int idx_y = threadIdx.y + blockIdx.y * blockDim.y;

  // Calculate the stride values to avoid redundant computation
  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;

  // Iterate over the interior elements of the matrix
  for (int i = idx_x+1; i < num_elements - 1; i += stride_x) {
    for (int j = idx_y+1; j < num_elements - 1; j += stride_y) {
      // Compute the new value of the current element using the Jacobi method
      out_array[i + num_elements * j] = 0.25 * (inp_array[i + num_elements * (j + 1)] +
                                                inp_array[i + num_elements * (j - 1)] +
                                                inp_array[(i - 1) + num_elements * j] +
                                                inp_array[(i + 1) + num_elements * j]);
    }
  }
}

__global__ void compute_diff(float* new_solution, float* old_solution, float* diff_array, int num_elements) {
  // Get the thread's x and y indices in the block and compute the global indices
  int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  int idx_y = threadIdx.y + blockIdx.y * blockDim.y;

  // Compute the stride values for each dimension
  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;

  // Iterate over the interior elements of the matrix
  for (int i = idx_x+1; i < num_elements - 1; i += stride_x) {
    for (int j = idx_y+1; j < num_elements - 1; j += stride_y) {
      // Compute the difference between the old and new solution at this element
      diff_array[i + num_elements * j] = fabsf(new_solution[i + num_elements * j] - old_solution[i + num_elements * j]);
    }
  }
}

}  // namespace gpu