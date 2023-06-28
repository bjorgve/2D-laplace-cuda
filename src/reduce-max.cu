#include <iostream>

#include "reduce-max.h"


#define checkCudaErrors(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA error at: " << __FILE__ << "(" << __LINE__ << "): " << cudaGetErrorString(err) << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}

// Helper function to perform a warp-level max reduction
__device__ void warp_reduce_max(volatile float* sdata, int tid) {
  // Compare each element in the warp to its neighbor and keep the maximum
  sdata[tid] = max(sdata[tid + 32], sdata[tid]);
  sdata[tid] = max(sdata[tid + 16], sdata[tid]);
  sdata[tid] = max(sdata[tid + 8], sdata[tid]);
  sdata[tid] = max(sdata[tid + 4], sdata[tid]);
  sdata[tid] = max(sdata[tid + 2], sdata[tid]);
  sdata[tid] = max(sdata[tid + 1], sdata[tid]);
}

// Kernel function to reduce the maximum value in an array
__global__ void reduce_max(float *g_idata, float *g_odata) {
  // Declare shared memory array to hold values for this block
  extern __shared__ float sdata[];

  // Each thread loads two elements from global to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  sdata[tid] = max(g_idata[i], g_idata[i + blockDim.x]);
  __syncthreads();

  // Do reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] = max(sdata[tid + s], sdata[tid]);
    }
    __syncthreads();
  }

  // Perform warp-level reduction for the remaining threads
  if (tid < 32) {
    warp_reduce_max(sdata, tid);
  }

  // Write the result for this block to global memory
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

// Function to find the maximum value in an array using CUDA
void find_max(float *data, float *max, int n) {
  int threads_per_block = 128;
  int blocks_per_grid = (n + threads_per_block * 2 - 1) / (threads_per_block * 2);

  // Launch the kernel to perform the initial reduction
  reduce_max<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(data, max);
  checkCudaErrors(cudaGetLastError());
  // Keep reducing until there is only one block left
  while (blocks_per_grid > 1) {
    blocks_per_grid = (blocks_per_grid + threads_per_block * 2 - 1) / (threads_per_block * 2);
    reduce_max<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(max, max);
    checkCudaErrors(cudaGetLastError());
  }

  // Synchronize to ensure all kernels have finished
  checkCudaErrors(cudaDeviceSynchronize());
}
