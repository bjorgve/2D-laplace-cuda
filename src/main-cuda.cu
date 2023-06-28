#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <vector>

#include "laplace-cuda.h"
#include "reduce-max.h"

#define checkCudaErrors(call) {\
  cudaError_t err = call;\
  if (err != cudaSuccess) {\
      std::cerr << "CUDA error at: " << __FILE__ << "(" << __LINE__ << "): " << cudaGetErrorString(err) << std::endl;\
      exit(EXIT_FAILURE);\
  }\
}

int main(int argc, char** argv) {
  auto start = std::chrono::high_resolution_clock::now();

  // Constants and initializations
  const auto num_elements = 1000;
  const auto max_iter = 1000;
  const auto max_error = 1.0e-2;
  srand(12345);

  // Memory allocations
  size_t array_size = num_elements * num_elements * sizeof(float);

  std::vector<float> old_solution(num_elements * num_elements);
  std::vector<float> new_solution(num_elements * num_elements);

  float* d_old_solution;
  float* d_new_solution;
  float* d_diff_array;
  float* d_error_array;

  checkCudaErrors(cudaMalloc(&d_old_solution, array_size));
  checkCudaErrors(cudaMalloc(&d_new_solution, array_size));
  checkCudaErrors(cudaMalloc(&d_diff_array, array_size));
  checkCudaErrors(cudaMalloc(&d_error_array, array_size));

  // Fill old_solution with random values between [0, 1]
  for (auto i = 0u; i < num_elements; i++) {
    for (auto j = 0u; j < num_elements; j++) {
      old_solution[i + num_elements * j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
  }

  // Copy boundary conditions from old_solution to new_solution
  for (auto i = 0u; i < num_elements; i++) {
    new_solution[i] = old_solution[i];
    new_solution[i + num_elements * (num_elements - 1)] = old_solution[i + num_elements * (num_elements - 1)];
    new_solution[i * num_elements] = old_solution[i * num_elements];
    new_solution[(i + 1) * num_elements - 1] = old_solution[(i + 1) * num_elements - 1];
  }

  // Copy old_solution and new_solution to the device
  checkCudaErrors(cudaMemcpy(d_old_solution, old_solution.data(), array_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_new_solution, new_solution.data(), array_size, cudaMemcpyHostToDevice));


  // Jacobi iterations
  float error = 10.0f; // Random initial value
  int iterations = 0;
  int sync_interval = 100; // Sync to the host every 10 iterations

  while (error > max_error && iterations < max_iter) {
    for (int iter = 0; iter < sync_interval && iterations < max_iter; ++iter) {
      // error = 0.0f;
      gpu::jacobi<<<dim3(32, 32, 1), dim3(32, 32, 1), 0>>>(d_new_solution, d_old_solution, num_elements);
      checkCudaErrors(cudaGetLastError());
      gpu::compute_diff<<<dim3(32, 32, 1), dim3(32, 32, 1), 0>>>(d_new_solution, d_old_solution, d_diff_array, num_elements);
      checkCudaErrors(cudaGetLastError());

      int threads_per_block = 128;
      int blocks_per_grid = (num_elements * num_elements + threads_per_block * 2 - 1) / (threads_per_block * 2);
      reduce_max<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(d_diff_array, d_error_array);
      checkCudaErrors(cudaGetLastError());

      while (blocks_per_grid > 1) {
        blocks_per_grid = (blocks_per_grid + threads_per_block * 2 - 1) / (threads_per_block * 2);
        reduce_max<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(d_error_array, d_error_array);
        checkCudaErrors(cudaGetLastError());
      }

      std::swap(d_old_solution, d_new_solution);
      iterations += 1;
    }

    checkCudaErrors(cudaMemcpy(&error, d_error_array, sizeof(float), cudaMemcpyDeviceToHost));
}
cudaDeviceSynchronize();

  // Calculate and display the results
  auto finish = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);

  // Copy new and old solution back to the host
  checkCudaErrors(cudaMemcpy(old_solution.data(), d_old_solution, array_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(new_solution.data(), d_new_solution, array_size, cudaMemcpyDeviceToHost));

  std::cout << "Solution value at [20][20]: " << old_solution[20 + num_elements * 20] << std::endl;
  std::cout << "New solution value at [20][20]: " << new_solution[20 + num_elements * 20] << std::endl;
  std::cout << "Number of iterations: " << iterations << std::endl;
  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

  // Free memory
  checkCudaErrors(cudaFree(d_old_solution));
  checkCudaErrors(cudaFree(d_new_solution));
  checkCudaErrors(cudaFree(d_diff_array));
  checkCudaErrors(cudaFree(d_error_array));



  return 0;
}
