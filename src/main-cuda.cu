#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <omp.h>

#include "laplace-cuda.h"
#include "reduce-max.h"


int main(int argc, char** argv) {
  auto start = std::chrono::high_resolution_clock::now();

  // Check if the user provided a command line argument
  if (argc < 2) {
    std::cout << "Please provide the number of elements as a command line argument.\n";
    return 1; // Return an error code
  }

  // Constants and initializations
  const auto num_elements = std::stoi(argv[1]);
  const auto max_iter = 10000;
  const auto max_error = 0.01f;
  srand(12345);

  // Memory allocations
  float *old_solution, *new_solution;
  float *diff_array, *error_array;
  size_t array_size = num_elements * num_elements * sizeof(float);

  // Allocate memory on the host/device
  cudaMallocManaged(&old_solution, array_size);
  cudaMallocManaged(&new_solution, array_size);
  cudaMallocManaged(&diff_array, array_size);
  cudaMallocManaged(&error_array, array_size);

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

  auto error = 10.0f; // Random initial value
  auto iterations = 0;

  // Perform Jacobi iterations until we either have low enough error or too
  // many iterations
  while (error > max_error && iterations < max_iter) {
    error = 0.0f;
    gpu::jacobi<<<dim3(32, 32, 1), dim3(32, 32, 1)>>>(new_solution, old_solution, num_elements);
    gpu::compute_diff<<<dim3(32, 32, 1), dim3(32, 32, 1)>>>(new_solution, old_solution, diff_array, num_elements);
    cudaDeviceSynchronize();
    find_max(diff_array, error_array, num_elements * num_elements);
    error = error_array[0];

    // Swap the pointers between old_solution and new_solution
    std::swap(old_solution, new_solution);
    iterations += 1;
  }


  // Calculate and display the results
  auto finish = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish - start);

  std::cout << "Number of elements: " << num_elements << std::endl;
  std::cout << "Number of iterations: " << iterations << std::endl;
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  // Free memory
  cudaFree(old_solution);
  cudaFree(new_solution);
  cudaFree(diff_array);
  cudaFree(error_array);

  return 0;
}
