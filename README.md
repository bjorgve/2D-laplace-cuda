# 2D-Laplace Solver with CUDA

This project provides an implementation of a 2D-Laplace solver using CUDA.
The goal is to examine the performance and scaling characteristics
of the solver.

## Running the Project on Saga

Follow these steps to run the project on the Saga supercomputer:

1. First, navigate to the jobs directory:

   - cd jobs

2. Next, run the compilation script:

   - sh compile.sh


3. Finally, submit the job to the job scheduler using your project number
   (replace `nnXXXXk` with your actual project number):

   - sbatch --account=nnXXXXk



The job should take approximately 10 minutes to run. Once the job has completed,
you can examine the output to analyze how the solver's performance scales with
the size of the matrix.
