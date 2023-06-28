#ifndef REDUCE_MAX_H
#define REDUCE_MAX_H

#include <cuda_runtime.h>

__device__ void warp_reduce_max(volatile float* sdata, int tid);

__global__ void reduce_max(float *g_idata, float *g_odata);

void find_max(float *d_idata, float *d_odata, int size);

#endif
