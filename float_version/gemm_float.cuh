#ifndef GEMM_H
#define GEMM_H
#include <cstdio>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

float *gemm(const int input_width, const int input_height, const int weight_width, const int weight_height,
			float *weight, float *bias, float *input);

#endif