#ifndef GEMM_H
#define GEMM_H
#include <cstdio>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

void gemm(const int A_height, const int A_width, const int B_height, const int B_width,
			 double *MatrixA, double *MatrixB, double *bias, double *out);

#endif