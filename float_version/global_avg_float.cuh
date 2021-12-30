#ifndef GLOBAL_AVG_H
#define GLOBAL_AVG_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

float *global_avg(float *input, int depth, int width);

#endif