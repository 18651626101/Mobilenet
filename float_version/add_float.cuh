#ifndef ADD_H
#define ADD_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

void add(const int channel, const int shape, float *inputA, float *inputB);

#endif