#ifndef CONV_H
#define CONV_H
#include <cstdio>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
float* conv(const int input_depth, const int input_width, 
    const int filter_num, const int out_width_,
    const int filter_width,
    const int padding, const int stride, const int dilation,
    float* filter, float* bias,
    float* input, const bool clip=true);

#endif