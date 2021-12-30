#ifndef CONV_H
#define CONV_H
#include <cstdio>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
double* conv(const int input_depth, const int input_width, 
    const int filter_num, const int out_width_,
    const int filter_width,
    const int padding, const int stride, const int dilation,
    double* filter, double* bias,
    double* input, int next_padding, const bool clip = true);
    
double* conv_group(const int input_depth, const int input_width, 
    const int filter_num, const int out_width_,
    const int filter_width,
    const int padding, const int stride, const int dilation,
    double* filter, double* bias,
    double* input, int next_padding, const bool clip = true);

double* pad(const int input_depth, const int input_width, const int padding, double* input);

#endif