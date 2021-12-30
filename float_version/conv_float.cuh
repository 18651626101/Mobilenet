#ifndef CONV_H
#define CONV_H
#include <cstdio>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
void conv(const int input_depth, const int input_width, 
    const int filter_num, const int out_width_,
    const int filter_width,
    const int padding, const int stride, const int dilation,
    float* filter, float* bias,
    float* &input, float* &output, int next_padding, const bool clip = true);
    
void conv_group(const int input_depth, const int input_width, 
    const int filter_num, const int out_width_,
    const int filter_width,
    const int padding, const int stride, const int dilation,
    float* filter, float* bias,
    float* &input, float* &output,int next_padding, const bool clip = true);

float* pad(const int input_depth, const int input_width, const int padding, float* input);

#endif