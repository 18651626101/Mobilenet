#include <cstdio>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

__global__ void convKernel(float *input,float *filter,float *output, float* bias,int filter_width,int filter_num,int input_width,int input_depth,int stride,int out_width)
{
    int output_row = blockIdx.x*blockDim.x+threadIdx.x;
    int output_col = blockIdx.y*blockDim.y+threadIdx.y;
    int input_row = output_row*stride;
    int input_col = output_col*stride;

    for(int fnum=0;fnum<filter_num;fnum++){
        float tmp = 0.0;
        for(int d=0;d<input_depth;d++)
            for(int r=0;r<filter_width;r++)
                for(int c=0;c<filter_width;c++)
                    tmp += input[d*input_width*input_width + (input_row+r)*input_width + input_col+c]
                        *filter[fnum*input_depth*filter_width*filter_width + d*filter_width*filter_width + r*filter_width + c];
        output[fnum*out_width*out_width + output_row*out_width+output_col] = tmp+bias[fnum];
    }
}


void conv(float *input, float *filter, float *output, float* bias, int filter_width, int filter_num,int input_width, int input_depth, int padding, int stride){
    float *img_cuda,*filter_cuda,*output_cuda,*bias_cuda;
    int in_width=input_width+padding*2;
    int out_width=(in_width-filter_width)/stride+1;

    cudaMalloc(&img_cuda,sizeof(float)*input_depth*in_width*in_width);
    cudaMalloc(&filter_cuda,sizeof(float)*filter_num*input_depth*filter_width*filter_width);
    cudaMalloc(&output_cuda,sizeof(float)*filter_num*out_width*out_width);
    cudaMalloc(&bias_cuda,sizeof(float)*filter_num);
    cudaMemcpy(filter_cuda, filter, sizeof(float)*filter_num*input_depth*filter_width*filter_width, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_cuda, bias, sizeof(float)*filter_num, cudaMemcpyHostToDevice);

    for(int mapid=0;mapid<input_depth;mapid++)
        for(int line=0;line<input_width;line++)
            cudaMemcpy(&img_cuda[mapid*in_width*in_width+(padding+line)*in_width+padding], &input[mapid*input_width*input_width+line*input_width], sizeof(float)*input_width, cudaMemcpyHostToDevice);


    dim3 threads(1, 1);
    dim3 grid(out_width, out_width);

    convKernel<<<grid,threads>>>(img_cuda,filter_cuda,output_cuda,bias_cuda,filter_width,filter_num,in_width,input_depth,stride,out_width);
    // cudaDeviceSynchronize();

    cudaMemcpy(output, output_cuda, sizeof(float)*filter_num*out_width*out_width, cudaMemcpyDeviceToHost);

    cudaFree(img_cuda);
    cudaFree(filter_cuda);
    cudaFree(output_cuda);
    cudaFree(bias_cuda);
}



int test_conv0_main()
{
     float* filter=new float[2*3*3*3];
     float* image=new float[3*5*5];
     float* out= new float[2*25];
     float* bias = new float[2];
     for(int i=0;i<2*3*3*3;i++)filter[i]=1;
     for(int i=0;i<3*5*5;i++)image[i]=1;
     bias[0] = 100; bias[1] = 10000;
     conv(image,filter,out, bias,3,2,5,3,1,1);
     for(int i=0;i<50;i++)printf("%f\n",out[i]);
     return 0;
}