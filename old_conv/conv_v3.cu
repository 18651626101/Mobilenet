#include "conv_v2.cuh"
#define BLOCKSIZE 32
using namespace std;

/**
 * the output shape: (out_channel, out_width, out_width)
 * parallel: (out_width, out_width) with maximum #thread per block
 * with fusion of conv and clip
*/

__global__ void convKernel(double *input,double *filter,double *output, double* bias, int filter_width,int filter_num,int input_width,int input_depth,int stride,int out_width, bool clip){
    int output_row = blockIdx.x*blockDim.x+threadIdx.x;
    int output_col = blockIdx.y*blockDim.y+threadIdx.y;
    int input_row = output_row*stride;
    int input_col = output_col*stride;
    if(output_row >= out_width || output_col >= out_width) return;

    int fnum = blockIdx.z;
    double tmp = 0.0;
    for(int d=0;d<input_depth;d++)
    for(int r=0;r<filter_width;r++)
    for(int c=0;c<filter_width;c++)
        tmp += input[d*input_width*input_width + (input_row+r)*input_width + input_col+c]*filter[fnum*input_depth*filter_width*filter_width + d*filter_width*filter_width + r*filter_width + c];
    
    tmp += bias[fnum];
    if(clip){
        if(tmp < 0) tmp = 0.0;
        else if(tmp > 6) tmp = 6.0;
    }
    output[fnum*out_width*out_width + output_row*out_width+output_col] = tmp;
}

__global__ void convGroupKernel(double *input,double *filter,double *output, double* bias, int filter_width,int filter_num,int input_width,int input_depth,int stride,int out_width, bool clip){
    int layer_idx = blockIdx.z;
    int output_row = blockIdx.x*blockDim.x+threadIdx.x;
    int output_col = blockIdx.y*blockDim.y+threadIdx.y;
    int input_row = output_row*stride;
    int input_col = output_col*stride;
    if(output_row >= out_width || output_col >= out_width) return;

    
        double tmp = 0.0;
        for(int r=0;r<filter_width;r++)
        for(int c=0;c<filter_width;c++){
            tmp += input[layer_idx*input_width*input_width + (input_row+r)*input_width + input_col+c]*filter[layer_idx*filter_width*filter_width + r*filter_width + c];
        }
        
        tmp += bias[layer_idx];
        if(clip){
            if(tmp < 0) tmp = 0.0;
            else if(tmp > 6) tmp = 6.0;
        }
        output[layer_idx*out_width*out_width + output_row*out_width+output_col] = tmp;
    
}


/* the double*  are assumed to point to cuda mem. */
double* conv(const int input_depth, const int input_width, 
    const int filter_num, const int out_width_,
    const int filter_width,
    const int padding, const int stride, const int dilation,
    double* filter, double* bias,
    double* input, const bool clip){
    // printf("========== conv_v1::begin conv ==========\n");
    double *img_cuda,*filter_cuda,*output_cuda,*bias_cuda; //img_cuda for padded tensor.
    int in_width=input_width+padding*2;
    int out_width=out_width_;
    

    cudaMalloc(&output_cuda,sizeof(double)*filter_num*out_width*out_width);
    filter_cuda = filter;
    bias_cuda=bias;
    if(padding){
        cudaMalloc(&img_cuda,sizeof(double)*input_depth*in_width*in_width);
        cudaMemset(img_cuda, 0, sizeof(double)*input_depth*in_width*in_width);
        for(int mapid=0;mapid<input_depth;mapid++)
        for(int line=0;line<input_width;line++){
            cudaMemcpy(&img_cuda[mapid*in_width*in_width+(padding+line)*in_width+padding], &input[mapid*input_width*input_width+line*input_width], sizeof(double)*input_width, cudaMemcpyDeviceToDevice);
        }
    }
    else img_cuda = input;

    int g=(out_width+BLOCKSIZE-1)/BLOCKSIZE;
    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 grid(g, g, filter_num);

    convKernel<<<grid,threads>>>(img_cuda,filter_cuda,output_cuda,bias_cuda,filter_width,filter_num,in_width,input_depth,stride,out_width, clip);
    if(img_cuda != input)
        cudaFree(img_cuda);
    cudaFree(input);

    // printf("========== conv_v1::end conv ==========\n");
    return output_cuda;
}

double* conv_group(const int input_depth, const int input_width, 
    const int filter_num, const int out_width_,
    const int filter_width,
    const int padding, const int stride, const int dilation,
    double* filter, double* bias,
    double* input, const bool clip){
    // printf("========== conv_v1::begin conv ==========\n");
    double *img_cuda,*filter_cuda,*output_cuda,*bias_cuda; //img_cuda for padded tensor.
    int in_width=input_width+padding*2;
    int out_width=out_width_;
    

    cudaMalloc(&output_cuda,sizeof(double)*filter_num*out_width*out_width);
    filter_cuda = filter;
    bias_cuda=bias;

    if(padding){
        cudaMalloc(&img_cuda,sizeof(double)*input_depth*in_width*in_width);
        cudaMemset(img_cuda, 0, sizeof(double)*input_depth*in_width*in_width);
        for(int mapid=0;mapid<input_depth;mapid++)
        for(int line=0;line<input_width;line++){
            cudaMemcpy(&img_cuda[mapid*in_width*in_width+(padding+line)*in_width+padding], &input[mapid*input_width*input_width+line*input_width], sizeof(double)*input_width, cudaMemcpyDeviceToDevice);
        }
    }else img_cuda = input;    

    dim3 threads(1, 1);
    dim3 grid(out_width, out_width, input_depth);

    convGroupKernel<<<grid,threads>>>(img_cuda,filter_cuda,output_cuda,bias_cuda,filter_width,filter_num,in_width,input_depth,stride,out_width, clip);

    if(img_cuda != input)
        cudaFree(img_cuda);
    cudaFree(input);

    // printf("========== conv_v1::end conv ==========\n");
    return output_cuda;
}



int test_conv1_main()
{
    double* filter=new double[2*3*3*3];
    double* image=new double[3*5*5];
    double* out= new double[2*25];
    double* bias = new double[2];
    for(int i=0;i<2*3*3*3;i++)filter[i]=1;
    for(int i=0;i<3*5*5;i++)image[i]=1;
    bias[0] = 100; bias[1] = 10000;
    double* filter_cu, *image_cu, *out_cu, *bias_cu;
    struct timeval t0;
    gettimeofday(&t0, NULL);
    cudaMalloc(&filter_cu, 2*3*3*3*sizeof(double));
    cudaMemcpy(filter_cu, filter,2*3*3*3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&image_cu, 3*5*5*sizeof(double));
    cudaMemcpy(image_cu, image, 3*5*5*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&bias_cu, 2*sizeof(double));
    cudaMemcpy(bias_cu, bias,2*sizeof(double),cudaMemcpyHostToDevice);
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
	double timeuse;
    out_cu = conv(3,5,2,5,3,1,1,0,filter_cu,bias_cu,image_cu, false);
    gettimeofday(&t2, NULL);
	timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
	printf("Conv Use Time:%fs\n", timeuse);
    timeuse = t1.tv_sec - t0.tv_sec + (t1.tv_usec - t0.tv_usec) / 1000000.0;
	printf("Malloc Use Time:%fs\n", timeuse);
    cudaMemcpy(out, out_cu, 50*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i=0;i<50;i++)printf("%f\n",out[i]);
    return 0;
}
// int main(){test_conv1_main(); return 0;}