
#define BLOCKSIZE 8
using namespace std;

/**
 * conv1Kernel只处理kernel size 1*1的卷积，省去了许多不必要的循环、乘法等操作。
 * 原始版本（及没有去掉循环的版本）可以参照convKernel
*/
__global__ void conv1Kernel(double *input,double *filter,double *output, double* bias, int filter_width,int filter_num,int input_width,int input_depth,int stride,int out_width, int next_padding, bool clip){
    /* params:
     * input_width: 已经考虑padding在内的input宽度
     * out_width: 未考虑next padding的情况下的输出整体图像的宽度
     * next_padding: 需要输入下一层的padding参数，在本层产生运算结果的同时进行pad
     */
    int output_row = blockIdx.x*blockDim.x+threadIdx.x;
    int output_col = blockIdx.y*blockDim.y+threadIdx.y;
    if(output_row >= out_width || output_col >= out_width) return;
    out_width += next_padding * 2; //此后out_width表示加pad之后的宽度

    int fnum = blockIdx.z;
    double tmp = 0.0;
    for(int d=0;d<input_depth;d++)
        tmp = tmp + input[d * input_width*input_width + (output_row)*input_width + output_col]*filter[fnum*input_depth + d];
    tmp += bias[fnum];
    if(clip){
        if(tmp < 0) tmp = 0.0;
        else if(tmp > 6) tmp = 6.0;
    }
    output[fnum*out_width*out_width + (output_row+next_padding)*out_width+output_col+next_padding] = tmp;
}


__global__ void convKernel(double *input,double *filter,double *output, double* bias, int filter_width,int filter_num,int input_width,int input_depth,int stride,int out_width, int next_padding, bool clip){
    /* 
     * 每个线程负责计算output中的一个double值
     * params:
     * input_width: 已经考虑padding在内的input宽度
     * out_width: 未考虑next padding的情况下的输出整体图像的宽度
     * next_padding: 需要输入下一层的padding参数，在本层产生运算结果的同时进行pad
     */
    int output_row = blockIdx.x*blockDim.x+threadIdx.x;
    int output_col = blockIdx.y*blockDim.y+threadIdx.y;
    int input_row = output_row*stride;
    int input_col = output_col*stride;
    if(output_row >= out_width || output_col >= out_width) return;
    out_width += next_padding * 2;//此后out_width表示加pad之后的宽度

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
    output[fnum*out_width*out_width + (output_row+next_padding)*out_width+output_col+next_padding] = tmp;
}

__global__ void convGroupKernel(double *input,double *filter,double *output, double* bias, int filter_width,int filter_num,int input_width,int input_depth,int stride,int out_width, int next_padding, bool clip){
    int layer_idx = blockIdx.z;
    int output_row = blockIdx.x*blockDim.x+threadIdx.x;
    int output_col = blockIdx.y*blockDim.y+threadIdx.y;
    int input_row = output_row*stride;
    int input_col = output_col*stride;
    if(output_row >= out_width || output_col >= out_width) return;
    out_width += next_padding * 2;

    
    double tmp = 0.0; //本地存储，减少global mem的访问次数
    //二重循环计算9个元素与9个元素对应相乘再相加
    for(int r=0;r<filter_width;r++)
    for(int c=0;c<filter_width;c++){
        tmp += input[layer_idx*input_width*input_width + (input_row+r)*input_width + input_col+c]*filter[layer_idx*filter_width*filter_width + r*filter_width + c];
    }
    
    tmp += bias[layer_idx];
    if(clip){
        if(tmp < 0) tmp = 0.0;
        else if(tmp > 6) tmp = 6.0;
    }
    output[layer_idx*out_width*out_width + (output_row+next_padding)*out_width+output_col+next_padding] = tmp;
    
}



void conv(const int input_depth, const int input_width, 
    const int filter_num, const int out_width_,
    const int filter_width,
    const int padding, const int stride, const int dilation,
    double* filter, double* bias,
    double* &input, double* &output, int next_padding, const bool clip = true){
    /* 
     * 此处假设filter, bias, input均指向global mem，而非host mem。
     * 整个网络的计算、中间结果的存储，都在gpu上进行。
     */
    double *img_cuda,*filter_cuda,*output_cuda,*bias_cuda; //img_cuda for padded tensor.
    int in_width=input_width+padding*2;
    int out_width=out_width_+next_padding*2;
    size_t outsize = sizeof(double)*filter_num*out_width*out_width;
    
    output_cuda = output;
    cudaMemset(output_cuda, 0, outsize);
    filter_cuda = filter;
    bias_cuda=bias;
    img_cuda = input;

    int g=(out_width_+BLOCKSIZE-1)/BLOCKSIZE; //向上取整
    dim3 grid(g, g, filter_num);
    dim3 threads(BLOCKSIZE, BLOCKSIZE);  
    if(filter_width == 1){  
        conv1Kernel<<<grid,threads>>>(img_cuda,filter_cuda,output_cuda,bias_cuda,filter_width,filter_num,in_width,input_depth,stride,out_width_, next_padding, clip);
        }
    else {
        dim3 threads(BLOCKSIZE, BLOCKSIZE);
        convKernel<<<grid,threads>>>(img_cuda,filter_cuda,output_cuda,bias_cuda,filter_width,filter_num,in_width,input_depth,stride,out_width_, next_padding, clip);
    }
    
    input = output_cuda;    
}

void conv_group(const int input_depth, const int input_width, 
    const int filter_num, const int out_width_,
    const int filter_width,
    const int padding, const int stride, const int dilation,
    double* filter, double* bias,
    double* &input, double* &output,int next_padding, const bool clip = true){
    // 本函数处理group参数等于input_depth的卷积计算，结构与conv几乎完全相同
    double *img_cuda,*filter_cuda,*output_cuda,*bias_cuda; //img_cuda for padded tensor.
    int in_width=input_width+padding*2;
    int out_width=out_width_+next_padding*2;
    size_t outsize = sizeof(double)*filter_num*out_width*out_width;

    output_cuda = output;
    cudaMemset(output_cuda,0, outsize);
    filter_cuda = filter;
    bias_cuda=bias;
    img_cuda = input;    

    int g=(out_width_+BLOCKSIZE-1)/BLOCKSIZE;
    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 grid(g, g, filter_num);
    convGroupKernel<<<grid,threads>>>(img_cuda,filter_cuda,output_cuda,bias_cuda,filter_width,filter_num,in_width,input_depth,stride,out_width_,next_padding,clip);

    input = output;    
}

double* pad(const int input_depth, const int input_width, const int padding, double* input){
    double *img_cuda; //img_cuda for padded tensor.
    int in_width=input_width+padding*2;
    cudaMalloc(&img_cuda,sizeof(double)*input_depth*in_width*in_width);
    cudaMemset(img_cuda, 0, sizeof(double)*input_depth*in_width*in_width);
    for(int mapid=0;mapid<input_depth;mapid++)
    for(int line=0;line<input_width;line++){
        cudaMemcpy(&img_cuda[mapid*in_width*in_width+(padding+line)*in_width+padding], &input[mapid*input_width*input_width+line*input_width], sizeof(double)*input_width, cudaMemcpyHostToDevice);
    }
    return img_cuda;
}


// 以下是开发过程中为了检验conv函数正确性而设计的测试函数
// int test_conv1_main()
// {
//     double* filter=new double[2*3*3*3];
//     double* image=new double[3*5*5];
//     double* out= new double[2*25];
//     double* bias = new double[2];
//     for(int i=0;i<2*3*3*3;i++)filter[i]=1;
//     for(int i=0;i<3*5*5;i++)image[i]=1;
//     bias[0] = 100; bias[1] = 10000;
//     double* filter_cu, *image_cu, *out_cu, *bias_cu;
    
//     cudaMalloc(&filter_cu, 2*3*3*3*sizeof(double));
//     cudaMemcpy(filter_cu, filter,2*3*3*3*sizeof(double), cudaMemcpyHostToDevice);
//     cudaMalloc(&image_cu, 3*5*5*sizeof(double));
//     cudaMemcpy(image_cu, image, 3*5*5*sizeof(double), cudaMemcpyHostToDevice);
//     cudaMalloc(&bias_cu, 2*sizeof(double));
//     cudaMemcpy(bias_cu, bias,2*sizeof(double),cudaMemcpyHostToDevice);
    
	
//     conv(3,5,2,5,3,1,1,0,filter_cu,bias_cu,image_cu,out_cu,0,false);

//     cudaMemcpy(out, out_cu, 50*sizeof(double), cudaMemcpyDeviceToHost);
//     for(int i=0;i<50;i++)printf("%f\n",out[i]);
//     return 0;
// }
// int main(){test_conv1_main(); return 0;}