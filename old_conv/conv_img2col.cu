__global__ void img2col(const int input_depth, const int input_width, const int out_width, const int padding, const int stride, double* input, double* dest){ 
    // 只考虑 kernelsize =  (3,3), stride = 1
    int input_row = blockIdx.x * blockDim.x + threadIdx.x;
    int input_col = blockIdx.y * blockDim.y + threadIdx.y;
    int input_channel = blockIdx.z;
    double tmp = input[((input_channel*input_width)+input_row)*input_width + input_col];
    //每个进程取input中的一个element，把它放到矩阵中该出现的（可能不止一个）的位置上
    // 第一组位置，input的这个element对应卷积核的第三行
    // col: out_width * (input_row-2) + input_col-2
    // row: 9*input_channel +8
    if(input_row >= 2){
        dest[(out_width*(input_row-2)+input_col )+ (9*input_channel+6)*out_width*out_width ] = tmp;
        if(input_col >=1){
            dest[(out_width*(input_row-2)+input_col-1)+ (9*input_channel+7)*out_width*out_width ] = tmp;
            if(input_col>=2)
                dest[(out_width*(input_row-2)+input_col-2)+ (9*input_channel+8)*out_width*out_width ] = tmp;
        }
    }
    // col: out_width * (input_row - 1) + input_col-2
    // row: 9*input_channel + 5
    if(input_row>=1){
        dest[(out_width*(input_row-1)+input_col )+ (9*input_channel+3)*out_width*out_width ] = tmp;
        if(input_col>=1){
            dest[(out_width*(input_row-1)+input_col-1)+ (9*input_channel+4)*out_width*out_width ] = tmp;
            if(input_col>=2)
                dest[(out_width*(input_row-1)+input_col-2)+ (9*input_channel+5)*out_width*out_width ] = tmp;
        }
    }


    dest[(out_width*(input_row)+input_col   )+ (9*input_channel)*out_width*out_width ] = tmp;
    if(input_col >= 1){
        dest[(out_width*(input_row)+input_col-1 )+ (9*input_channel+1)*out_width*out_width ] = tmp;
        if(input_col >= 2)
            dest[(out_width*(input_row)+input_col-2 )+ (9*input_channel+2)*out_width*out_width ] = tmp;
    }

    
}



float* conv_col(const int input_depth, const int input_width,  const int filter_num, const int out_width,    const int filter_width,    const int padding, const int stride, const int dilation,    double* filter, double* bias,    double* input, const bool clip=true){

    //大致流程：先调用img2col函数重排input
    //然后调用经过加入了output padding功能的gemm进行计算
    in_width = input_width;
    


    return NULL;

}