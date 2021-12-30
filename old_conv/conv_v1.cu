#include "conv_v1.cuh"
#define BLOCKSIZE 32
using namespace std;

/**
 * the output shape: (out_channel, out_width, out_width)
 * parallel: (out_width, out_width) with maximum #thread per block
 * with fusion of conv and clip
*/

__global__ void convKernel(float *input, float *filter, float *output, float *bias, int filter_width, int filter_num, int input_width, int input_depth, int stride, int out_width, bool clip)
{
    int output_row = blockIdx.x * blockDim.x + threadIdx.x;
    int output_col = blockIdx.y * blockDim.y + threadIdx.y;
    int input_row = output_row * stride;
    int input_col = output_col * stride;
    if (output_row >= out_width || output_col >= out_width)
        return;

    for (int fnum = 0; fnum < filter_num; fnum++)
    {
        float tmp = 0.0;
        for (int d = 0; d < input_depth; d++)
            for (int r = 0; r < filter_width; r++)
                for (int c = 0; c < filter_width; c++)
                    tmp += input[d * input_width * input_width + (input_row + r) * input_width + input_col + c] * filter[fnum * input_depth * filter_width * filter_width + d * filter_width * filter_width + r * filter_width + c];

        tmp += bias[fnum];
        if (clip)
        {
            if (tmp < 0)
                tmp = 0.0;
            else if (tmp > 6)
                tmp = 6.0;
        }
        output[fnum * out_width * out_width + output_row * out_width + output_col] = tmp;
    }
}

/* the float*  are assumed to point to cuda mem. */
float *conv(const int input_depth, const int input_width,
            const int filter_num, const int out_width_,
            const int filter_width,
            const int padding, const int stride, const int dilation,
            float *filter, float *bias,
            float *input, const bool clip)
{
    // printf("========== conv_v1::begin conv ==========\n");
    float *img_cuda, *filter_cuda, *output_cuda, *bias_cuda; //img_cuda for padded tensor.
    int in_width = input_width + padding * 2;
    int out_width = (in_width - filter_width) / stride + 1;

    cudaMalloc(&img_cuda, sizeof(float) * input_depth * in_width * in_width);
    // cudaMalloc(&filter_cuda,sizeof(float)*filter_num*input_depth*filter_width*filter_width);
    cudaMalloc(&output_cuda, sizeof(float) * filter_num * out_width * out_width);
    // cudaMalloc(&bias_cuda,sizeof(float)*filter_num);
    // cudaMemcpy(filter_cuda, filter, sizeof(float)*filter_num*input_depth*filter_width*filter_width, cudaMemcpyHostToDevice);
    filter_cuda = filter;
    // cudaMemcpy(bias_cuda, bias, sizeof(float)*filter_num, cudaMemcpyHostToDevice);
    bias_cuda = bias;

    for (int mapid = 0; mapid < input_depth; mapid++)
        for (int line = 0; line < input_width; line++)
        {
            cudaMemcpy(&img_cuda[mapid * in_width * in_width + (padding + line) * in_width + padding], &input[mapid * input_width * input_width + line * input_width], sizeof(float) * input_width, cudaMemcpyDeviceToDevice);
        }

    int g = (out_width + BLOCKSIZE - 1) / BLOCKSIZE;
    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    dim3 grid(g, g);

    convKernel<<<grid, threads>>>(img_cuda, filter_cuda, output_cuda, bias_cuda, filter_width, filter_num, in_width, input_depth, stride, out_width, clip);
    // cudaDeviceSynchronize();
    // cudaMemcpy(output, output_cuda, sizeof(float)*32*122*122, cudaMemcpyDeviceToHost);

    cudaFree(img_cuda);
    cudaFree(input);
    // cudaFree(filter_cuda);
    // cudaFree(output_cuda);
    // cudaFree(bias_cuda);

    // printf("========== conv_v1::end conv ==========\n");
    return output_cuda;
}

int test_conv1_main()
{
    float *filter = new float[2 * 3 * 3 * 3];
    float *image = new float[3 * 5 * 5];
    float *out = new float[2 * 25];
    float *bias = new float[2];
    for (int i = 0; i < 2 * 3 * 3 * 3; i++)
        filter[i] = 1;
    for (int i = 0; i < 3 * 5 * 5; i++)
        image[i] = 1;
    bias[0] = 100;
    bias[1] = 10000;
    float *filter_cu, *image_cu, *out_cu, *bias_cu;
    cudaMalloc(&filter_cu, 2 * 3 * 3 * 3 * sizeof(float));
    cudaMemcpy(filter_cu, filter, 2 * 3 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&image_cu, 3 * 5 * 5 * sizeof(float));
    cudaMemcpy(image_cu, image, 3 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&bias_cu, 2 * sizeof(float));
    cudaMemcpy(bias_cu, bias, 2 * sizeof(float), cudaMemcpyHostToDevice);

    out_cu = conv(3, 5, 2, 5, 3, 1, 1, 0, filter_cu, bias_cu, image_cu, false);
    cudaMemcpy(out, out_cu, 50 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 50; i++)
        printf("%f\n", out[i]);
    return 0;
}
// int main(){test_conv1_main(); return 0;}