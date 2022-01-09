#include "global_avg.cuh"
#define BLOCKSIZE 32
using namespace std;

/* 
 * GlobalAvgKernel: 平均池化操作的卷积核函数
 */
__global__ void GlobalAvgKernel(double *input, double *output, int depth, int width)
{
    // 一个thread计算一个channel对应的值
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    double amount = 0;
    for (int i = 0; i < width * width; i++)
    {
        amount += input[k * width * width + i];
    }
    output[k] = amount / (width * width);
}

/* 
 * global_avg: 平均池化操作
 * params:
 *   input: 输入，维度为 depth * width * width 
 *   depth: 输入的channel个数
 *   width: 输入的宽度
 *   output: 返回值, 维度为 depth * 1 * 1
 */
double *global_avg(double *input, int depth, int width)
{
    double *output;
    int grid_sz = depth / BLOCKSIZE;
    if (depth % BLOCKSIZE)
        grid_sz++;

    cudaMalloc(&output, sizeof(double) * depth);

    dim3 dimGrid(grid_sz);
    dim3 dimBlock(BLOCKSIZE);
    GlobalAvgKernel<<<dimGrid, dimBlock>>>((double *)input, (double *)output, depth, width);

    cudaFree(input);
    return output;
}

/* 调试函数 */
int test_global_main()
{
    double *test_inputd;
    double test_input[3 * 2 * 2];
    double *test_outputd;
    double test_output[3];

    for (int i = 0; i < 12; i++)
    {
        test_input[i] = double(i);
        printf("%f\n", test_input[i]);
    }

    cudaMalloc(&test_inputd, sizeof(double) * 3 * 2 * 2);
    cudaMemcpy(test_inputd, test_input, sizeof(double) * 3 * 2 * 2, cudaMemcpyHostToDevice);

    test_outputd = global_avg(test_inputd, 3, 2);

    cudaMemcpy(test_output, test_outputd, sizeof(double) * 3, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 3; i++)
    {
        printf("%f\n", test_output[i]);
    }
    return 0;
}