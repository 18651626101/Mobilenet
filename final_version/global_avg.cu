#include "global_avg.cuh"
#define BLOCKSIZE 32
using namespace std;

__global__ void GlobalAvgKernel(double *input, double *output, int depth, int width)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    double amount = 0;
    for (int i = 0; i < width * width; i++)
    {
        amount += input[k * width * width + i];
    }
    output[k] = amount / (width * width);
}

double *global_avg(double *input, int depth, int width)
{
    // printf("========== global_avg:: begin global average pooling ==========\n");
    double *output;
    int grid_sz = depth / BLOCKSIZE;
    if (depth % BLOCKSIZE)
        grid_sz++;

    //alloc memory for output
    cudaMalloc(&output, sizeof(double) * depth);

    //invoke function on device
    dim3 dimGrid(grid_sz);
    dim3 dimBlock(BLOCKSIZE);
    GlobalAvgKernel<<<dimGrid, dimBlock>>>((double *)input, (double *)output, depth, width);

    //return result
    cudaFree(input);
    // printf("========== global_avg:: end global average pooling ==========\n");
    return output;
}

// int main()
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