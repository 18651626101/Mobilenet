#include "add.cuh"
#define BLOCKSIZE 32
using namespace std;

__global__ void AddKernel(const int size, double *inputA, double *inputB)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    inputA[k] += inputB[k];

}

void add(const int channel, const int shape, double *inputA, double *inputB)
{
    // printf("========== add:: begin add ==========\n");
    int size = channel*shape*shape;
    int grid_sz = size / BLOCKSIZE;
    if (size % BLOCKSIZE)
        grid_sz++;

    //invoke function on device
    dim3 dimGrid(grid_sz);
    dim3 dimBlock(BLOCKSIZE);
    AddKernel<<<dimGrid, dimBlock>>>(size, (double *)inputA, (double *)inputB);

    //return result
    cudaFree(inputB);
    // printf("========== add:: end add ==========\n");
}

// int main()
int test_add_main()
{
    double *test_inputda;
    double test_inputa[3 * 2 * 2];
    double *test_inputdb;
    double test_inputb[3 * 2 * 2];
    // double *test_outputd;
    double test_output[3 * 2 * 2];

    for (int i = 0; i < 12; i++)
    {
        test_inputa[i] = double(i);
        test_inputb[i] = double(i);
        printf("%f %f\n", test_inputa[i], test_inputb[i]);
    }

    cudaMalloc(&test_inputda, sizeof(double) * 3 * 2 * 2);
    cudaMemcpy(test_inputda, test_inputa, sizeof(double) * 3 * 2 * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&test_inputdb, sizeof(double) * 3 * 2 * 2);
    cudaMemcpy(test_inputdb, test_inputb, sizeof(double) * 3 * 2 * 2, cudaMemcpyHostToDevice);   
    
    add(3, 2, test_inputda, test_inputdb);

    cudaMemcpy(test_output, test_inputda, sizeof(double) * 3*2*2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 12; i++)
    {
        printf("%f\n", test_output[i]);
    }
    return 0;
}