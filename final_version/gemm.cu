#include "gemm.cuh"

__global__ void matMulKernel(const int A_height, const int A_width, const int B_height, const int B_width, double *A, double *B, double *C, double *D)
{
	double Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(row<A_height&&col<B_width){
		for (int i = 0; i < A_width; ++i)
		{
			Cvalue += A[row * A_width + i] * B[i*B_width + col];
		}
		Cvalue += D[row * B_width + col];
		C[row * B_width + col] = Cvalue;
	}
}

void gemm(const int A_height, const int A_width, const int B_height, const int B_width,
			 double *MatrixA, double *MatrixB, double *bias, double *out)
{

	dim3 blockSize(32, 32);
	dim3 gridSize((B_width + blockSize.x - 1) / blockSize.x,
				  (A_height + blockSize.y - 1) / blockSize.y);

	matMulKernel<<<gridSize, blockSize>>>(A_height, A_width, B_height, B_width, MatrixA, MatrixB, out, bias);
}

// int main()
// int test_gemm_main()
// {
// 	double *test_inputd;
// 	double test_input[1 * 2];
// 	double *test_weightd;
// 	double test_weight[3 * 2];
// 	double *test_biasd;
// 	double test_bias[3];
// 	double *test_outputd;
// 	double test_output[3];

// 	for (int i = 0; i < 2; i++)
// 	{
// 		test_input[i] = double(i);
// 		printf("input %f\n", test_input[i]);
// 	}
// 	for (int i = 0; i < 2 * 3; i++)
// 	{
// 		test_weight[i] = double(i);
// 		printf("weight %f\n", test_weight[i]);
// 	}
// 	for (int i = 0; i < 3; i++)
// 	{
// 		test_bias[i] = double(i);
// 		printf("bias %f\n", test_bias[i]);
// 	}

// 	cudaMalloc(&test_inputd, sizeof(double) * 2);
// 	cudaMemcpy(test_inputd, test_input, sizeof(double) * 2, cudaMemcpyHostToDevice);
// 	cudaMalloc(&test_weightd, sizeof(double) * 2 * 3);
// 	cudaMemcpy(test_weightd, test_weight, sizeof(double) * 2 * 3, cudaMemcpyHostToDevice);
// 	cudaMalloc(&test_biasd, sizeof(double) * 3);
// 	cudaMemcpy(test_biasd, test_bias, sizeof(double) * 3, cudaMemcpyHostToDevice);

// 	// test_outputd = gemm(3, 2, 2, 1, test_weightd, test_inputd,test_biasd);

// 	cudaMemcpy(test_output, test_outputd, sizeof(double) * 3, cudaMemcpyDeviceToHost);
// 	for (int i = 0; i < 3; i++)
// 	{
// 		printf("%f\n", test_output[i]);
// 	}
// 	return 0;
// // }

// int test_gemm_main()
// {
// 	int width = w;
// 	int height = w;

// 	Matrix *A, *B, *C;

// 	cudaMallocManaged((void**)&A, sizeof(Matrix));
// 	cudaMallocManaged((void**)&B, sizeof(Matrix));
// 	cudaMallocManaged((void**)&C, sizeof(Matrix));

// 	int nBytes = width * height * sizeof(double);

// 	cudaMallocManaged((void**)&A->elements, nBytes);
// 	cudaMallocManaged((void**)&B->elements, nBytes);
// 	cudaMallocManaged((void**)&C->elements, nBytes);

// 	A->height = height;
// 	A->width = width;
// 	B->height = height;
// 	B->width = width;
// 	C->height = height;
// 	C->width = width;

// 	for (int i = 0; i < width * height; ++i)
// 	{
// 		A->elements[i] = 1.0;
// 		B->elements[i] = 2.0;
// 	}

// 	dim3 blockSize(32, 32);
// 	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
// 		(height + blockSize.y - 1) / blockSize.y);

// 	struct timeval t1,t2;
// 	gettimeofday(&t1,NULL);
// 	double timeuse;

// 	matMulKernel << < gridSize, blockSize >> >(A, B, C);

// 	cudaDeviceSynchronize();

// 	gettimeofday(&t2,NULL);
// 	timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
// 	printf("Use Time:%fs\n", timeuse);

// 	return 0;
// }