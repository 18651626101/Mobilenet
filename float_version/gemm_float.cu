#include "gemm_float.cuh"
struct Matrix
{
	int width;
	int height;
	float *elements;
};

__device__ float getElement(Matrix *A, int row, int col)
{
	return A->elements[row * A->width + col];
}

__device__ void setElement(Matrix *A, int row, int col, float value)
{
	A->elements[row * A->width + col] = value;
}

__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C, Matrix *D)
{
	float Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = 0; i < A->width; ++i)
	{
		Cvalue += getElement(A, row, i) * getElement(B, i, col);
	}
	Cvalue += getElement(D, row, col);
	setElement(C, row, col, Cvalue);
}

float *gemm(const int input_width, const int input_height, const int weight_width, const int weight_height,
			float *weight, float *bias, float *input)
{
	Matrix *A, *B, *C, *D;

	cudaMallocManaged((void **)&A, sizeof(Matrix));
	cudaMallocManaged((void **)&B, sizeof(Matrix));
	cudaMallocManaged((void **)&C, sizeof(Matrix));
	cudaMallocManaged((void **)&D, sizeof(Matrix));

	A->width = input_width;
	A->height = input_height;
	B->width = weight_width;
	B->height = weight_height;
	C->width = B->width;
	C->height = A->height;
	D->width = B->width;
	D->height = A->height;
	A->elements = input;
	B->elements = weight;
	D->elements = bias;

	cudaMallocManaged((void **)&C->elements, C->width * C->height * sizeof(float));

	dim3 blockSize(32, 32);
	dim3 gridSize((C->width + blockSize.x - 1) / blockSize.x,
				  (C->height + blockSize.y - 1) / blockSize.y);

	struct timeval t1, t2;
	gettimeofday(&t1, NULL);
	float timeuse;

	matMulKernel<<<gridSize, blockSize>>>(A, B, C, D);

	// cudaDeviceSynchronize();

	gettimeofday(&t2, NULL);
	timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0;
	printf("Use Time:%fs\n", timeuse);

	return C->elements;
}

int test_gemm_main()
{
	float *test_inputd;
	float test_input[1 * 2];
	float *test_weightd;
	float test_weight[2 * 3];
	float *test_biasd;
	float test_bias[3];
	float *test_outputd;
	float test_output[3];

	for (int i = 0; i < 2; i++)
	{
		test_input[i] = float(i);
		printf("input %f\n", test_input[i]);
	}
	for (int i = 0; i < 2 * 3; i++)
	{
		test_weight[i] = float(i);
		printf("weight %f\n", test_weight[i]);
	}
	for (int i = 0; i < 3; i++)
	{
		test_bias[i] = float(1);
		printf("bias %f\n", test_bias[i]);
	}

	cudaMalloc(&test_inputd, sizeof(float) * 2);
	cudaMemcpy(test_inputd, test_input, sizeof(float) * 2, cudaMemcpyHostToDevice);
	cudaMalloc(&test_weightd, sizeof(float) * 2 * 3);
	cudaMemcpy(test_weightd, test_weight, sizeof(float) * 2 * 3, cudaMemcpyHostToDevice);
	cudaMalloc(&test_biasd, sizeof(float) * 3);
	cudaMemcpy(test_biasd, test_bias, sizeof(float) * 3, cudaMemcpyHostToDevice);

	test_outputd = gemm(2, 1, 3, 2, test_weightd, test_biasd, test_inputd);

	cudaMemcpy(test_output, test_outputd, sizeof(float) * 3, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 3; i++)
	{
		printf("%f\n", test_output[i]);
	}
	return 0;
}

// int test_gemm_main()
// {
// 	int width = w;
// 	int height = w;

// 	Matrix *A, *B, *C;

// 	cudaMallocManaged((void**)&A, sizeof(Matrix));
// 	cudaMallocManaged((void**)&B, sizeof(Matrix));
// 	cudaMallocManaged((void**)&C, sizeof(Matrix));

// 	int nBytes = width * height * sizeof(float);

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
// 	float timeuse;

// 	matMulKernel << < gridSize, blockSize >> >(A, B, C);

// 	cudaDeviceSynchronize();

// 	gettimeofday(&t2,NULL);
// 	timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
// 	printf("Use Time:%fs\n", timeuse);

// 	return 0;
// }