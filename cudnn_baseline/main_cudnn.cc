#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

#define INPUTSHAPE 3 * 244 * 244
#define OUTPUTSHAPE 1000
#define TESTNUM 10
#define ITERNUM 50
#define checkCUDNN(expression)                                  \
  {                                                             \
    cudnnStatus_t status = (expression);                        \
    if (status != CUDNN_STATUS_SUCCESS) {                       \
	    std::cerr << "Error on line " << __LINE__ << ": "       \
	    << cudnnGetErrorString(status) << std::endl;            \
    }                                                           \
 }

double inputArr[TESTNUM][INPUTSHAPE];
double benchOutArr[TESTNUM][OUTPUTSHAPE];

double* Conv_0_W;
double* Conv_0_B;
double* Conv_2_W;
double* Conv_2_B;
double* Conv_4_W;
double* Conv_4_B;
double* Conv_5_W;
double* Conv_5_B;
double* Conv_7_W;
double* Conv_7_B;
double* Conv_9_W;
double* Conv_9_B;
double* Conv_10_W;
double* Conv_10_B;
double* Conv_12_W;
double* Conv_12_B;
double* Conv_14_W;
double* Conv_14_B;
double* Conv_16_W;
double* Conv_16_B;
double* Conv_18_W;
double* Conv_18_B;
double* Conv_20_W;
double* Conv_20_B;
double* Conv_21_W;
double* Conv_21_B;
double* Conv_23_W;
double* Conv_23_B;
double* Conv_25_W;
double* Conv_25_B;
double* Conv_27_W;
double* Conv_27_B;
double* Conv_29_W;
double* Conv_29_B;
double* Conv_31_W;
double* Conv_31_B;
double* Conv_33_W;
double* Conv_33_B;
double* Conv_35_W;
double* Conv_35_B;
double* Conv_37_W;
double* Conv_37_B;
double* Conv_38_W;
double* Conv_38_B;
double* Conv_40_W;
double* Conv_40_B;
double* Conv_42_W;
double* Conv_42_B;
double* Conv_44_W;
double* Conv_44_B;
double* Conv_46_W;
double* Conv_46_B;
double* Conv_48_W;
double* Conv_48_B;
double* Conv_50_W;
double* Conv_50_B;
double* Conv_52_W;
double* Conv_52_B;
double* Conv_54_W;
double* Conv_54_B;
double* Conv_56_W;
double* Conv_56_B;
double* Conv_58_W;
double* Conv_58_B;
double* Conv_60_W;
double* Conv_60_B;
double* Conv_61_W;
double* Conv_61_B;
double* Conv_63_W;
double* Conv_63_B;
double* Conv_65_W;
double* Conv_65_B;
double* Conv_67_W;
double* Conv_67_B;
double* Conv_69_W;
double* Conv_69_B;
double* Conv_71_W;
double* Conv_71_B;
double* Conv_73_W;
double* Conv_73_B;
double* Conv_75_W;
double* Conv_75_B;
double* Conv_77_W;
double* Conv_77_B;
double* Conv_78_W;
double* Conv_78_B;
double* Conv_80_W;
double* Conv_80_B;
double* Conv_82_W;
double* Conv_82_B;
double* Conv_84_W;
double* Conv_84_B;
double* Conv_86_W;
double* Conv_86_B;
double* Conv_88_W;
double* Conv_88_B;
double* Conv_90_W;
double* Conv_90_B;
double* Conv_92_W;
double* Conv_92_B;
double* Conv_94_W;
double* Conv_94_B;
double* Conv_95_W;
double* Conv_95_B;
double* Gemm_W;
double* Gemm_B;
double* temp_h;
double* temp_d;

double* Conv_0_out;
double* Conv_2_out;
double* Conv_4_out;
double* Conv_5_out;
double* Conv_7_out;
double* Conv_9_out;
double* Conv_10_out;
double* Conv_12_out;
double* Conv_14_out;
double* Conv_16_out;
double* Conv_18_out;
double* Conv_20_out;
double* Conv_21_out;
double* Conv_23_out;
double* Conv_25_out;
double* Conv_27_out;
double* Conv_29_out;
double* Conv_31_out;
double* Conv_33_out;
double* Conv_35_out;
double* Conv_37_out;
double* Conv_38_out;
double* Conv_40_out;
double* Conv_42_out;
double* Conv_44_out;
double* Conv_46_out;
double* Conv_48_out;
double* Conv_50_out;
double* Conv_52_out;
double* Conv_54_out;
double* Conv_56_out;
double* Conv_58_out;
double* Conv_60_out;
double* Conv_61_out;
double* Conv_63_out;
double* Conv_65_out;
double* Conv_67_out;
double* Conv_69_out;
double* Conv_71_out;
double* Conv_73_out;
double* Conv_75_out;
double* Conv_77_out;
double* Conv_78_out;
double* Conv_80_out;
double* Conv_82_out;
double* Conv_84_out;
double* Conv_86_out;
double* Conv_88_out;
double* Conv_90_out;
double* Conv_92_out;
double* Conv_94_out;
double* Conv_95_out;
double* Avgpool_out;
double* Gemm_out;
void readInput(char *filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < INPUTSHAPE; j++)
            fscanf(fp, "%lf", &inputArr[i][j]);
}

void readOutput(char *filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < OUTPUTSHAPE; j++)
            fscanf(fp, "%lf", &benchOutArr[i][j]);
}

void checkOutput(double *out1, double *out2)
{
    double maxDiff = 0;
    for (int i = 0; i < OUTPUTSHAPE; i++)
    {
        maxDiff = (fabs(out1[i] - out2[i]) > maxDiff) ? fabs(out1[i] - out2[i]) : maxDiff;
    }
    if (maxDiff > 1e-5)
    {
        printf("Output dismatch. MaxDiff is %.7f\n", maxDiff);
    }
}

// TODO: 读取权重
void initModel()
{
    FILE *fp = NULL;
    fp = fopen("./params.txt", "r");
    temp_h = new double[32*3*3*3];
    for(int i=0;i<32*3*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*32*3*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*32*3*3*3, cudaMemcpyHostToDevice);
    Conv_0_W = temp_d;
    delete(temp_h);
    temp_h = new double[32];
    for(int i=0;i<32;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*32);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*32, cudaMemcpyHostToDevice);
    Conv_0_B = temp_d;
    delete(temp_h);
    temp_h = new double[32*1*3*3];
    for(int i=0;i<32*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*32*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*32*1*3*3, cudaMemcpyHostToDevice);
    Conv_2_W = temp_d;
    delete(temp_h);
    temp_h = new double[32];
    for(int i=0;i<32;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*32);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*32, cudaMemcpyHostToDevice);
    Conv_2_B = temp_d;
    delete(temp_h);
    temp_h = new double[16*32*1*1];
    for(int i=0;i<16*32*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*16*32*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*16*32*1*1, cudaMemcpyHostToDevice);
    Conv_4_W = temp_d;
    delete(temp_h);
    temp_h = new double[16];
    for(int i=0;i<16;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*16);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*16, cudaMemcpyHostToDevice);
    Conv_4_B = temp_d;
    delete(temp_h);
    temp_h = new double[96*16*1*1];
    for(int i=0;i<96*16*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*96*16*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*96*16*1*1, cudaMemcpyHostToDevice);
    Conv_5_W = temp_d;
    delete(temp_h);
    temp_h = new double[96];
    for(int i=0;i<96;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*96);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*96, cudaMemcpyHostToDevice);
    Conv_5_B = temp_d;
    delete(temp_h);
    temp_h = new double[96*1*3*3];
    for(int i=0;i<96*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*96*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*96*1*3*3, cudaMemcpyHostToDevice);
    Conv_7_W = temp_d;
    delete(temp_h);
    temp_h = new double[96];
    for(int i=0;i<96;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*96);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*96, cudaMemcpyHostToDevice);
    Conv_7_B = temp_d;
    delete(temp_h);
    temp_h = new double[24*96*1*1];
    for(int i=0;i<24*96*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*24*96*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*24*96*1*1, cudaMemcpyHostToDevice);
    Conv_9_W = temp_d;
    delete(temp_h);
    temp_h = new double[24];
    for(int i=0;i<24;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*24);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*24, cudaMemcpyHostToDevice);
    Conv_9_B = temp_d;
    delete(temp_h);
    temp_h = new double[144*24*1*1];
    for(int i=0;i<144*24*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*144*24*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*144*24*1*1, cudaMemcpyHostToDevice);
    Conv_10_W = temp_d;
    delete(temp_h);
    temp_h = new double[144];
    for(int i=0;i<144;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*144);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*144, cudaMemcpyHostToDevice);
    Conv_10_B = temp_d;
    delete(temp_h);
    temp_h = new double[144*1*3*3];
    for(int i=0;i<144*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*144*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*144*1*3*3, cudaMemcpyHostToDevice);
    Conv_12_W = temp_d;
    delete(temp_h);
    temp_h = new double[144];
    for(int i=0;i<144;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*144);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*144, cudaMemcpyHostToDevice);
    Conv_12_B = temp_d;
    delete(temp_h);
    temp_h = new double[24*144*1*1];
    for(int i=0;i<24*144*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*24*144*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*24*144*1*1, cudaMemcpyHostToDevice);
    Conv_14_W = temp_d;
    delete(temp_h);
    temp_h = new double[24];
    for(int i=0;i<24;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*24);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*24, cudaMemcpyHostToDevice);
    Conv_14_B = temp_d;
    delete(temp_h);
    temp_h = new double[144*24*1*1];
    for(int i=0;i<144*24*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*144*24*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*144*24*1*1, cudaMemcpyHostToDevice);
    Conv_16_W = temp_d;
    delete(temp_h);
    temp_h = new double[144];
    for(int i=0;i<144;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*144);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*144, cudaMemcpyHostToDevice);
    Conv_16_B = temp_d;
    delete(temp_h);
    temp_h = new double[144*1*3*3];
    for(int i=0;i<144*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*144*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*144*1*3*3, cudaMemcpyHostToDevice);
    Conv_18_W = temp_d;
    delete(temp_h);
    temp_h = new double[144];
    for(int i=0;i<144;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*144);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*144, cudaMemcpyHostToDevice);
    Conv_18_B = temp_d;
    delete(temp_h);
    temp_h = new double[32*144*1*1];
    for(int i=0;i<32*144*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*32*144*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*32*144*1*1, cudaMemcpyHostToDevice);
    Conv_20_W = temp_d;
    delete(temp_h);
    temp_h = new double[32];
    for(int i=0;i<32;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*32);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*32, cudaMemcpyHostToDevice);
    Conv_20_B = temp_d;
    delete(temp_h);
    temp_h = new double[192*32*1*1];
    for(int i=0;i<192*32*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*192*32*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*192*32*1*1, cudaMemcpyHostToDevice);
    Conv_21_W = temp_d;
    delete(temp_h);
    temp_h = new double[192];
    for(int i=0;i<192;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*192);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*192, cudaMemcpyHostToDevice);
    Conv_21_B = temp_d;
    delete(temp_h);
    temp_h = new double[192*1*3*3];
    for(int i=0;i<192*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*192*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*192*1*3*3, cudaMemcpyHostToDevice);
    Conv_23_W = temp_d;
    delete(temp_h);
    temp_h = new double[192];
    for(int i=0;i<192;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*192);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*192, cudaMemcpyHostToDevice);
    Conv_23_B = temp_d;
    delete(temp_h);
    temp_h = new double[32*192*1*1];
    for(int i=0;i<32*192*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*32*192*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*32*192*1*1, cudaMemcpyHostToDevice);
    Conv_25_W = temp_d;
    delete(temp_h);
    temp_h = new double[32];
    for(int i=0;i<32;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*32);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*32, cudaMemcpyHostToDevice);
    Conv_25_B = temp_d;
    delete(temp_h);
    temp_h = new double[192*32*1*1];
    for(int i=0;i<192*32*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*192*32*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*192*32*1*1, cudaMemcpyHostToDevice);
    Conv_27_W = temp_d;
    delete(temp_h);
    temp_h = new double[192];
    for(int i=0;i<192;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*192);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*192, cudaMemcpyHostToDevice);
    Conv_27_B = temp_d;
    delete(temp_h);
    temp_h = new double[192*1*3*3];
    for(int i=0;i<192*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*192*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*192*1*3*3, cudaMemcpyHostToDevice);
    Conv_29_W = temp_d;
    delete(temp_h);
    temp_h = new double[192];
    for(int i=0;i<192;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*192);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*192, cudaMemcpyHostToDevice);
    Conv_29_B = temp_d;
    delete(temp_h);
    temp_h = new double[32*192*1*1];
    for(int i=0;i<32*192*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*32*192*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*32*192*1*1, cudaMemcpyHostToDevice);
    Conv_31_W = temp_d;
    delete(temp_h);
    temp_h = new double[32];
    for(int i=0;i<32;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*32);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*32, cudaMemcpyHostToDevice);
    Conv_31_B = temp_d;
    delete(temp_h);
    temp_h = new double[192*32*1*1];
    for(int i=0;i<192*32*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*192*32*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*192*32*1*1, cudaMemcpyHostToDevice);
    Conv_33_W = temp_d;
    delete(temp_h);
    temp_h = new double[192];
    for(int i=0;i<192;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*192);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*192, cudaMemcpyHostToDevice);
    Conv_33_B = temp_d;
    delete(temp_h);
    temp_h = new double[192*1*3*3];
    for(int i=0;i<192*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*192*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*192*1*3*3, cudaMemcpyHostToDevice);
    Conv_35_W = temp_d;
    delete(temp_h);
    temp_h = new double[192];
    for(int i=0;i<192;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*192);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*192, cudaMemcpyHostToDevice);
    Conv_35_B = temp_d;
    delete(temp_h);
    temp_h = new double[64*192*1*1];
    for(int i=0;i<64*192*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*64*192*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*64*192*1*1, cudaMemcpyHostToDevice);
    Conv_37_W = temp_d;
    delete(temp_h);
    temp_h = new double[64];
    for(int i=0;i<64;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*64);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*64, cudaMemcpyHostToDevice);
    Conv_37_B = temp_d;
    delete(temp_h);
    temp_h = new double[384*64*1*1];
    for(int i=0;i<384*64*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384*64*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384*64*1*1, cudaMemcpyHostToDevice);
    Conv_38_W = temp_d;
    delete(temp_h);
    temp_h = new double[384];
    for(int i=0;i<384;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384, cudaMemcpyHostToDevice);
    Conv_38_B = temp_d;
    delete(temp_h);
    temp_h = new double[384*1*3*3];
    for(int i=0;i<384*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384*1*3*3, cudaMemcpyHostToDevice);
    Conv_40_W = temp_d;
    delete(temp_h);
    temp_h = new double[384];
    for(int i=0;i<384;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384, cudaMemcpyHostToDevice);
    Conv_40_B = temp_d;
    delete(temp_h);
    temp_h = new double[64*384*1*1];
    for(int i=0;i<64*384*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*64*384*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*64*384*1*1, cudaMemcpyHostToDevice);
    Conv_42_W = temp_d;
    delete(temp_h);
    temp_h = new double[64];
    for(int i=0;i<64;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*64);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*64, cudaMemcpyHostToDevice);
    Conv_42_B = temp_d;
    delete(temp_h);
    temp_h = new double[384*64*1*1];
    for(int i=0;i<384*64*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384*64*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384*64*1*1, cudaMemcpyHostToDevice);
    Conv_44_W = temp_d;
    delete(temp_h);
    temp_h = new double[384];
    for(int i=0;i<384;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384, cudaMemcpyHostToDevice);
    Conv_44_B = temp_d;
    delete(temp_h);
    temp_h = new double[384*1*3*3];
    for(int i=0;i<384*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384*1*3*3, cudaMemcpyHostToDevice);
    Conv_46_W = temp_d;
    delete(temp_h);
    temp_h = new double[384];
    for(int i=0;i<384;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384, cudaMemcpyHostToDevice);
    Conv_46_B = temp_d;
    delete(temp_h);
    temp_h = new double[64*384*1*1];
    for(int i=0;i<64*384*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*64*384*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*64*384*1*1, cudaMemcpyHostToDevice);
    Conv_48_W = temp_d;
    delete(temp_h);
    temp_h = new double[64];
    for(int i=0;i<64;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*64);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*64, cudaMemcpyHostToDevice);
    Conv_48_B = temp_d;
    delete(temp_h);
    temp_h = new double[384*64*1*1];
    for(int i=0;i<384*64*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384*64*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384*64*1*1, cudaMemcpyHostToDevice);
    Conv_50_W = temp_d;
    delete(temp_h);
    temp_h = new double[384];
    for(int i=0;i<384;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384, cudaMemcpyHostToDevice);
    Conv_50_B = temp_d;
    delete(temp_h);
    temp_h = new double[384*1*3*3];
    for(int i=0;i<384*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384*1*3*3, cudaMemcpyHostToDevice);
    Conv_52_W = temp_d;
    delete(temp_h);
    temp_h = new double[384];
    for(int i=0;i<384;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384, cudaMemcpyHostToDevice);
    Conv_52_B = temp_d;
    delete(temp_h);
    temp_h = new double[64*384*1*1];
    for(int i=0;i<64*384*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*64*384*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*64*384*1*1, cudaMemcpyHostToDevice);
    Conv_54_W = temp_d;
    delete(temp_h);
    temp_h = new double[64];
    for(int i=0;i<64;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*64);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*64, cudaMemcpyHostToDevice);
    Conv_54_B = temp_d;
    delete(temp_h);
    temp_h = new double[384*64*1*1];
    for(int i=0;i<384*64*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384*64*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384*64*1*1, cudaMemcpyHostToDevice);
    Conv_56_W = temp_d;
    delete(temp_h);
    temp_h = new double[384];
    for(int i=0;i<384;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384, cudaMemcpyHostToDevice);
    Conv_56_B = temp_d;
    delete(temp_h);
    temp_h = new double[384*1*3*3];
    for(int i=0;i<384*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384*1*3*3, cudaMemcpyHostToDevice);
    Conv_58_W = temp_d;
    delete(temp_h);
    temp_h = new double[384];
    for(int i=0;i<384;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*384);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*384, cudaMemcpyHostToDevice);
    Conv_58_B = temp_d;
    delete(temp_h);
    temp_h = new double[96*384*1*1];
    for(int i=0;i<96*384*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*96*384*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*96*384*1*1, cudaMemcpyHostToDevice);
    Conv_60_W = temp_d;
    delete(temp_h);
    temp_h = new double[96];
    for(int i=0;i<96;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*96);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*96, cudaMemcpyHostToDevice);
    Conv_60_B = temp_d;
    delete(temp_h);
    temp_h = new double[576*96*1*1];
    for(int i=0;i<576*96*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*576*96*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*576*96*1*1, cudaMemcpyHostToDevice);
    Conv_61_W = temp_d;
    delete(temp_h);
    temp_h = new double[576];
    for(int i=0;i<576;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*576);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*576, cudaMemcpyHostToDevice);
    Conv_61_B = temp_d;
    delete(temp_h);
    temp_h = new double[576*1*3*3];
    for(int i=0;i<576*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*576*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*576*1*3*3, cudaMemcpyHostToDevice);
    Conv_63_W = temp_d;
    delete(temp_h);
    temp_h = new double[576];
    for(int i=0;i<576;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*576);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*576, cudaMemcpyHostToDevice);
    Conv_63_B = temp_d;
    delete(temp_h);
    temp_h = new double[96*576*1*1];
    for(int i=0;i<96*576*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*96*576*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*96*576*1*1, cudaMemcpyHostToDevice);
    Conv_65_W = temp_d;
    delete(temp_h);
    temp_h = new double[96];
    for(int i=0;i<96;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*96);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*96, cudaMemcpyHostToDevice);
    Conv_65_B = temp_d;
    delete(temp_h);
    temp_h = new double[576*96*1*1];
    for(int i=0;i<576*96*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*576*96*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*576*96*1*1, cudaMemcpyHostToDevice);
    Conv_67_W = temp_d;
    delete(temp_h);
    temp_h = new double[576];
    for(int i=0;i<576;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*576);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*576, cudaMemcpyHostToDevice);
    Conv_67_B = temp_d;
    delete(temp_h);
    temp_h = new double[576*1*3*3];
    for(int i=0;i<576*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*576*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*576*1*3*3, cudaMemcpyHostToDevice);
    Conv_69_W = temp_d;
    delete(temp_h);
    temp_h = new double[576];
    for(int i=0;i<576;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*576);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*576, cudaMemcpyHostToDevice);
    Conv_69_B = temp_d;
    delete(temp_h);
    temp_h = new double[96*576*1*1];
    for(int i=0;i<96*576*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*96*576*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*96*576*1*1, cudaMemcpyHostToDevice);
    Conv_71_W = temp_d;
    delete(temp_h);
    temp_h = new double[96];
    for(int i=0;i<96;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*96);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*96, cudaMemcpyHostToDevice);
    Conv_71_B = temp_d;
    delete(temp_h);
    temp_h = new double[576*96*1*1];
    for(int i=0;i<576*96*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*576*96*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*576*96*1*1, cudaMemcpyHostToDevice);
    Conv_73_W = temp_d;
    delete(temp_h);
    temp_h = new double[576];
    for(int i=0;i<576;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*576);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*576, cudaMemcpyHostToDevice);
    Conv_73_B = temp_d;
    delete(temp_h);
    temp_h = new double[576*1*3*3];
    for(int i=0;i<576*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*576*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*576*1*3*3, cudaMemcpyHostToDevice);
    Conv_75_W = temp_d;
    delete(temp_h);
    temp_h = new double[576];
    for(int i=0;i<576;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*576);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*576, cudaMemcpyHostToDevice);
    Conv_75_B = temp_d;
    delete(temp_h);
    temp_h = new double[160*576*1*1];
    for(int i=0;i<160*576*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*160*576*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*160*576*1*1, cudaMemcpyHostToDevice);
    Conv_77_W = temp_d;
    delete(temp_h);
    temp_h = new double[160];
    for(int i=0;i<160;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*160);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*160, cudaMemcpyHostToDevice);
    Conv_77_B = temp_d;
    delete(temp_h);
    temp_h = new double[960*160*1*1];
    for(int i=0;i<960*160*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*960*160*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*960*160*1*1, cudaMemcpyHostToDevice);
    Conv_78_W = temp_d;
    delete(temp_h);
    temp_h = new double[960];
    for(int i=0;i<960;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*960);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*960, cudaMemcpyHostToDevice);
    Conv_78_B = temp_d;
    delete(temp_h);
    temp_h = new double[960*1*3*3];
    for(int i=0;i<960*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*960*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*960*1*3*3, cudaMemcpyHostToDevice);
    Conv_80_W = temp_d;
    delete(temp_h);
    temp_h = new double[960];
    for(int i=0;i<960;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*960);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*960, cudaMemcpyHostToDevice);
    Conv_80_B = temp_d;
    delete(temp_h);
    temp_h = new double[160*960*1*1];
    for(int i=0;i<160*960*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*160*960*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*160*960*1*1, cudaMemcpyHostToDevice);
    Conv_82_W = temp_d;
    delete(temp_h);
    temp_h = new double[160];
    for(int i=0;i<160;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*160);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*160, cudaMemcpyHostToDevice);
    Conv_82_B = temp_d;
    delete(temp_h);
    temp_h = new double[960*160*1*1];
    for(int i=0;i<960*160*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*960*160*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*960*160*1*1, cudaMemcpyHostToDevice);
    Conv_84_W = temp_d;
    delete(temp_h);
    temp_h = new double[960];
    for(int i=0;i<960;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*960);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*960, cudaMemcpyHostToDevice);
    Conv_84_B = temp_d;
    delete(temp_h);
    temp_h = new double[960*1*3*3];
    for(int i=0;i<960*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*960*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*960*1*3*3, cudaMemcpyHostToDevice);
    Conv_86_W = temp_d;
    delete(temp_h);
    temp_h = new double[960];
    for(int i=0;i<960;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*960);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*960, cudaMemcpyHostToDevice);
    Conv_86_B = temp_d;
    delete(temp_h);
    temp_h = new double[160*960*1*1];
    for(int i=0;i<160*960*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*160*960*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*160*960*1*1, cudaMemcpyHostToDevice);
    Conv_88_W = temp_d;
    delete(temp_h);
    temp_h = new double[160];
    for(int i=0;i<160;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*160);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*160, cudaMemcpyHostToDevice);
    Conv_88_B = temp_d;
    delete(temp_h);
    temp_h = new double[960*160*1*1];
    for(int i=0;i<960*160*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*960*160*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*960*160*1*1, cudaMemcpyHostToDevice);
    Conv_90_W = temp_d;
    delete(temp_h);
    temp_h = new double[960];
    for(int i=0;i<960;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*960);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*960, cudaMemcpyHostToDevice);
    Conv_90_B = temp_d;
    delete(temp_h);
    temp_h = new double[960*1*3*3];
    for(int i=0;i<960*1*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*960*1*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*960*1*3*3, cudaMemcpyHostToDevice);
    Conv_92_W = temp_d;
    delete(temp_h);
    temp_h = new double[960];
    for(int i=0;i<960;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*960);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*960, cudaMemcpyHostToDevice);
    Conv_92_B = temp_d;
    delete(temp_h);
    temp_h = new double[320*960*1*1];
    for(int i=0;i<320*960*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*320*960*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*320*960*1*1, cudaMemcpyHostToDevice);
    Conv_94_W = temp_d;
    delete(temp_h);
    temp_h = new double[320];
    for(int i=0;i<320;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*320);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*320, cudaMemcpyHostToDevice);
    Conv_94_B = temp_d;
    delete(temp_h);
    temp_h = new double[1280*320*1*1];
    for(int i=0;i<1280*320*1*1;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*1280*320*1*1);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*1280*320*1*1, cudaMemcpyHostToDevice);
    Conv_95_W = temp_d;
    delete(temp_h);
    temp_h = new double[1280];
    for(int i=0;i<1280;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*1280);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*1280, cudaMemcpyHostToDevice);
    Conv_95_B = temp_d;
    delete(temp_h);


    double _;
    fscanf(fp,"%lf %lf",&_, &_); // const 1 -1

    temp_h = new double[1280*1000];
    for(int i=0;i<1280*1000;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*1280*1000);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*1280*1000, cudaMemcpyHostToDevice);
    Gemm_W = temp_d;
    delete(temp_h);
    temp_h = new double[1000];
    for(int i=0;i<1000;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*1000);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*1000, cudaMemcpyHostToDevice);
    Gemm_B = temp_d;
    delete(temp_h);

    cudaMalloc(&Conv_0_out,sizeof(double)*32*122*122);
    cudaMalloc(&Conv_2_out,sizeof(double)*32*122*122);
    cudaMalloc(&Conv_4_out,sizeof(double)*16*122*122);
    cudaMalloc(&Conv_5_out,sizeof(double)*96*122*122);
    cudaMalloc(&Conv_7_out,sizeof(double)*96*61*61);
    cudaMalloc(&Conv_9_out,sizeof(double)*24*61*61);
    cudaMalloc(&Conv_10_out,sizeof(double)*144*61*61);
    cudaMalloc(&Conv_12_out,sizeof(double)*144*61*61);
    cudaMalloc(&Conv_14_out,sizeof(double)*24*61*61);
    cudaMalloc(&Conv_16_out,sizeof(double)*144*61*61);
    cudaMalloc(&Conv_18_out,sizeof(double)*144*31*31);
    cudaMalloc(&Conv_20_out,sizeof(double)*32*31*31);
    cudaMalloc(&Conv_21_out,sizeof(double)*192*31*31);
    cudaMalloc(&Conv_23_out,sizeof(double)*192*31*31);
    cudaMalloc(&Conv_25_out,sizeof(double)*32*31*31);
    cudaMalloc(&Conv_27_out,sizeof(double)*192*31*31);
    cudaMalloc(&Conv_29_out,sizeof(double)*192*31*31);
    cudaMalloc(&Conv_31_out,sizeof(double)*32*31*31);
    cudaMalloc(&Conv_33_out,sizeof(double)*192*31*31);
    cudaMalloc(&Conv_35_out,sizeof(double)*192*16*16);
    cudaMalloc(&Conv_37_out,sizeof(double)*64*16*16);
    cudaMalloc(&Conv_38_out,sizeof(double)*384*16*16);
    cudaMalloc(&Conv_40_out,sizeof(double)*384*16*16);
    cudaMalloc(&Conv_42_out,sizeof(double)*64*16*16);
    cudaMalloc(&Conv_44_out,sizeof(double)*384*16*16);
    cudaMalloc(&Conv_46_out,sizeof(double)*384*16*16);
    cudaMalloc(&Conv_48_out,sizeof(double)*64*16*16);
    cudaMalloc(&Conv_50_out,sizeof(double)*384*16*16);
    cudaMalloc(&Conv_52_out,sizeof(double)*384*16*16);
    cudaMalloc(&Conv_54_out,sizeof(double)*64*16*16);
    cudaMalloc(&Conv_56_out,sizeof(double)*384*16*16);
    cudaMalloc(&Conv_58_out,sizeof(double)*384*16*16);
    cudaMalloc(&Conv_60_out,sizeof(double)*96*16*16);
    cudaMalloc(&Conv_61_out,sizeof(double)*576*16*16);
    cudaMalloc(&Conv_63_out,sizeof(double)*576*16*16);
    cudaMalloc(&Conv_65_out,sizeof(double)*96*16*16);
    cudaMalloc(&Conv_67_out,sizeof(double)*576*16*16);
    cudaMalloc(&Conv_69_out,sizeof(double)*576*16*16);
    cudaMalloc(&Conv_71_out,sizeof(double)*96*16*16);
    cudaMalloc(&Conv_73_out,sizeof(double)*576*16*16);
    cudaMalloc(&Conv_75_out,sizeof(double)*576*8*8);
    cudaMalloc(&Conv_77_out,sizeof(double)*160*8*8);
    cudaMalloc(&Conv_78_out,sizeof(double)*960*8*8);
    cudaMalloc(&Conv_80_out,sizeof(double)*960*8*8);
    cudaMalloc(&Conv_82_out,sizeof(double)*160*8*8);
    cudaMalloc(&Conv_84_out,sizeof(double)*960*8*8);
    cudaMalloc(&Conv_86_out,sizeof(double)*960*8*8);
    cudaMalloc(&Conv_88_out,sizeof(double)*160*8*8);
    cudaMalloc(&Conv_90_out,sizeof(double)*960*8*8);
    cudaMalloc(&Conv_92_out,sizeof(double)*960*8*8);
    cudaMalloc(&Conv_94_out,sizeof(double)*320*8*8);
    cudaMalloc(&Conv_95_out,sizeof(double)*1280*8*8);
    cudaMalloc(&Avgpool_out,sizeof(double)*1280);
    cudaMalloc(&Gemm_out,sizeof(double)*1000);
}

void conv_layer(const int x_channel, const int x_shape,
                const int y_channel, const int y_shape,
                const int kernel_shape,
                const int padding, const int stride, const int dilation,
                double* Conv_W, double* Conv_B,
                double* &input_g, double* &output_g, cudnnHandle_t &handle, bool is_clip = true)
{
    int conv_2nd_channel = (x_channel==y_channel)?1:x_channel;

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_DOUBLE,
                               1,x_channel,x_shape,x_shape));
    
     cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_DOUBLE,
                               1,y_channel,y_shape,y_shape));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_DOUBLE,
                               CUDNN_TENSOR_NCHW,
                               y_channel,conv_2nd_channel,kernel_shape,kernel_shape));
    
    cudnnTensorDescriptor_t bias_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_DOUBLE,
                               1,y_channel,1,1));

    
    cudnnConvolutionDescriptor_t conv_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_descriptor,
                                    padding,padding, // zero-padding
                                    stride,stride, // stride
                                    dilation,dilation,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));
    
    checkCUDNN(cudnnSetConvolutionGroupCount(conv_descriptor, (x_channel==y_channel)?x_channel:1));
    
    cudnnConvolutionFwdAlgoPerf_t algo;
    int algo_num;
    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(handle,
                                        input_descriptor,
                                        kernel_descriptor,
                                        conv_descriptor,
                                        output_descriptor,
                                        1,
                                        &algo_num,
                                        &algo));
    
    size_t workspace_size = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                            input_descriptor,
                                            kernel_descriptor,
                                            conv_descriptor,
                                            output_descriptor,
                                            algo.algo,
                                            &workspace_size));
    
    void * workspace = NULL;
    cudaMalloc(&workspace, workspace_size);

    double * Conv_W_g = Conv_W;
    double * Conv_B_g = Conv_B;

    double alpha = 1.0, beta = 0.0, beta_add = 1.0;
    checkCUDNN(cudnnConvolutionForward(handle,
                            &alpha, input_descriptor, input_g,
                            kernel_descriptor, Conv_W_g,
                            conv_descriptor, algo.algo,
                            workspace, workspace_size,
                            &beta, output_descriptor, output_g));
    
    checkCUDNN(cudnnAddTensor(
			handle,
			&alpha,
			bias_descriptor,
			Conv_B_g,
			&beta_add,
			output_descriptor,
			output_g));
    if(is_clip){
    cudnnActivationDescriptor_t activation_descriptor;
	checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
				CUDNN_ACTIVATION_CLIPPED_RELU,
				CUDNN_PROPAGATE_NAN,
				6.0));
    checkCUDNN(cudnnActivationForward(handle,
			activation_descriptor,
			&alpha,
			output_descriptor,
			output_g,
			&beta,
			output_descriptor,
			output_g));
            cudnnDestroyActivationDescriptor(activation_descriptor);
            }
    
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(conv_descriptor);
    cudnnDestroyTensorDescriptor(bias_descriptor);
    input_g = output_g;
}
double* imgcpy(double* image_g, int size)
{
    double* residual_g = image_g;
    // cudaMalloc((void**)&residual_g,sizeof(double)*size);
    // cudaMemcpy(residual_g, image_g, sizeof(double)*size, cudaMemcpyDeviceToDevice);
    return residual_g;
}
void add_layer(const int channel, const int shape,
                double* &input_g, double* &residual_g,
                cudnnHandle_t &handle)
{
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_DOUBLE,
                               1,channel,shape,shape));
    cudnnTensorDescriptor_t residual_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&residual_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(residual_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_DOUBLE,
                               1,channel,shape,shape));
    double alpha=1.0, beta_add=1.0;
        checkCUDNN(cudnnAddTensor(
			handle,
			&alpha,
			residual_descriptor,
			residual_g,
			&beta_add,
			input_descriptor,
			input_g));
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(residual_descriptor);
    //cudaFree(residual_g);
}

void global_avg_layer(double* &input_g, double* &output_g, cudnnHandle_t &handle)
{
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_DOUBLE,
                               1,1280,8,8));
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_DOUBLE,
                               1,1280,1,1));
    cudnnPoolingDescriptor_t pooling_descriptor;
    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    checkCUDNN(cudnnSetPooling2dDescriptor(pooling_descriptor,
                               CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                               CUDNN_PROPAGATE_NAN,
                               8,8,0,0,1,1));
    double alpha = 1.0, beta = 0.0;
    checkCUDNN(cudnnPoolingForward(
        handle,
        pooling_descriptor,
        &alpha,
        input_descriptor,
        input_g,
        &beta,
        output_descriptor,
        output_g));
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyPoolingDescriptor(pooling_descriptor);

    input_g = output_g;
}

// TODO: 实现自己的inference
void inference(double *input, double *output, cudnnHandle_t &handle)
{
    double * image_g = NULL;
    cudaMalloc((void**)&image_g, sizeof(double)*1*3*244*244);
    double * img_g_bak = image_g;
    cudaMemcpy(image_g, input, sizeof(double)*1*3*244*244,cudaMemcpyHostToDevice);
    conv_layer(3,244,32,122,3,1,2,1,Conv_0_W,Conv_0_B,image_g,Conv_0_out, handle);
    conv_layer(32,122,32,122,3,1,1,1,Conv_2_W,Conv_2_B,image_g,Conv_2_out,handle);
    conv_layer(32,122,16,122,1,0,1,1,Conv_4_W,Conv_4_B,image_g,Conv_4_out,handle,false);
    conv_layer(16,122,96,122,1,0,1,1,Conv_5_W,Conv_5_B,image_g,Conv_5_out,handle);
    conv_layer(96,122,96,61,3,1,2,1,Conv_7_W,Conv_7_B,image_g,Conv_7_out,handle);
    conv_layer(96,61,24,61,1,0,1,1,Conv_9_W,Conv_9_B,image_g,Conv_9_out,handle,false);
    double* residual_g = imgcpy(image_g, 1*24*61*61);
    conv_layer(24,61,144,61,1,0,1,1,Conv_10_W,Conv_10_B,image_g,Conv_10_out,handle);
    conv_layer(144,61,144,61,3,1,1,1,Conv_12_W,Conv_12_B,image_g,Conv_12_out,handle);
    conv_layer(144,61,24,61,1,0,1,1,Conv_14_W,Conv_14_B,image_g,Conv_14_out,handle,false);
    add_layer(24,61,image_g,residual_g,handle);
    conv_layer(24,61,144,61,1,0,1,1,Conv_16_W,Conv_16_B,image_g,Conv_16_out,handle);
    conv_layer(144,61,144,31,3,1,2,1,Conv_18_W,Conv_18_B,image_g,Conv_18_out,handle);
    conv_layer(144,31,32,31,1,0,1,1,Conv_20_W,Conv_20_B,image_g,Conv_20_out,handle,false);
    residual_g = imgcpy(image_g,32*31*31);
    conv_layer(32,31,192,31,1,0,1,1,Conv_21_W,Conv_21_B,image_g,Conv_21_out,handle);
    conv_layer(192,31,192,31,3,1,1,1,Conv_23_W,Conv_23_B,image_g,Conv_23_out,handle);
    conv_layer(192,31,32,31,1,0,1,1,Conv_25_W,Conv_25_B,image_g,Conv_25_out,handle,false);
    add_layer(32,31,image_g,residual_g,handle);
    residual_g = imgcpy(image_g,32*31*31);
    conv_layer(32,31,192,31,1,0,1,1,Conv_27_W,Conv_27_B,image_g,Conv_27_out,handle);
    conv_layer(192,31,192,31,3,1,1,1,Conv_29_W,Conv_29_B,image_g,Conv_29_out,handle);
    conv_layer(192,31,32,31,1,0,1,1,Conv_31_W,Conv_31_B,image_g,Conv_31_out,handle,false);
    add_layer(32,31,image_g,residual_g,handle);
    conv_layer(32,31,192,31,1,0,1,1,Conv_33_W,Conv_33_B,image_g,Conv_33_out,handle);
    conv_layer(192,31,192,16,3,1,2,1,Conv_35_W,Conv_35_B,image_g,Conv_35_out,handle);
    conv_layer(192,16,64,16,1,0,1,1,Conv_37_W,Conv_37_B,image_g,Conv_37_out,handle,false);
    residual_g = imgcpy(image_g,64*16*16);
    conv_layer(64,16,384,16,1,0,1,1,Conv_38_W,Conv_38_B,image_g,Conv_38_out,handle);
    conv_layer(384,16,384,16,3,1,1,1,Conv_40_W,Conv_40_B,image_g,Conv_40_out,handle);
    conv_layer(384,16,64,16,1,0,1,1,Conv_42_W,Conv_42_B,image_g,Conv_42_out,handle,false);
    add_layer(64,16,image_g,residual_g,handle);
    residual_g = imgcpy(image_g,64*16*16);
    conv_layer(64,16,384,16,1,0,1,1,Conv_44_W,Conv_44_B,image_g,Conv_44_out,handle);
    conv_layer(384,16,384,16,3,1,1,1,Conv_46_W,Conv_46_B,image_g,Conv_46_out,handle);
    conv_layer(384,16,64,16,1,0,1,1,Conv_48_W,Conv_48_B,image_g,Conv_48_out,handle,false);
    add_layer(64,16,image_g,residual_g,handle);
    residual_g = imgcpy(image_g,64*16*16);
    conv_layer(64,16,384,16,1,0,1,1,Conv_50_W,Conv_50_B,image_g,Conv_50_out,handle);
    conv_layer(384,16,384,16,3,1,1,1,Conv_52_W,Conv_52_B,image_g,Conv_52_out,handle);
    conv_layer(384,16,64,16,1,0,1,1,Conv_54_W,Conv_54_B,image_g,Conv_54_out,handle,false);
    add_layer(64,16,image_g,residual_g,handle);
    conv_layer(64,16,384,16,1,0,1,1,Conv_56_W,Conv_56_B,image_g,Conv_56_out,handle);
    conv_layer(384,16,384,16,3,1,1,1,Conv_58_W,Conv_58_B,image_g,Conv_58_out,handle);
    conv_layer(384,16,96,16,1,0,1,1,Conv_60_W,Conv_60_B,image_g,Conv_60_out,handle,false);
    residual_g = imgcpy(image_g,96*16*16);
    conv_layer(96,16,576,16,1,0,1,1,Conv_61_W,Conv_61_B,image_g,Conv_61_out,handle);
    conv_layer(576,16,576,16,3,1,1,1,Conv_63_W,Conv_63_B,image_g,Conv_63_out,handle);
    conv_layer(576,16,96,16,1,0,1,1,Conv_65_W,Conv_65_B,image_g,Conv_65_out,handle,false);
    add_layer(96,16,image_g,residual_g,handle);
    residual_g = imgcpy(image_g,96*16*16);
    conv_layer(96,16,576,16,1,0,1,1,Conv_67_W,Conv_67_B,image_g,Conv_67_out,handle);
    conv_layer(576,16,576,16,3,1,1,1,Conv_69_W,Conv_69_B,image_g,Conv_69_out,handle);
    conv_layer(576,16,96,16,1,0,1,1,Conv_71_W,Conv_71_B,image_g,Conv_71_out,handle,false);
    add_layer(96,16,image_g,residual_g,handle);
    conv_layer(96,16,576,16,1,0,1,1,Conv_73_W,Conv_73_B,image_g,Conv_73_out,handle);
    conv_layer(576,16,576,8,3,1,2,1,Conv_75_W,Conv_75_B,image_g,Conv_75_out,handle);
    conv_layer(576,8,160,8,1,0,1,1,Conv_77_W,Conv_77_B,image_g,Conv_77_out,handle,false);
    residual_g = imgcpy(image_g,160*8*8);
    conv_layer(160,8,960,8,1,0,1,1,Conv_78_W,Conv_78_B,image_g,Conv_78_out,handle);
    conv_layer(960,8,960,8,3,1,1,1,Conv_80_W,Conv_80_B,image_g,Conv_80_out,handle);
    conv_layer(960,8,160,8,1,0,1,1,Conv_82_W,Conv_82_B,image_g,Conv_82_out,handle,false);
    add_layer(160,8,image_g,residual_g,handle);
    residual_g = imgcpy(image_g,160*8*8);
    conv_layer(160,8,960,8,1,0,1,1,Conv_84_W,Conv_84_B,image_g,Conv_84_out,handle);
    conv_layer(960,8,960,8,3,1,1,1,Conv_86_W,Conv_86_B,image_g,Conv_86_out,handle);
    conv_layer(960,8,160,8,1,0,1,1,Conv_88_W,Conv_88_B,image_g,Conv_88_out,handle,false);
    add_layer(160,8,image_g,residual_g,handle);
    conv_layer(160,8,960,8,1,0,1,1,Conv_90_W,Conv_90_B,image_g,Conv_90_out,handle);
    conv_layer(960,8,960,8,3,1,1,1,Conv_92_W,Conv_92_B,image_g,Conv_92_out,handle);
    conv_layer(960,8,320,8,1,0,1,1,Conv_94_W,Conv_94_B,image_g,Conv_94_out,handle,false);
    conv_layer(320,8,1280,8,1,0,1,1,Conv_95_W,Conv_95_B,image_g,Conv_95_out,handle);
    global_avg_layer(image_g,Avgpool_out,handle);
    conv_layer(1280,1,1000,1,1,0,1,1,Gemm_W,Gemm_B,image_g,Gemm_out,handle,false);
    //conv_layer(image_g,handle);

    cudaMemcpy(output, image_g, sizeof(double)*1000, cudaMemcpyDeviceToHost);
    cudaFree(img_g_bak);
}


using namespace std;
int main()
{
    cudnnHandle_t handle;
    checkCUDNN(cudnnCreate(&handle));
    initModel(); // 读取网络权重
    readInput("./mobilenetInput.txt");   // 读取输入
    readOutput("./mobilenetOutput.txt"); // 读取标准输出
    double sumTime = 0;
    for (int i = 0; i < TESTNUM; i++)
    {
        double inferOut[1000];
        for (int j = 0; j < ITERNUM; j++)
        {
            float Onetime;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

            // 执行Inference
            inference(inputArr[i], inferOut, handle);
            
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&Onetime, start, stop);
            // 累加单次推理消耗时间
            sumTime += Onetime;
        }
        checkOutput(benchOutArr[i], inferOut);
    }
    printf("Average Time is: %lf\n", (sumTime / TESTNUM / ITERNUM));

}