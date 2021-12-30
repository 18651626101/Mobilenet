#include "conv_v5.cuh"
#include "global_avg.cuh"
#include "add.cuh"
#include "gemm.cuh"

#define INPUTSHAPE 3 * 244 * 244
#define OUTPUTSHAPE 1000
#define TESTNUM 1
#define ITERNUM 1
#define DEBUG false
double inputArr[TESTNUM][INPUTSHAPE];
double benchOutArr[TESTNUM][OUTPUTSHAPE];

// double Conv_0_W[32 * 3 * 3 * 3];
// double Conv_0_B[32];
// double Conv_2_W[32 * 1 * 3 * 3];
// double Conv_2_B[32];
// double Conv_4_W[16 * 32 * 1 * 1];
// double Conv_4_B[16];
// double Conv_5_W[96 * 16 * 1 * 1];
// double Conv_5_B[96];
// double Conv_7_W[96 * 1 * 3 * 3];
// double Conv_7_B[96];
// double Conv_9_W[24 * 96 * 1 * 1];
// double Conv_9_B[24];
// double Conv_10_W[144 * 24 * 1 * 1];
// double Conv_10_B[144];
// double Conv_12_W[144 * 1 * 3 * 3];
// double Conv_12_B[144];
// double Conv_14_W[24 * 144 * 1 * 1];
// double Conv_14_B[24];
// double Conv_16_W[144 * 24 * 1 * 1];
// double Conv_16_B[144];
// double Conv_18_W[144 * 1 * 3 * 3];
// double Conv_18_B[144];
// double Conv_20_W[32 * 144 * 1 * 1];
// double Conv_20_B[32];
// double Conv_21_W[192 * 32 * 1 * 1];
// double Conv_21_B[192];
// double Conv_23_W[192 * 1 * 3 * 3];
// double Conv_23_B[192];
// double Conv_25_W[32 * 192 * 1 * 1];
// double Conv_25_B[32];
// double Conv_27_W[192 * 32 * 1 * 1];
// double Conv_27_B[192];
// double Conv_29_W[192 * 1 * 3 * 3];
// double Conv_29_B[192];
// double Conv_31_W[32 * 192 * 1 * 1];
// double Conv_31_B[32];
// double Conv_33_W[192 * 32 * 1 * 1];
// double Conv_33_B[192];
// double Conv_35_W[192 * 1 * 3 * 3];
// double Conv_35_B[192];
// double Conv_37_W[64 * 192 * 1 * 1];
// double Conv_37_B[64];
// double Conv_38_W[384 * 64 * 1 * 1];
// double Conv_38_B[384];
// double Conv_40_W[384 * 1 * 3 * 3];
// double Conv_40_B[384];
// double Conv_42_W[64 * 384 * 1 * 1];
// double Conv_42_B[64];
// double Conv_44_W[384 * 64 * 1 * 1];
// double Conv_44_B[384];
// double Conv_46_W[384 * 1 * 3 * 3];
// double Conv_46_B[384];
// double Conv_48_W[64 * 384 * 1 * 1];
// double Conv_48_B[64];
// double Conv_50_W[384 * 64 * 1 * 1];
// double Conv_50_B[384];
// double Conv_52_W[384 * 1 * 3 * 3];
// double Conv_52_B[384];
// double Conv_54_W[64 * 384 * 1 * 1];
// double Conv_54_B[64];
// double Conv_56_W[384 * 64 * 1 * 1];
// double Conv_56_B[384];
// double Conv_58_W[384 * 1 * 3 * 3];
// double Conv_58_B[384];
// double Conv_60_W[96 * 384 * 1 * 1];
// double Conv_60_B[96];
// double Conv_61_W[576 * 96 * 1 * 1];
// double Conv_61_B[576];
// double Conv_63_W[576 * 1 * 3 * 3];
// double Conv_63_B[576];
// double Conv_65_W[96 * 576 * 1 * 1];
// double Conv_65_B[96];
// double Conv_67_W[576 * 96 * 1 * 1];
// double Conv_67_B[576];
// double Conv_69_W[576 * 1 * 3 * 3];
// double Conv_69_B[576];
// double Conv_71_W[96 * 576 * 1 * 1];
// double Conv_71_B[96];
// double Conv_73_W[576 * 96 * 1 * 1];
// double Conv_73_B[576];
// double Conv_75_W[576 * 1 * 3 * 3];
// double Conv_75_B[576];
// double Conv_77_W[160 * 576 * 1 * 1];
// double Conv_77_B[160];
// double Conv_78_W[960 * 160 * 1 * 1];
// double Conv_78_B[960];
// double Conv_80_W[960 * 1 * 3 * 3];
// double Conv_80_B[960];
// double Conv_82_W[160 * 960 * 1 * 1];
// double Conv_82_B[160];
// double Conv_84_W[960 * 160 * 1 * 1];
// double Conv_84_B[960];
// double Conv_86_W[960 * 1 * 3 * 3];
// double Conv_86_B[960];
// double Conv_88_W[160 * 960 * 1 * 1];
// double Conv_88_B[160];
// double Conv_90_W[960 * 160 * 1 * 1];
// double Conv_90_B[960];
// double Conv_92_W[960 * 1 * 3 * 3];
// double Conv_92_B[960];
// double Conv_94_W[320 * 960 * 1 * 1];
// double Conv_94_B[320];
// double Conv_95_W[1280 * 320 * 1 * 1];
// double Conv_95_B[1280];
// double Gemm_W[1000 * 1280];
// double Gemm_B[1000];
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

double* imgcpy(const double* image_g, int size)
{
    double* residual_g = NULL;
    cudaMalloc((void**)&residual_g,sizeof(double)*size);
    cudaMemcpy(residual_g, image_g, sizeof(double)*size, cudaMemcpyDeviceToDevice);
    return residual_g;
}

// TODO: 读取权重
// void initModel()
// {
//     FILE *fp = NULL;
//     fp = fopen("./params.txt", "r");
//     for (int i = 0; i < 32 * 3 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_0_W[i]);
//     for (int i = 0; i < 32; i++)
//         fscanf(fp, "%lf", &Conv_0_B[i]);
//     for (int i = 0; i < 32 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_2_W[i]);
//     for (int i = 0; i < 32; i++)
//         fscanf(fp, "%lf", &Conv_2_B[i]);
//     for (int i = 0; i < 16 * 32 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_4_W[i]);
//     for (int i = 0; i < 16; i++)
//         fscanf(fp, "%lf", &Conv_4_B[i]);
//     for (int i = 0; i < 96 * 16 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_5_W[i]);
//     for (int i = 0; i < 96; i++)
//         fscanf(fp, "%lf", &Conv_5_B[i]);
//     for (int i = 0; i < 96 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_7_W[i]);
//     for (int i = 0; i < 96; i++)
//         fscanf(fp, "%lf", &Conv_7_B[i]);
//     for (int i = 0; i < 24 * 96 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_9_W[i]);
//     for (int i = 0; i < 24; i++)
//         fscanf(fp, "%lf", &Conv_9_B[i]);
//     for (int i = 0; i < 144 * 24 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_10_W[i]);
//     for (int i = 0; i < 144; i++)
//         fscanf(fp, "%lf", &Conv_10_B[i]);
//     for (int i = 0; i < 144 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_12_W[i]);
//     for (int i = 0; i < 144; i++)
//         fscanf(fp, "%lf", &Conv_12_B[i]);
//     for (int i = 0; i < 24 * 144 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_14_W[i]);
//     for (int i = 0; i < 24; i++)
//         fscanf(fp, "%lf", &Conv_14_B[i]);
//     for (int i = 0; i < 144 * 24 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_16_W[i]);
//     for (int i = 0; i < 144; i++)
//         fscanf(fp, "%lf", &Conv_16_B[i]);
//     for (int i = 0; i < 144 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_18_W[i]);
//     for (int i = 0; i < 144; i++)
//         fscanf(fp, "%lf", &Conv_18_B[i]);
//     for (int i = 0; i < 32 * 144 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_20_W[i]);
//     for (int i = 0; i < 32; i++)
//         fscanf(fp, "%lf", &Conv_20_B[i]);
//     for (int i = 0; i < 192 * 32 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_21_W[i]);
//     for (int i = 0; i < 192; i++)
//         fscanf(fp, "%lf", &Conv_21_B[i]);
//     for (int i = 0; i < 192 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_23_W[i]);
//     for (int i = 0; i < 192; i++)
//         fscanf(fp, "%lf", &Conv_23_B[i]);
//     for (int i = 0; i < 32 * 192 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_25_W[i]);
//     for (int i = 0; i < 32; i++)
//         fscanf(fp, "%lf", &Conv_25_B[i]);
//     for (int i = 0; i < 192 * 32 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_27_W[i]);
//     for (int i = 0; i < 192; i++)
//         fscanf(fp, "%lf", &Conv_27_B[i]);
//     for (int i = 0; i < 192 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_29_W[i]);
//     for (int i = 0; i < 192; i++)
//         fscanf(fp, "%lf", &Conv_29_B[i]);
//     for (int i = 0; i < 32 * 192 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_31_W[i]);
//     for (int i = 0; i < 32; i++)
//         fscanf(fp, "%lf", &Conv_31_B[i]);
//     for (int i = 0; i < 192 * 32 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_33_W[i]);
//     for (int i = 0; i < 192; i++)
//         fscanf(fp, "%lf", &Conv_33_B[i]);
//     for (int i = 0; i < 192 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_35_W[i]);
//     for (int i = 0; i < 192; i++)
//         fscanf(fp, "%lf", &Conv_35_B[i]);
//     for (int i = 0; i < 64 * 192 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_37_W[i]);
//     for (int i = 0; i < 64; i++)
//         fscanf(fp, "%lf", &Conv_37_B[i]);
//     for (int i = 0; i < 384 * 64 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_38_W[i]);
//     for (int i = 0; i < 384; i++)
//         fscanf(fp, "%lf", &Conv_38_B[i]);
//     for (int i = 0; i < 384 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_40_W[i]);
//     for (int i = 0; i < 384; i++)
//         fscanf(fp, "%lf", &Conv_40_B[i]);
//     for (int i = 0; i < 64 * 384 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_42_W[i]);
//     for (int i = 0; i < 64; i++)
//         fscanf(fp, "%lf", &Conv_42_B[i]);
//     for (int i = 0; i < 384 * 64 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_44_W[i]);
//     for (int i = 0; i < 384; i++)
//         fscanf(fp, "%lf", &Conv_44_B[i]);
//     for (int i = 0; i < 384 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_46_W[i]);
//     for (int i = 0; i < 384; i++)
//         fscanf(fp, "%lf", &Conv_46_B[i]);
//     for (int i = 0; i < 64 * 384 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_48_W[i]);
//     for (int i = 0; i < 64; i++)
//         fscanf(fp, "%lf", &Conv_48_B[i]);
//     for (int i = 0; i < 384 * 64 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_50_W[i]);
//     for (int i = 0; i < 384; i++)
//         fscanf(fp, "%lf", &Conv_50_B[i]);
//     for (int i = 0; i < 384 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_52_W[i]);
//     for (int i = 0; i < 384; i++)
//         fscanf(fp, "%lf", &Conv_52_B[i]);
//     for (int i = 0; i < 64 * 384 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_54_W[i]);
//     for (int i = 0; i < 64; i++)
//         fscanf(fp, "%lf", &Conv_54_B[i]);
//     for (int i = 0; i < 384 * 64 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_56_W[i]);
//     for (int i = 0; i < 384; i++)
//         fscanf(fp, "%lf", &Conv_56_B[i]);
//     for (int i = 0; i < 384 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_58_W[i]);
//     for (int i = 0; i < 384; i++)
//         fscanf(fp, "%lf", &Conv_58_B[i]);
//     for (int i = 0; i < 96 * 384 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_60_W[i]);
//     for (int i = 0; i < 96; i++)
//         fscanf(fp, "%lf", &Conv_60_B[i]);
//     for (int i = 0; i < 576 * 96 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_61_W[i]);
//     for (int i = 0; i < 576; i++)
//         fscanf(fp, "%lf", &Conv_61_B[i]);
//     for (int i = 0; i < 576 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_63_W[i]);
//     for (int i = 0; i < 576; i++)
//         fscanf(fp, "%lf", &Conv_63_B[i]);
//     for (int i = 0; i < 96 * 576 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_65_W[i]);
//     for (int i = 0; i < 96; i++)
//         fscanf(fp, "%lf", &Conv_65_B[i]);
//     for (int i = 0; i < 576 * 96 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_67_W[i]);
//     for (int i = 0; i < 576; i++)
//         fscanf(fp, "%lf", &Conv_67_B[i]);
//     for (int i = 0; i < 576 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_69_W[i]);
//     for (int i = 0; i < 576; i++)
//         fscanf(fp, "%lf", &Conv_69_B[i]);
//     for (int i = 0; i < 96 * 576 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_71_W[i]);
//     for (int i = 0; i < 96; i++)
//         fscanf(fp, "%lf", &Conv_71_B[i]);
//     for (int i = 0; i < 576 * 96 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_73_W[i]);
//     for (int i = 0; i < 576; i++)
//         fscanf(fp, "%lf", &Conv_73_B[i]);
//     for (int i = 0; i < 576 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_75_W[i]);
//     for (int i = 0; i < 576; i++)
//         fscanf(fp, "%lf", &Conv_75_B[i]);
//     for (int i = 0; i < 160 * 576 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_77_W[i]);
//     for (int i = 0; i < 160; i++)
//         fscanf(fp, "%lf", &Conv_77_B[i]);
//     for (int i = 0; i < 960 * 160 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_78_W[i]);
//     for (int i = 0; i < 960; i++)
//         fscanf(fp, "%lf", &Conv_78_B[i]);
//     for (int i = 0; i < 960 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_80_W[i]);
//     for (int i = 0; i < 960; i++)
//         fscanf(fp, "%lf", &Conv_80_B[i]);
//     for (int i = 0; i < 160 * 960 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_82_W[i]);
//     for (int i = 0; i < 160; i++)
//         fscanf(fp, "%lf", &Conv_82_B[i]);
//     for (int i = 0; i < 960 * 160 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_84_W[i]);
//     for (int i = 0; i < 960; i++)
//         fscanf(fp, "%lf", &Conv_84_B[i]);
//     for (int i = 0; i < 960 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_86_W[i]);
//     for (int i = 0; i < 960; i++)
//         fscanf(fp, "%lf", &Conv_86_B[i]);
//     for (int i = 0; i < 160 * 960 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_88_W[i]);
//     for (int i = 0; i < 160; i++)
//         fscanf(fp, "%lf", &Conv_88_B[i]);
//     for (int i = 0; i < 960 * 160 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_90_W[i]);
//     for (int i = 0; i < 960; i++)
//         fscanf(fp, "%lf", &Conv_90_B[i]);
//     for (int i = 0; i < 960 * 1 * 3 * 3; i++)
//         fscanf(fp, "%lf", &Conv_92_W[i]);
//     for (int i = 0; i < 960; i++)
//         fscanf(fp, "%lf", &Conv_92_B[i]);
//     for (int i = 0; i < 320 * 960 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_94_W[i]);
//     for (int i = 0; i < 320; i++)
//         fscanf(fp, "%lf", &Conv_94_B[i]);
//     for (int i = 0; i < 1280 * 320 * 1 * 1; i++)
//         fscanf(fp, "%lf", &Conv_95_W[i]);
//     for (int i = 0; i < 1280; i++)
//         fscanf(fp, "%lf", &Conv_95_B[i]);
//     double _;
//     fscanf(fp, "%lf %lf", &_, &_); // const 1 -1
//     for (int i = 0; i < 1000 * 1280; i++)
//         fscanf(fp, "%lf", &Gemm_W[i]);
//     for (int i = 0; i < 1000; i++)
//         fscanf(fp, "%lf", &Gemm_B[i]);
// }
void initModel()
{
    FILE *fp = NULL;
    fp = fopen("./params.txt", "r");
    temp_h = new double[32*3*3*3];
    for(int i=0;i<32*3*3*3;i++)fscanf(fp,"%lf",&temp_h[i]);
    cudaMalloc((void**)&temp_d, sizeof(double)*32*3*3*3);
    cudaMemcpy(temp_d, temp_h, sizeof(double)*32*3*3*3, cudaMemcpyHostToDevice);
    Conv_0_W = temp_d;
    delete [] temp_h;
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
}

// TODO: 实现自己的inference
void inference(double *input, double *output)
{
    if(DEBUG) printf("========== main::begin inference ==========\n");
    double * image_g = NULL;
    double test_output[1280*64];
    // cudaMalloc((void**)&image_g, sizeof(double)*1*3*244*244);
    // cudaMemcpy(image_g, input, sizeof(double)*1*3*244*244, cudaMemcpyHostToDevice);
    image_g = pad(3,244,1,input);
    image_g = conv(3,244,32,122,3,1,2,1,Conv_0_W, Conv_0_B,image_g,1);
    image_g = conv_group(32,122,32,122,3,1,1,1,Conv_2_W,Conv_2_B,image_g,0);
    image_g = conv(32,122,16,122,1,0,1,1,Conv_4_W,Conv_4_B,image_g,0,false);
    image_g = conv(16,122,96,122,1,0,1,1,Conv_5_W,Conv_5_B,image_g,1);
    image_g = conv_group(96,122,96,61,3,1,2,1,Conv_7_W,Conv_7_B,image_g,0);
    image_g = conv(96,61,24,61,1,0,1,1,Conv_9_W,Conv_9_B,image_g,0,false);
    double* residual_g = imgcpy(image_g, 1*24*61*61);
    image_g = conv(24,61,144,61,1,0,1,1,Conv_10_W,Conv_10_B,image_g,1);
    image_g = conv_group(144,61,144,61,3,1,1,1,Conv_12_W,Conv_12_B,image_g,0);
    image_g = conv(144,61,24,61,1,0,1,1,Conv_14_W,Conv_14_B,image_g,0,false);
    add(24,61,image_g,residual_g);
    image_g = conv(24,61,144,61,1,0,1,1,Conv_16_W,Conv_16_B,image_g,1);
    image_g = conv_group(144,61,144,31,3,1,2,1,Conv_18_W,Conv_18_B,image_g,0);
    image_g = conv(144,31,32,31,1,0,1,1,Conv_20_W,Conv_20_B,image_g,0,false);
    residual_g = imgcpy(image_g,32*31*31);
    image_g = conv(32,31,192,31,1,0,1,1,Conv_21_W,Conv_21_B,image_g,1);
    image_g = conv_group(192,31,192,31,3,1,1,1,Conv_23_W,Conv_23_B,image_g,0);
    image_g = conv(192,31,32,31,1,0,1,1,Conv_25_W,Conv_25_B,image_g,0,false);
    add(32,31,image_g,residual_g);
    residual_g = imgcpy(image_g,32*31*31);
    image_g = conv(32,31,192,31,1,0,1,1,Conv_27_W,Conv_27_B,image_g,1);
    image_g = conv_group(192,31,192,31,3,1,1,1,Conv_29_W,Conv_29_B,image_g,0);
    image_g = conv(192,31,32,31,1,0,1,1,Conv_31_W,Conv_31_B,image_g,0,false);
    add(32,31,image_g,residual_g);
    image_g = conv(32,31,192,31,1,0,1,1,Conv_33_W,Conv_33_B,image_g,1);
    image_g = conv_group(192,31,192,16,3,1,2,1,Conv_35_W,Conv_35_B,image_g,0);
    image_g = conv(192,16,64,16,1,0,1,1,Conv_37_W,Conv_37_B,image_g,0,false);
    residual_g = imgcpy(image_g,64*16*16);
    image_g = conv(64,16,384,16,1,0,1,1,Conv_38_W,Conv_38_B,image_g,1);
    image_g = conv_group(384,16,384,16,3,1,1,1,Conv_40_W,Conv_40_B,image_g,0);
    image_g = conv(384,16,64,16,1,0,1,1,Conv_42_W,Conv_42_B,image_g,0,false);
    add(64,16,image_g,residual_g);
    residual_g = imgcpy(image_g,64*16*16);
    image_g = conv(64,16,384,16,1,0,1,1,Conv_44_W,Conv_44_B,image_g,1);
    image_g = conv_group(384,16,384,16,3,1,1,1,Conv_46_W,Conv_46_B,image_g,0);
    image_g = conv(384,16,64,16,1,0,1,1,Conv_48_W,Conv_48_B,image_g,0,false);
    add(64,16,image_g,residual_g);
    residual_g = imgcpy(image_g,64*16*16);
    image_g = conv(64,16,384,16,1,0,1,1,Conv_50_W,Conv_50_B,image_g,1);
    image_g = conv_group(384,16,384,16,3,1,1,1,Conv_52_W,Conv_52_B,image_g,0);
    image_g = conv(384,16,64,16,1,0,1,1,Conv_54_W,Conv_54_B,image_g,0,false);
    add(64,16,image_g,residual_g);
    image_g = conv(64,16,384,16,1,0,1,1,Conv_56_W,Conv_56_B,image_g,1);
    image_g = conv_group(384,16,384,16,3,1,1,1,Conv_58_W,Conv_58_B,image_g,0);
    image_g = conv(384,16,96,16,1,0,1,1,Conv_60_W,Conv_60_B,image_g,0,false);
    residual_g = imgcpy(image_g,96*16*16);
    image_g = conv(96,16,576,16,1,0,1,1,Conv_61_W,Conv_61_B,image_g,1);
    image_g = conv_group(576,16,576,16,3,1,1,1,Conv_63_W,Conv_63_B,image_g,0);
    image_g = conv(576,16,96,16,1,0,1,1,Conv_65_W,Conv_65_B,image_g,0,false);
    add(96,16,image_g,residual_g);
    residual_g = imgcpy(image_g,96*16*16);
    image_g = conv(96,16,576,16,1,0,1,1,Conv_67_W,Conv_67_B,image_g,1);
    image_g = conv_group(576,16,576,16,3,1,1,1,Conv_69_W,Conv_69_B,image_g,0);
    image_g = conv(576,16,96,16,1,0,1,1,Conv_71_W,Conv_71_B,image_g,0,false);
    add(96,16,image_g,residual_g);
    image_g = conv(96,16,576,16,1,0,1,1,Conv_73_W,Conv_73_B,image_g,1);
    image_g = conv_group(576,16,576,8,3,1,2,1,Conv_75_W,Conv_75_B,image_g,0);
    image_g = conv(576,8,160,8,1,0,1,1,Conv_77_W,Conv_77_B,image_g,0,false);
    residual_g = imgcpy(image_g,160*8*8);
    image_g = conv(160,8,960,8,1,0,1,1,Conv_78_W,Conv_78_B,image_g,1);
    image_g = conv_group(960,8,960,8,3,1,1,1,Conv_80_W,Conv_80_B,image_g,0);
    image_g = conv(960,8,160,8,1,0,1,1,Conv_82_W,Conv_82_B,image_g,0,false);
    add(160,8,image_g,residual_g);
    residual_g = imgcpy(image_g,160*8*8);
    image_g = conv(160,8,960,8,1,0,1,1,Conv_84_W,Conv_84_B,image_g,1);
    image_g = conv_group(960,8,960,8,3,1,1,1,Conv_86_W,Conv_86_B,image_g,0);
    image_g = conv(960,8,160,8,1,0,1,1,Conv_88_W,Conv_88_B,image_g,0,false);
    add(160,8,image_g,residual_g);
    image_g = conv(160,8,960,8,1,0,1,1,Conv_90_W,Conv_90_B,image_g,1);
    image_g = conv_group(960,8,960,8,3,1,1,1,Conv_92_W,Conv_92_B,image_g,0);
    image_g = conv(960,8,320,8,1,0,1,1,Conv_94_W,Conv_94_B,image_g,0,false);
    image_g = conv(320,8,1280,8,1,0,1,1,Conv_95_W,Conv_95_B,image_g,0);
    image_g = global_avg(image_g, 1280, 8);
    image_g = conv(1280,1,1000,1,1,0,1,1,Gemm_W,Gemm_B,image_g,0,false);


    cudaMemcpy(output, image_g, sizeof(double)*1000, cudaMemcpyDeviceToHost);
    cudaFree(image_g);
    // cudaMemcpy(test_output, image_g, sizeof(double)*1*1280*8*8, cudaMemcpyDeviceToHost);
    // for(int i=0;i<1280*8*8;i++){
    //     printf("%lf\n", test_output[i]);
    // }
    if(DEBUG) printf("========== main::end inference ==========\n");
}

int main()
{

    initModel(); // 读取网络权重

    readInput("./mobilenetInput.txt");   // 读取输入
    readOutput("./mobilenetOutput.txt"); // 读取标准输出
    float sumTime = 0;
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
            inference(inputArr[i], inferOut);

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