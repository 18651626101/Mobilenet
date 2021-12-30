# Mobilenet
在本项目，我们使用cuda实现了经典的图卷积模型[MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf）版本。Mobilenet 采用depthwise separable convolution的方式代替传统的卷积，在维持一定准确性的同时，有效地提高了模型推理的速度。结合GPU运算和mobilenet一些性质，我们对计算过程，特别是卷积过程进行了优化和加速，并与cudnn实现的结果进行了对比，结果表明我们的实现速度要远快于cudnn的实现。

## 文件结构
```text
Mobilenet/
+-- README.md
+-- .gitignore
+-- *_version/(模型实现代码，final_version下为最终实现)
    +-- add_*.cu (矩阵相加)
    +-- add_*.cuh
    +-- conv_*.cu (卷积)
    +-- conv_*.cuh
    +-- gemm_*.cu (矩阵相乘)
    +-- gemm_*.cuh
    +-- global_avg_*.cu (平均池化)
    +-- global_avg_*.cuh
    +-- main_*.cc (主函数，包含模型的拓扑结构实现)
    +-- Makefile
    +-- mobilenetInput.txt (输入)
    +-- mobilenetOutput.txt (参考输出)
    +-- params.txt (模型参数)
+-- cudnn_baseline/ (cudnn实现的baseline)
    +-- main_cudnn.cc (主函数，包含模型的拓扑结构实现) 
    +-- run_cudnn.sh
    +-- mobilenetInput.txt (输入)
    +-- mobilenetOutput.txt (参考输出)
    +-- params.txt (模型参数) 
+-- old_conv/(一些旧版本的卷积实现)
+-- onnx_parse/
    +-- mobilenet_v2_inplace.onnx(最终实现的mobilenet结构)
    +-- mobilenet_v2_inplace.onnx
    +-- utils.py(解析onnx)
+-- 答辩.pptx
+-- 实验文档.pdf
```
## 代码运行
1.cuda实现版本执行
```shell
cd ${*_version} #进入目录
make all #编译项目
./mobilenet #执行
```
2.cudnn实现版本执行
```shell
cd cudnn_baseline
sh run_cudnn.sh
```
## 实验结果
1.实验环境
GV100GL [Tesla V100 PCIe 32GB] 16.04.1-Ubuntu

2.实验结果
cuda优化加速后的结果（具体优化方法见文档）
```shell
# 单次inference时间
Average Time is: 7.713534
# nvprof结果
==27280== NVPROF is profiling process 27280, command: ./mobilenet
Average Time is: 10.748224
==27280== Profiling application: ./mobilenet
==27280== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   66.97%  5.5811ms       838  6.6600us  1.1510us  2.1937ms  [CUDA memcpy HtoD]
                   25.29%  2.1077ms        34  61.991us  23.039us  150.24us  conv1Kernel(double*, double*, double*, double*, int, int, int, int, int, int, int, bool)
                    2.41%  200.67us         1  200.67us  200.67us  200.67us  matMulKernel(int, int, int, int, double*, double*, double*, double*)
                    2.37%  197.66us        17  11.627us  4.7680us  35.263us  convGroupKernel(double*, double*, double*, double*, int, int, int, int, int, int, int, bool)
                    1.73%  144.06us        53  2.7180us  1.6000us  14.687us  [CUDA memset]
                    0.79%  65.984us         1  65.984us  65.984us  65.984us  convKernel(double*, double*, double*, double*, int, int, int, int, int, int, int, bool)
                    0.34%  28.031us        10  2.8030us  1.8240us  6.6880us  AddKernel(int, double*, double*)
                    0.07%  6.0800us         1  6.0800us  6.0800us  6.0800us  GlobalAvgKernel(double*, double*, int, int)
                    0.03%  2.5920us         1  2.5920us  2.5920us  2.5920us  [CUDA memcpy DtoH]
      API calls:   93.41%  282.07ms       162  1.7412ms  4.8760us  274.39ms  cudaMalloc
                    5.10%  15.387ms       839  18.339us  6.2420us  2.3962ms  cudaMemcpy
                    0.63%  1.8953ms         1  1.8953ms  1.8953ms  1.8953ms  cudaFree
                    0.41%  1.2528ms         1  1.2528ms  1.2528ms  1.2528ms  cuDeviceTotalMem
                    0.18%  546.08us        53  10.303us  8.1230us  69.851us  cudaMemset
                    0.17%  507.26us        64  7.9250us  6.2300us  33.378us  cudaLaunchKernel
                    0.08%  226.56us        97  2.3350us     225ns  80.258us  cuDeviceGetAttribute
                    0.01%  20.301us         1  20.301us  20.301us  20.301us  cuDeviceGetName
                    0.01%  19.261us         2  9.6300us  1.7340us  17.527us  cudaEventCreate
                    0.01%  15.302us         2  7.6510us  4.6540us  10.648us  cudaEventRecord
                    0.00%  5.6530us         1  5.6530us  5.6530us  5.6530us  cudaEventSynchronize
                    0.00%  5.4170us         1  5.4170us  5.4170us  5.4170us  cudaDeviceSynchronize
                    0.00%  4.6630us         1  4.6630us  4.6630us  4.6630us  cuDeviceGetPCIBusId
                    0.00%  2.7790us         1  2.7790us  2.7790us  2.7790us  cudaEventElapsedTime
                    0.00%  2.1090us         3     703ns     322ns  1.3210us  cuDeviceGetCount
                    0.00%  1.1240us         2     562ns     291ns     833ns  cuDeviceGet
                    0.00%     589ns         1     589ns     589ns     589ns  cuDeviceGetUuid
```
