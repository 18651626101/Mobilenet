# Mobilenet
在本项目，我们使用cuda实现了经典的图卷积模型[MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf)版本。Mobilenet 采用depthwise separable convolution的方式代替传统的卷积，在维持一定准确性的同时，有效地提高了模型推理的速度。结合GPU运算和mobilenet一些性质，我们对计算过程，特别是卷积过程进行了优化和加速，并与cudnn实现的结果进行了对比，结果表明我们的实现速度要远快于cudnn的实现。

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
+-- 文档.pdf
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

3.cudnn baseline
cudnn的实现中我们仅做了简单优化，运行速度相对较慢。后续更多可能的优化方法由于时间所限没有继续实验。详细分析见文档和报告。
```shell
# 单次inference时间
Average Time is: 106.805535
# nvprof结果
==30019== NVPROF is profiling process 30019, command: ./cudnn.out
Average Time is: 116.022863
==30019== Profiling application: ./cudnn.out
==30019== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   41.30%  4.76116s     71497  66.592us  11.328us  343.07us  void implicit_convolve_dgemm<int=128, int=5, int=7, int=3, int=3, int=5, int=1, bool=0, bool=1, bool=1>(int, int, int, double const *, int, double*, double const *, kernel_conv_params, __int64, int, double, double, int, double const *, double const *, bool, int, int)
                   14.47%  1.66810s     22000  75.822us  9.7910us  416.12us  volta_zgemm_32x32_nt
                   11.47%  1.32161s     18003  73.410us  14.816us  338.68us  void explicit_convolve_dgemm<int, int=128, int=6, int=7, int=3, int=3, int=5, int=0, bool=0>(int, int, int, double const *, int, double const *, int, double*, kernel_conv_params, __int64, int, __int64, int, double, double, int, double const *, double const *)
                   11.00%  1.26757s     17500  72.432us  9.4070us  840.60us  void fft1d_r2c_32<double, double, double2, bool=1, bool=0>(double2*, double const *, int, int3, int3, int2, int2)
                    8.40%  968.17ms     18003  53.778us  10.272us  146.27us  void cudnn::cnn::im2col4d_kernel<double, long>(cudnn::cnn::im2col4d_params, cudnnConvolutionStruct, cudnnTensorStruct, double const *, cudnnTensorStruct*)
                    4.16%  479.37ms     22500  21.305us  9.9830us  47.071us  void fft1d_r2c_32<double, double, double2, bool=0, bool=0>(double2*, double const *, int, int3, int3, int2, int2)
                    3.15%  362.70ms     42500  8.5340us  3.1350us  32.671us  void cudnn::cnn::conv2d_grouped_direct_kernel<bool=0, bool=1, bool=0, bool=0, int=1, int=3, int, double, double, double, double, double, double>(cudnn::cnn::GroupedDirectFpropParams, double const *, double const *, double*, double, double*, double const * const *, double const *, cudnnActivationStruct)
                    1.85%  213.30ms       500  426.59us  422.43us  430.56us  void gemv2N_kernel<int, int, double2, double2, double2, int=128, int=16, int=4, int=4, int=1, cublasGemvParams<cublasGemvTensorStridedBatched<double2 const >, cublasGemvTensorStridedBatched<double2>, double2>>(double2 const )
                    1.45%  167.40ms     22500  7.4390us  3.4550us  24.192us  void fft1d_c2r_32<double2, double, double, bool=0, bool=1, bool=0, bool=0>(double*, double2 const *, int, int3, int3, int2, int, double, double, double*, double*)
                    1.02%  117.33ms     26000  4.5120us  2.1760us  37.471us  void op_generic_tensor_kernel<int=3, double, double, double, int=256, cudnnGenericOp_t=0, cudnnNanPropagation_t=0, int=0>(cudnnTensorStruct, double*, cudnnTensorStruct, double const *, cudnnTensorStruct, double const *, double, double, double, cudnnActivationStruct, reducedDivisorArray, int)
                    0.64%  73.901ms     17500  4.2220us  1.8560us  37.183us  void op_generic_tensor_kernel<int=1, double, double, double, int=256, cudnnGenericOp_t=9, cudnnNanPropagation_t=1, int=1>(cudnnTensorStruct, double*, cudnnTensorStruct, double const *, cudnnTensorStruct, double const *, double, double, double, cudnnActivationStruct, reducedDivisorArray, int)
                    0.60%  69.601ms       608  114.47us  1.1200us  3.2834ms  [CUDA memcpy HtoD]
                    0.28%  32.639ms       500  65.278us  55.423us  71.583us  void precomputed_convolve_dgemm<int=128, int=5, int=7, int=3, int=3, int=5, int=1, bool=0>(int, int, int, double const *, int, double*, double const *, kernel_conv_params, __int64, int, double, double, int, double const *, double const *, int*)
                    0.15%  17.492ms      5500  3.1800us  2.5920us  5.6320us  void op_generic_tensor_kernel<int=1, double, double, double, int=256, cudnnGenericOp_t=0, cudnnNanPropagation_t=0, int=0>(cudnnTensorStruct, double*, cudnnTensorStruct, double const *, cudnnTensorStruct, double const *, double, double, double, cudnnActivationStruct, reducedDivisorArray, int)
                    0.04%  4.5934ms       500  9.1860us  8.6720us  10.592us  void cudnn::ops::pooling_fw_4d_kernel<double, double, cudnn::averpooling_func<double, bool=1>, cudnnPoolingMode_t=2, bool=0>(cudnnTensorStruct, double const *, cudnn::ops::pooling_fw_4d_kernel<double, double, cudnn::averpooling_func<double, bool=1>, cudnnPoolingMode_t=2, bool=0>, cudnnTensorStruct*, cudnnPoolingStruct, double, cudnnPoolingStruct, int, cudnn::reduced_divisor, double)
                    0.01%  1.2106ms       500  2.4210us  2.2710us  3.4560us  [CUDA memcpy DtoH]
                    0.01%  953.75us       500  1.9070us  1.6960us  2.4640us  void cudnn::cnn::kern_precompute_indices<bool=0>(int*, int, int, int, int, int, int)
                    0.00%  8.4160us         4  2.1040us  1.7280us  3.2320us  [CUDA memset]
      API calls:   35.52%  17.3520s     89669  193.51us     533ns  1.15551s  cudaMalloc
                   28.62%  13.9803s    124000  112.74us  3.0120us  34.958ms  cudaEventSynchronize
                   16.59%  8.10530s     89505  90.556us     445ns  36.382ms  cudaFree
                    8.20%  4.00720s    285503  14.035us  6.0820us  9.9889ms  cudaLaunchKernel
                    3.43%  1.67541s    423000  3.9600us     645ns  2.2198ms  cudaEventRecord
                    2.86%  1.39904s         8  174.88ms  2.4810us  1.39902s  cudaStreamCreateWithFlags
                    1.22%  597.57ms       152  3.9314ms  6.2880us  82.128ms  cuModuleUnload
                    1.10%  537.84ms    124000  4.3370us  1.7300us  1.7278ms  cudaEventElapsedTime
                    0.79%  385.79ms    302500  1.2750us     653ns  1.5799ms  cudaStreamWaitEvent
                    0.68%  330.12ms      1108  297.94us  6.7870us  4.3752ms  cudaMemcpy
                    0.30%  148.79ms    400503     371ns     175ns  989.99us  cudaGetLastError
                    0.27%  133.69ms     53030  2.5200us     583ns  1.0629ms  cudaEventCreateWithFlags
                    0.24%  117.41ms     70500  1.6650us     644ns  1.0695ms  cudaEventDestroy
                    0.14%  68.187ms     18500  3.6850us     734ns  2.7415ms  cudaEventCreate
                    0.01%  3.0205ms         3  1.0068ms  963.88us  1.0901ms  cuDeviceTotalMem
                    0.01%  2.6389ms       500  5.2770us  3.9580us  10.860us  cudaDeviceSynchronize
                    0.00%  1.4679ms         1  1.4679ms  1.4679ms  1.4679ms  cudaHostAlloc
                    0.00%  658.82us       285  2.3110us     195ns  81.682us  cuDeviceGetAttribute
                    0.00%  260.38us         4  65.095us  2.3280us  250.79us  cudaStreamCreateWithPriority
                    0.00%  187.40us         1  187.40us  187.40us  187.40us  cudaGetDeviceProperties
                    0.00%  154.35us       150  1.0290us     500ns  4.6540us  cudaFuncSetAttribute
                    0.00%  113.31us         4  28.326us  15.237us  64.183us  cudaMemsetAsync
                    0.00%  103.02us         3  34.341us  20.009us  57.201us  cuDeviceGetName
                    0.00%  21.301us        40     532ns     411ns  1.3940us  cudaDeviceGetAttribute
                    0.00%  6.0900us         3  2.0300us     722ns  2.9230us  cudaGetDevice
                    0.00%  3.6250us         1  3.6250us  3.6250us  3.6250us  cuDeviceGetPCIBusId
                    0.00%  3.3550us         5     671ns     340ns  1.5550us  cuDeviceGetCount
                    0.00%  3.1590us         2  1.5790us  1.1640us  1.9950us  cuInit
                    0.00%  2.2720us         1  2.2720us  2.2720us  2.2720us  cudaHostGetDevicePointer
                    0.00%  2.0870us         4     521ns     292ns     846ns  cuDeviceGet
                    0.00%  1.9710us         2     985ns     518ns  1.4530us  cuDriverGetVersion
                    0.00%  1.6020us         1  1.6020us  1.6020us  1.6020us  cudaDeviceGetStreamPriorityRange
                    0.00%  1.3300us         3     443ns     368ns     577ns  cuDeviceGetUuid
                    0.00%  1.1840us         1  1.1840us  1.1840us  1.1840us  cuDevicePrimaryCtxRelease
                    0.00%  1.0490us         1  1.0490us  1.0490us  1.0490us  cudaGetDeviceCount
                    0.00%     461ns         1     461ns     461ns     461ns  cudaDriverGetVersion
```
