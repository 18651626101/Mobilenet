# Mobilenet
Implemented by cuda

## 文件结构
```text
mobilenet/
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
cuda实现版本
```shell
cd ${*_version} #进入目录
make all #编译项目
./mobilenet #执行
```