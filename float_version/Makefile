NY: all clean

CUDA_LIB_DIR := /usr/local/cuda-10.2/lib64

LIBS += -lcudart
LIBS += -lcuda
CU_SRCS := $(wildcard conv_float.cu global_avg_float.cu add_float.cu gemm_float.cu)
CU_OBJS := $(CU_SRCS:%.cu=%.o)

C_SRCS := $(wildcard main_float.cc)
C_OBJS := $(C_SRCS:%.c=%.o)

EXECUTABLE := mobilenet

all : $(EXECUTABLE)

$(CU_OBJS):$(CU_SRCS)
	nvcc -c $^

$(C_OBJS):$(C_SRCS) 
	nvcc -c $^

$(EXECUTABLE):$(CU_OBJS) $(C_OBJS) 
	nvcc -o $@ $^ -L$(CUDA_LIB_DIR) $(LIBS)

clean:
	rm $(EXECUTABLE) $(CU_OBJS)

