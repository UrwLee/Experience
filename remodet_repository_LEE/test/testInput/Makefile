#可执行文件
PROJECT=testInput
#-------------------------------------------------------------------------------
# 配置
# 使用Cafe
# CAFFE_DIR= $(HOME)/workspace/trunk/caffe
CAFFE_DIR= $(HOME)/work/repository
# 使用gtest
GTEST_DIR= /usr/include
GTEST_LIB= /usr/lib/x86_64-linux-gnu
# 使用CUDNN加速
USE_CUDNN=1
USE_OPENCV=1
USE_LEVELDB=0
USE_LMDB=1
CUDA_DIR=/usr/local/cuda
CUDA_ARCH=-gencode arch=compute_20,code=sm_20 \
					-gencode arch=compute_20,code=sm_21 \
					-gencode arch=compute_30,code=sm_30 \
					-gencode arch=compute_35,code=sm_35 \
					-gencode arch=compute_50,code=sm_50 \
					-gencode arch=compute_50,code=compute_50 \
					-gencode arch=compute_52,code=sm_52 \
					-gencode arch=compute_52,code=compute_52 \
					-gencode arch=compute_60,code=sm_60 \
					-gencode arch=compute_60,code=compute_60
DEBUG=0
#-------------------------------------------------------------------------------
# 输出路径
RELEASE_BUILD_DIR= release
DEBUG_BUILD_DIR= debug
ifeq ($(DEBUG), 1)
BUILD_DIR= $(DEBUG_BUILD_DIR)
OTHER_BUILD_DIR= $(RELEASE_BUILD_DIR)
else
BUILD_DIR= $(RELEASE_BUILD_DIR)
OTHER_BUILD_DIR= $(DEBUG_BUILD_DIR)
endif
# 获取源文件列表
CXX_SRCS= $(shell find src/ -name "*.cpp")
CU_SRCS= $(shell find src/ -name "*.cu")
CXX_SRC_NAMES= $(notdir $(CXX_SRCS))
CU_SRC_NAMES= $(notdir $(CU_SRCS))
# 获取目标文件列表
CXX_OBJS= $(addprefix $(BUILD_DIR)/, ${CXX_SRC_NAMES:.cpp=.o})
CU_OBJS= $(addprefix $(BUILD_DIR)/cuda/, ${CU_SRC_NAMES:.cu=.o})
OBJS= $(CXX_OBJS) $(CU_OBJS)
#-------------------------------------------------------------------------------
# 编译器
CXX=g++
NVCC=nvcc
#-------------------------------------------------------------------------------
# 获取包含和libs
# project headers and source files
# INCLUDE_DIRS+= ./include
# CAFFE: include
CAFFE_INCLUDE_DIR=$(CAFFE_DIR)/include
CAFFE_LIB_DIR=$(CAFFE_DIR)/build/lib
COMMON+= -I$(CAFFE_INCLUDE_DIR)
LDFLAGS+= -L$(CAFFE_LIB_DIR) -lcaffe
# CUDA
CUDA_INCLUDE_DIR=$(CUDA_DIR)/include
CUDA_LIB_DIR=$(CUDA_DIR)/lib64
COMMON+= -I$(CUDA_INCLUDE_DIR)
LDFLAGS+= -L$(CUDA_LIB_DIR) -lcudart -lcublas -lcurand
ifeq ($(USE_CUDNN), 1)
COMMON+= -DUSE_CUDNN
LDFLAGS+= -lcudnn
endif
# gtest
GTEST_INCLUDE_DIR=$(GTEST_DIR)
GTEST_LIB_DIR=$(GTEST_LIB)
COMMON+= -I$(GTEST_INCLUDE_DIR)
LDFLAGS+= -L$(GTEST_LIB_DIR) -lgtest -lgtest_main
# Use lmdb or leveldb
ifeq ($(USE_LEVELDB), 1)
COMMON+= -DUSE_LEVELDB
LDFLAGS+= -lleveldb -lsnappy
endif
ifeq ($(USE_LMDB), 1)
COMMON+= -DUSE_LMDB
LDFLAGS+= -llmdb
endif
ifeq ($(USE_OPENCV), 1)
COMMON+= -DUSE_OPENCV `pkg-config --cflags opencv`
LDFLAGS+= `pkg-config --libs opencv`
endif
# Other libs
LDFLAGS+= -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem
LDFLAGS+= -lboost_regex -lm -lhdf5_hl -lhdf5
LDFLAGS+= -lboost_thread -lstdc++ -pthread -lopenblas
#-------------------------------------------------------------------------------
# CXXFLAGS
WARNINGS= -Wall -Wno-sign-compare -Wfatal-errors
ifeq ($(DEBUG), 1)
COMMON+= -DDEBUG
CXXFLAGS= -g -O0
else
COMMON+= -DNDEBUG
CXXFLAGS= -Ofast
endif
CXXFLAGS+= $(WARNINGS)
#-------------------------------------------------------------------------------
VPATH=./src/
# 定义目标
all: dirs $(PROJECT)
# link of all object files
$(PROJECT): $(OBJS)
	$(CXX) $(COMMON) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $(COMMON) $(CXXFLAGS) -c $< -o $@
$(BUILD_DIR)/cuda/%.o: %.cu
	$(NVCC) $(CUDA_ARCH) $(COMMON) --compiler-options "$(CXXFLAGS)" -c $< -o $@
dirs:
	@ mkdir -p $(BUILD_DIR)
	@ mkdir -p $(BUILD_DIR)/cuda
	@ mkdir -p $(OTHER_BUILD_DIR)
	@ mkdir -p $(OTHER_BUILD_DIR)/cuda
# clean
.PHONY: clean
clean:
	@ rm -rf $(BUILD_DIR)
	@ rm -rf $(OTHER_BUILD_DIR)
	@ rm -rf $(PROJECT)
