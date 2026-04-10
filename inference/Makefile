CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -mavx2 -mavxvnni -fopenmp -Wall -Wextra
LDFLAGS = -fopenmp -lm

# oneDNN paths
DNNL_ROOT = /home/shub/intel/oneapi/dnnl/2025.3
DNNL_INC = -I$(DNNL_ROOT)/include
DNNL_LIB = -L$(DNNL_ROOT)/lib -ldnnl -Wl,-rpath,$(DNNL_ROOT)/lib

SRC_DIR = src
BUILD_DIR = build
TARGET = $(BUILD_DIR)/ternary_infer
TARGET_4BIT = $(BUILD_DIR)/ternary_infer_4bit
TARGET_4BIT_DPBUSD = $(BUILD_DIR)/ternary_infer_4bit_dpbusd
TARGET_DNNL = $(BUILD_DIR)/ternary_infer_dnnl

SRCS = $(SRC_DIR)/main.cpp
SRCS_4BIT = $(SRC_DIR)/main_4bit.cpp
SRCS_4BIT_DPBUSD = $(SRC_DIR)/main_4bit_dpbusd.cpp
SRCS_DNNL = $(SRC_DIR)/main_dnnl.cpp
HEADERS = $(SRC_DIR)/engine.h $(SRC_DIR)/model_loader.h
HEADERS_4BIT = $(SRC_DIR)/engine_4bit.h $(SRC_DIR)/model_loader.h
HEADERS_4BIT_DPBUSD = $(SRC_DIR)/engine_4bit_dpbusd.h $(SRC_DIR)/model_loader.h
HEADERS_DNNL = $(SRC_DIR)/engine_dnnl.h $(SRC_DIR)/model_loader.h

all: $(TARGET)

4bit: $(TARGET_4BIT)

4bit-dpbusd: $(TARGET_4BIT_DPBUSD)

dnnl: $(TARGET_DNNL)

$(TARGET): $(SRCS) $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) $(LDFLAGS)

$(TARGET_4BIT): $(SRCS_4BIT) $(HEADERS_4BIT) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS_4BIT) $(LDFLAGS)

$(TARGET_4BIT_DPBUSD): $(SRCS_4BIT_DPBUSD) $(HEADERS_4BIT_DPBUSD) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS_4BIT_DPBUSD) $(LDFLAGS)

$(TARGET_DNNL): $(SRCS_DNNL) $(HEADERS_DNNL) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(DNNL_INC) -o $@ $(SRCS_DNNL) $(LDFLAGS) $(DNNL_LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all 4bit 4bit-dpbusd dnnl clean
