CXX = g++
CXXFLAGS = -std=c++17 -O3 -mavx2 -mfma -march=native -Wall
TARGET = ternary_inference
SRC = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRC) src/ternary_engine.h src/model_loader.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) -I src

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) ../export/ternary_resnet18.bin

.PHONY: all clean run
