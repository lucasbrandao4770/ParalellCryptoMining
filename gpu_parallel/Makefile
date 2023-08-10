# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -O2

# Target executable
TARGET = cuda_miner

# CUDA source files
CU_SRC = blockchain.cu utilities.cu sha256.cu ../cuda_miner.cu

# CUDA header files
CU_HEADERS = blockchain.cuh utilities.cuh sha256.cuh

# CUDA object files
CU_OBJ = $(CU_SRC:.cu=.o)

# Compile and link target
$(TARGET): $(CU_OBJ)
	$(NVCC) $(CFLAGS) $(CU_OBJ) -o $(TARGET)

# Compile CUDA source files to object files
%.o: %.cu $(CU_HEADERS)
	$(NVCC) $(CFLAGS) -dc $< -o $@

# Clean target to remove object files and the executable
clean:
ifeq ($(OS),Windows_NT)
	-del /f /q ..\cuda_miner.o blockchain.o sha256.o utilities.o cuda_miner.exe cuda_miner.exp cuda_miner.lib
else
	-rm -f ..\cuda_miner.o blockchain.o sha256.o utilities.o cuda_miner cuda_miner.exp cuda_miner.lib
endif
