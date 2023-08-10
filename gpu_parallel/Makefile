# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -O2

# Target executable
TARGET = cuda_miner

# CUDA source files
CU_SRC = cuda_miner.cu

# CUDA header files
CU_HEADERS = sha256.cuh

# CUDA object files
CU_OBJ = $(CU_SRC:.cu=.o)

# Detecting the operating system
ifeq ($(OS),Windows_NT)
    RM = del /F /Q
    CLEAN_FILES = $(CU_OBJ) $(TARGET).exe $(TARGET).exp $(TARGET).lib
else
    RM = rm -f
    CLEAN_FILES = $(CU_OBJ) $(TARGET)
endif

# Compile and link target
$(TARGET): $(CU_OBJ)
	$(NVCC) $(CFLAGS) $(CU_OBJ) -o $(TARGET)

# Compile CUDA source files to object files
%.o: %.cu $(CU_HEADERS)
	$(NVCC) $(CFLAGS) -dc $< -o $@

# Clean target
clean:
	$(RM) $(CLEAN_FILES)
