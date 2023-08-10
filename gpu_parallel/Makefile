# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -O2

# Target executable
TARGET = ../cuda_miner

# CUDA source files
CU_SRC = blockchain.cu utilities.cu sha256.cu ../cuda_miner.cu

# CUDA header files
CU_HEADERS = blockchain.cuh utilities.cuh sha256.cuh

# C source files
C_SRC = ../utils/arg_parser.c

# Include paths for header files
INCLUDES = -I../utils

# CUDA object files
CU_OBJ = $(CU_SRC:.cu=.o)

# C object files
C_OBJ = $(C_SRC:.c=.o)

# Detecting the operating system
ifeq ($(OS),Windows_NT)
    RM = del /F /Q
    CLEAN_FILES = $(CU_OBJ) $(C_OBJ) $(TARGET).exe $(TARGET).exp $(TARGET).lib
else
    RM = rm -f
    CLEAN_FILES = $(CU_OBJ) $(C_OBJ) $(TARGET)
endif

# Compile and link target
$(TARGET): $(CU_OBJ) $(C_OBJ)
	$(NVCC) $(CFLAGS) $(INCLUDES) $(CU_OBJ) $(C_OBJ) -o $(TARGET)

# Compile CUDA source files to object files
%.o: %.cu $(CU_HEADERS)
	$(NVCC) $(CFLAGS) -dc $< -o $@

# Compile C source files to object files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean target
clean:
	$(RM) $(CLEAN_FILES)
