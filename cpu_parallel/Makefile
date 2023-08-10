# Compiler to use (GCC)
CC = gcc

# Compiler flags (warnings, optimization level, and OpenMP support)
CFLAGS = -Wall -O2 -fopenmp

# Target executable name
TARGET = openmp_miner

# Source file to be compiled
SOURCES = ../openmp_miner.c ../cpu_parallel/block.c blockchain.c sha256.c ../utils/arg_parser.c

# Include paths for header files
INCLUDES = -I../utils

# Object files generated from source files
OBJECTS = $(SOURCES:.c=.o)

# Rule to build the target executable
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJECTS) -o $(TARGET)

# Rule to compile C source files into object files
.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean target to remove object file and the executable
clean:
ifeq ($(OS),Windows_NT)
	-del /f /q ..\openmp_miner.o block.o blockchain.o sha256.o openmp_miner.exe ..\utils\arg_parser.o
else
	-rm -f ..\openmp_miner.o block.o blockchain.o sha256.o openmp_miner ..\utils\arg_parser.o
endif
