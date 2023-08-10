# Compiler to use (GCC)
CC = gcc

# Compiler flags (warnings and optimization level)
CFLAGS = -Wall -O2

# Target executable name
TARGET = sequential_miner

# Source files to be compiled
SOURCES = ../sequential_miner.c ../sequential/blockchain.c sha256.c ../utils/arg_parser.c

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

# Clean target to remove object files and the executable
clean:
ifeq ($(OS),Windows_NT)
	-del /f /q ..\sequential_miner.o blockchain.o sha256.o sequential_miner.exe ..\utils\arg_parser.o
else
	-rm -f ..\sequential_miner.o blockchain.o sha256.o sequential_miner ..\utils\arg_parser.o
endif
