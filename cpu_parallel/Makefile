# Compiler to use (GCC)
CC = gcc

# Compiler flags (warnings, optimization level, and OpenMP support)
CFLAGS = -Wall -O2 -fopenmp

# Target executable name
TARGET = blockchain_openmp

# Source file to be compiled
SOURCES = blockchain_openmp.c

# Object file generated from the source file
OBJECTS = $(SOURCES:.c=.o)

# Rule to build the target executable
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(TARGET)

# Rule to compile C source file into object file
.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

# Clean target to remove object file and the executable
clean:
ifeq ($(OS),Windows_NT)
	-del /f /q blockchain_openmp.o blockchain_openmp.exe
else
	-rm -f blockchain_openmp.o blockchain_openmp
endif

