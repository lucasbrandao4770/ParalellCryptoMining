#!/bin/bash

# Navigate to the top-level directory
cd ..

# Capture all command-line arguments
ARGS="$@"

# Compile and run the Sequential miner
echo "Running Sequential Miner:"
cd ./sequential
make
time ./sequential_miner $ARGS
make clean
cd ..

# Compile and run the OpenMP miner
echo "Running OpenMP Miner:"
cd ./cpu_parallel
make
time ./openmp_miner $ARGS
make clean
cd ..

# Compile and run the CUDA miner
echo "Running CUDA Miner:"
cd ./gpu_parallel
make
time ./cuda_miner $ARGS
make clean
cd ..
