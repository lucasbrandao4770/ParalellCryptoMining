#!/bin/bash
echo "Running Sequential Miner..."
time ./sequential_miner
echo "Running OpenMP Miner..."
time ./openmp_miner
echo "Running CUDA Miner..."
time ./cuda_miner
