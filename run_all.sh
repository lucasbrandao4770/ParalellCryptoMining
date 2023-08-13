#!/bin/bash

# Default values
DIFFICULTY=5

# Parse arguments
for arg in "$@"; do
    case $arg in
        difficulty=*)
        DIFFICULTY="${arg#*=}"
        ;;
        # Add other arguments here
    esac
done
# Use $DIFFICULTY as needed for your miners


# Navigate to the top-level directory
cd "$(dirname "$0")"

# Create the results directory if it doesn't exist
mkdir -p results

# Get the date and time
FILENAME=$(date "+%Y-%m-%d_%H-%M-%S")

# Construct filename with command-line arguments (if any)
for arg in "$@"; do
    FILENAME="${FILENAME}_${arg}"
done

# Add .txt extension
FILENAME="results/${FILENAME}.txt"

# Compile and run the Sequential miner
echo "Running Sequential Miner:"
cd sequential
make
echo "Time spent for sequential code:" >> "${FILENAME}"
/usr/bin/time -o "${FILENAME}" -a ./sequential_miner difficulty=$DIFFICULTY"$@"
make clean
cd ..

# Compile and run the OpenMP miner
echo "Running OpenMP Miner:"
cd cpu_parallel
make
echo "Time spent for OpenMP code:" >> "${FILENAME}"
/usr/bin/time -o "${FILENAME}" -a ./openmp_miner difficulty=$DIFFICULTY"$@"
make clean
cd ..

# Compile and run the CUDA miner
echo "Running CUDA Miner:"
cd gpu_parallel
make
echo "Time spent for CUDA code:" >> "${FILENAME}"
/usr/bin/time -o "${FILENAME}" -a ./cuda_miner difficulty=$DIFFICULTY"$@"
make clean
cd ..
