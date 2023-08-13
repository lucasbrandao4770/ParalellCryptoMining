# Base stage for common dependencies
FROM debian:bullseye-slim AS base
RUN apt-get update && apt-get install -y \
    gcc \
    make \
    git

# Stage for CPU Parallelization (OpenMP)
FROM base AS cpu_parallel
RUN apt-get install -y \
    libomp-dev
COPY . /app/
WORKDIR /app/cpu_parallel
RUN make

# Stage for GPU Parallelization (CUDA)
FROM nvidia/cuda:11.0.3-devel AS gpu_parallel
COPY . /app/
WORKDIR /app/gpu_parallel
RUN make

# Stage for Sequential Code
FROM base AS sequential
COPY . /app/
WORKDIR /app/sequential
RUN make

# Final stage to gather all binaries
FROM base AS final
COPY --from=cpu_parallel /app/cpu_parallel/openmp_miner /app/
COPY --from=gpu_parallel /app/gpu_parallel/cuda_miner /app/
COPY --from=sequential /app/sequential/sequential_miner /app/
WORKDIR /app
CMD ./run_all.sh

