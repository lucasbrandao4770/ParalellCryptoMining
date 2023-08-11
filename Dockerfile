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
COPY cpu_parallel/ /app/cpu_parallel/
COPY utils/ /app/utils/
COPY openmp_miner.c /app/
WORKDIR /app
RUN make -f cpu_parallel/Makefile

# Stage for GPU Parallelization (CUDA)
FROM nvidia/cuda:11.4.2-base AS gpu_parallel
COPY gpu_parallel/ /app/gpu_parallel/
COPY cuda_miner.cu /app/
WORKDIR /app
RUN make -f gpu_parallel/Makefile

# Stage for Sequential Code
FROM base AS sequential
COPY sequential/ /app/sequential/
COPY utils/ /app/utils/
COPY sequential_miner.c /app/
WORKDIR /app
RUN make -f sequential/Makefile

# Final stage to gather all binaries
FROM base AS final
COPY --from=cpu_parallel /app/openmp_miner /app/
COPY --from=gpu_parallel /app/cuda_miner /app/
COPY --from=sequential /app/sequential_miner /app/
WORKDIR /app
CMD ["./run_all.sh"]
