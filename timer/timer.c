#include "timer.h"

void timer_start(Timer* timer) {
    timer->start_time = clock();

    #ifdef _OPENMP
        timer->start_time_omp = omp_get_wtime();
    #endif

    #ifdef __CUDACC__
        cudaEventCreate(&timer->start_event);
        cudaEventCreate(&timer->stop_event);
        cudaEventRecord(timer->start_event, 0);
    #endif
}

void timer_stop(Timer* timer, double* cpu_time, double* omp_time, float* cuda_time) {
    clock_t end_time = clock();
    *cpu_time = ((double)(end_time - timer->start_time)) / CLOCKS_PER_SEC;

    #ifdef _OPENMP
        double end_time_omp = omp_get_wtime();
        *omp_time = end_time_omp - timer->start_time_omp;
    #endif

    #ifdef __CUDACC__
        cudaEventRecord(timer->stop_event, 0);
        cudaEventSynchronize(timer->stop_event);
        cudaEventElapsedTime(cuda_time, timer->start_event, timer->stop_event);
        cudaEventDestroy(timer->start_event);
        cudaEventDestroy(timer->stop_event);
    #endif
}
