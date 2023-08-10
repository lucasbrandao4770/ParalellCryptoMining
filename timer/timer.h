#ifndef TIMER_H
#define TIMER_H

#include <time.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

#ifdef __CUDACC__
    #include <cuda_runtime.h>
#endif

typedef struct {
    clock_t start_time;
    double start_time_omp;
#ifdef __CUDACC__
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
#endif
} Timer;

void timer_start(Timer* timer);
void timer_stop(Timer* timer, double* cpu_time, double* omp_time, float* cuda_time);

#endif // TIMER_H
