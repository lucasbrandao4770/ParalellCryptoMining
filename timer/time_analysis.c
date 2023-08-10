#include "timer/timer.h"

int main() {
    Timer timer;
    double cpu_time, omp_time;
    float cuda_time;

    timer_start(&timer);

    // Code to be measured

    timer_stop(&timer, &cpu_time, &omp_time, &cuda_time);
    printf("CPU Time: %f seconds\n", cpu_time);
    printf("OpenMP Time: %f seconds\n", omp_time);
    printf("CUDA Time: %f milliseconds\n", cuda_time);

    return 0;
}
