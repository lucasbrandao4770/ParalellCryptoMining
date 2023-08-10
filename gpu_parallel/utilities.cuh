#ifndef UTILITIES_H
#define UTILITIES_H

#include "sha256.cuh"

__device__ int custom_strlen(const char* str);
__device__ void custom_strncpy(char* dest, const char* src, int n);
__device__ void int_to_str(int value, char* result);
__device__ void to_hex_string(BYTE* bytes, char* hex, int length);

#endif   // UTILITIES_H
