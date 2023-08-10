#include <cuda_runtime.h>
#include <string.h>
#include <stdlib.h>

#include "sha256.cuh"
#include "utilities.cuh"

void parse_arguments(int argc, char *argv[], KeyValue *keyValues, size_t keyValueCount) {
    for (int i = 1; i < argc; i++) {
        char *equalSign = strchr(argv[i], '=');
        if (equalSign != NULL) {
            *equalSign = '\0'; // Replace '=' with null terminator to split key and value
            const char *key = argv[i];
            const char *value = equalSign + 1;

            for (size_t j = 0; j < keyValueCount; j++) {
                if (strcmp(key, keyValues[j].key) == 0) {
                    *keyValues[j].value = atoi(value);
                    break;
                }
            }
        }
    }
}

__device__ int custom_strlen(const char* str) {
    int length = 0;
    while (str[length] != '\0') {
        length++;
    }
    return length;
}

__device__ void custom_strncpy(char* dest, const char* src, int n) {
    for (int i = 0; i < n && src[i] != '\0'; i++) {
        dest[i] = src[i];
    }
}

__device__ void int_to_str(int value, char* result) {
    int i = 0;
    if (value == 0) {
        result[0] = '0';
        result[1] = '\0';
        return;
    }

    while (value != 0) {
        result[i++] = (value % 10) + '0';
        value /= 10;
    }
    result[i] = '\0';

    // Reverse the result string
    int start = 0;
    int end = i - 1;
    char temp;
    while (start < end) {
        temp = result[start];
        result[start] = result[end];
        result[end] = temp;
        start++;
        end--;
    }
}

__device__ void to_hex_string(BYTE* bytes, char* hex, int length) {
    const char hex_digits[] = "0123456789abcdef";
    for (int i = 0; i < length; i++) {
        hex[i * 2] = hex_digits[(bytes[i] >> 4) & 0xF];
        hex[i * 2 + 1] = hex_digits[bytes[i] & 0xF];
    }
    hex[length * 2] = '\0';
}
