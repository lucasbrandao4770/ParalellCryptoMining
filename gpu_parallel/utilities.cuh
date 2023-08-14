#ifndef UTILITIES_H
#define UTILITIES_H

#include <stddef.h>

#include "sha256.cuh"

/**
 * @struct KeyValue
 * @brief Structure representing a key-value pair.
 *
 * The key is a string representing the argument name, and the value is a pointer to an integer to store the value of the argument.
 */
typedef struct {
    const char *key; /**< The key or name of the argument. */
    int *value;      /**< Pointer to the integer value of the argument. */
} KeyValue;

/**
 * @brief Parses the command-line arguments and populates the key-value pairs.
 *
 * The function iterates through the command-line arguments, looks for keys and values separated by an equal sign ('='), and populates the provided key-value pairs.
 * If a match is found between the provided key and the command-line key, the corresponding value is updated.
 *
 * @param argc The count of command-line arguments.
 * @param argv Array of pointers to the command-line arguments.
 * @param keyValues Pointer to an array of key-value pairs to be populated.
 * @param keyValueCount The number of key-value pairs in the array.
 */

void parse_arguments(int argc, char *argv[], KeyValue *keyValues, size_t keyValueCount);

/**
 * @brief Calculates the length of a null-terminated string.
 *
 * @param str Pointer to the null-terminated string.
 * @return Length of the string.
 */
__device__ int custom_strlen(const char* str);

/**
 * @brief Copies the first 'n' characters from the source string to the destination string.
 *
 * @param dest Pointer to the destination buffer.
 * @param src Pointer to the source string.
 * @param n Number of characters to copy.
 */

__device__ void custom_strncpy(char* dest, const char* src, int n);

/**
 * @brief Converts an integer value to its string representation.
 *
 * @param value Integer value to convert.
 * @param result Pointer to the buffer to store the resulting string.
 */
__device__ void int_to_str(int value, char* result);

/**
 * @brief Converts a byte array to its corresponding hexadecimal string representation.
 *
 * @param bytes Pointer to the byte array.
 * @param hex Pointer to the buffer to store the resulting hexadecimal string.
 * @param length Length of the byte array.
 */
__device__ void to_hex_string(BYTE* bytes, char* hex, int length);

#endif   // UTILITIES_H
