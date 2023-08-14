#ifndef SHA256_H
#define SHA256_H

#include <stdint.h>

#define SHA256_DIGEST_LENGTH 32 /**< Length of the SHA-256 digest in bytes. */

/**
 * @brief Converts a 32-bit integer to its big-endian representation and stores it in the provided array.
 *
 * This function is used to ensure proper byte ordering when working with integer values in the SHA-256 algorithm.
 *
 * @param num The 32-bit integer to convert.
 * @param arr Pointer to the array where the big-endian representation will be stored. Must be pre-allocated with at least 4 bytes.
 */
void int_to_big_endian(uint32_t num, unsigned char* arr);

/**
 * @brief Converts a big-endian array to a 32-bit integer and returns the result.
 *
 * This function is used to interpret a 4-byte big-endian array as a 32-bit integer, commonly used in the SHA-256 algorithm.
 *
 * @param arr Pointer to the big-endian array. Must represent a valid 32-bit integer in big-endian format and contain at least 4 bytes.
 * @return The 32-bit integer representation of the array.
 */
uint32_t big_endian_to_int(const unsigned char* arr);

/**
 * @brief Calculates the SHA-256 hash of a given message.
 *
 * This function takes a message as input and computes its SHA-256 hash using the standard algorithm.
 * The result is a 32-byte (256-bit) hash, which is stored in the provided array.
 *
 * @param message Pointer to the message to be hashed. Must be a valid pointer to the message bytes.
 * @param message_len Length of the message in bytes. Must accurately represent the message length.
 * @param hash_result Pointer to the array where the hash result will be stored. Must be pre-allocated with at least SHA256_DIGEST_LENGTH bytes.
 *
 * Usage example:
 * @code
 * unsigned char message[] = "Hello, world!";
 * unsigned char hash_result[SHA256_DIGEST_LENGTH];
 * sha256(message, strlen(message), hash_result);
 * @endcode
 */
void sha256(const unsigned char* message, size_t message_len, unsigned char* hash_result);

#endif // SHA256_H
