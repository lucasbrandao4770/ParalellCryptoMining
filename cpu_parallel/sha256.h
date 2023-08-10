#ifndef SHA256_H
#define SHA256_H

#include <stdint.h>

#define SHA256_DIGEST_LENGTH 32

void int_to_big_endian(uint32_t num, unsigned char* arr);
uint32_t big_endian_to_int(const unsigned char* arr);
void sha256(const unsigned char* message, size_t message_len, unsigned char* hash_result);

#endif // SHA256_H
