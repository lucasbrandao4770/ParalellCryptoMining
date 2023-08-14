// Constants, Macros, and Functions related to SHA-256
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sha256.h"

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n)))) /**< Macro for right rotation of x by n bits. */

const unsigned int k[64] = { /**< Constants used in the SHA-256 algorithm, as defined in the standard. */
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

void int_to_big_endian(uint32_t num, unsigned char* arr) {
    /**< Converts a 32-bit integer to its big-endian representation and stores it in the provided array. */
    arr[0] = (unsigned char)(num >> 24);
    arr[1] = (unsigned char)(num >> 16);
    arr[2] = (unsigned char)(num >> 8);
    arr[3] = (unsigned char)num;
}

uint32_t big_endian_to_int(const unsigned char* arr) {
    /**< Converts a big-endian array to a 32-bit integer and returns the result. */
    return (uint32_t)((arr[0] << 24) | (arr[1] << 16) | (arr[2] << 8) | arr[3]);
}

void sha256(const unsigned char* message, size_t message_len, unsigned char* hash_result) {
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    size_t padded_len = message_len + 1 + 8;
    while (padded_len % 64 != 0) {
        padded_len++;
    }

    unsigned char* padded_message = (unsigned char*)calloc(padded_len, sizeof(unsigned char));
    /**< Padding the message as per the SHA-256 specification. */
    if (!padded_message) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    memcpy(padded_message, message, message_len);
    padded_message[message_len] = 0x80;
    int_to_big_endian((uint32_t)(message_len * 8), &padded_message[padded_len - 8]);

    for (size_t chunk_start = 0; chunk_start < padded_len; chunk_start += 64) {
        /**< Processing chunks of 64 bytes from the padded message. */
        uint32_t w[64] = { 0 };
        uint32_t a, b, c, d, e, f, g, h_temp;

        a = h[0];
        b = h[1];
        c = h[2];
        d = h[3];
        e = h[4];
        f = h[5];
        g = h[6];
        h_temp = h[7];

        #pragma omp parallel for
        for (int t = 0; t < 16; t++) {
            w[t] = big_endian_to_int(&padded_message[chunk_start + t * 4]);
        }

        #pragma omp parallel for
        for (int t = 16; t < 64; t++) {
            /**< Expanding the message schedule by applying the SHA-256 operations for each 32-bit word. */
            uint32_t s0 = ROTR(w[t - 15], 7) ^ ROTR(w[t - 15], 18) ^ (w[t - 15] >> 3);
            uint32_t s1 = ROTR(w[t - 2], 17) ^ ROTR(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16] + s0 + w[t - 7] + s1;
        }

        for (int t = 0; t < 64; t++) {
            /**< Main compression loop of the SHA-256 algorithm, applying bitwise operations as per the standard. */
            uint32_t s1 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h_temp + s1 + ch + k[t] + w[t];
            uint32_t s0 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = s0 + maj;

            h_temp = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        h[0] += a; /**< Updating the hash state with the temporary variables. */
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += h_temp;
    }

    for (int i = 0; i < 8; i++) {
        /**< Converting the hash state to big-endian and storing it in the result array. */
        int_to_big_endian(h[i], &hash_result[i * 4]);
    }

    free(padded_message); /**< Freeing the allocated memory for the padded message. */
}
