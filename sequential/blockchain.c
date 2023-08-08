#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "arg_parser.h"

#define MAX_DATA_SIZE 100
#define MAX_HASH_SIZE 65
#define SHA256_DIGEST_LENGTH 32

typedef struct {
    int index;
    time_t timestamp;
    char data[MAX_DATA_SIZE];
    char previous_hash[MAX_HASH_SIZE];
    int difficulty;
    char hash[MAX_HASH_SIZE];
} Block;

typedef struct {
    Block* blocks;
    int size;
    int capacity;
} Blockchain;

// SHA-256
const unsigned int k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

void int_to_big_endian(uint32_t num, unsigned char* arr) {
    arr[0] = (unsigned char)(num >> 24);
    arr[1] = (unsigned char)(num >> 16);
    arr[2] = (unsigned char)(num >> 8);
    arr[3] = (unsigned char)num;
}

uint32_t big_endian_to_int(const unsigned char* arr) {
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
    if (!padded_message) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    memcpy(padded_message, message, message_len);
    padded_message[message_len] = 0x80;
    int_to_big_endian((uint32_t)(message_len * 8), &padded_message[padded_len - 8]);

    for (size_t chunk_start = 0; chunk_start < padded_len; chunk_start += 64) {
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

        for (int t = 0; t < 16; t++) {
            w[t] = big_endian_to_int(&padded_message[chunk_start + t * 4]);
        }
        for (int t = 16; t < 64; t++) {
            uint32_t s0 = ROTR(w[t - 15], 7) ^ ROTR(w[t - 15], 18) ^ (w[t - 15] >> 3);
            uint32_t s1 = ROTR(w[t - 2], 17) ^ ROTR(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16] + s0 + w[t - 7] + s1;
        }

        for (int t = 0; t < 64; t++) {
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

        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += h_temp;
    }

    for (int i = 0; i < 8; i++) {
        int_to_big_endian(h[i], &hash_result[i * 4]);
    }

    free(padded_message);
}
// SHA-256

void calculate_hash(Block* block) {
    char combined_data[MAX_DATA_SIZE + MAX_HASH_SIZE + 20];
    unsigned char hash_result[SHA256_DIGEST_LENGTH];
    char hash_hex[SHA256_DIGEST_LENGTH * 2 + 1];
    int nonce = 0;

    while (1) {
        snprintf(combined_data, sizeof(combined_data), "%d%ld%s%s%d", block->index, block->timestamp, block->data, block->previous_hash, nonce);
        sha256((unsigned char*)combined_data, strlen(combined_data), hash_result);

        for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
            sprintf(&hash_hex[i * 2], "%02x", hash_result[i]);
        }

        int valid_hash = 1;
        for (int i = 0; i < block->difficulty; i++) {
            if (hash_hex[i] != '0') {
                valid_hash = 0;
                break;
            }
        }

        if (valid_hash) {
            strncpy(block->hash, hash_hex, MAX_HASH_SIZE);
            break;
        }

        nonce++;
    }
}

Block create_genesis_block(int difficulty) {
    Block genesis_block;
    genesis_block.index = 0;
    genesis_block.timestamp = time(NULL);
    strncpy(genesis_block.data, "Genesis Block", MAX_DATA_SIZE);
    strncpy(genesis_block.previous_hash, "0", MAX_HASH_SIZE);
    genesis_block.difficulty = difficulty;

    calculate_hash(&genesis_block);

    return genesis_block;
}

void add_block(Blockchain* blockchain, const char* data) {
    if (blockchain->size == blockchain->capacity) {
        blockchain->capacity *= 2;
        blockchain->blocks = realloc(blockchain->blocks, blockchain->capacity * sizeof(Block));
    }

    Block previous_block = blockchain->blocks[blockchain->size - 1];

    Block new_block;
    new_block.index = blockchain->size;
    new_block.timestamp = time(NULL);
    strncpy(new_block.data, data, MAX_DATA_SIZE);
    strncpy(new_block.previous_hash, previous_block.hash, MAX_HASH_SIZE);
    new_block.difficulty = previous_block.difficulty;

    calculate_hash(&new_block);

    blockchain->blocks[blockchain->size] = new_block;
    blockchain->size++;
}

void print_block(const Block* block) {
    printf("Index: %d\n", block->index);
    printf("Timestamp: %ld\n", block->timestamp);
    printf("Data: %s\n", block->data);
    printf("Previous Hash: %s\n", block->previous_hash);
    printf("Hash: %s\n", block->hash);
    printf("====================\n");
}

void print_blockchain(const Blockchain* blockchain) {
    for (int i = 0; i < blockchain->size; i++) {
        const Block* block = &blockchain->blocks[i];
        print_block(block);
    }
}

int main(int argc, char *argv[]) {
    int difficulty = 2;
    int size = 0;
    int capacity = 10;

    KeyValue keyValues[] = {
        {"difficulty", &difficulty},
        {"size", &size},
        {"capacity", &capacity}
    };

    parse_arguments(argc, argv, keyValues, sizeof(keyValues) / sizeof(KeyValue));

    Blockchain blockchain;
    blockchain.size = size;
    blockchain.capacity = capacity;
    blockchain.blocks = malloc(blockchain.capacity * sizeof(Block));

    Block genesis_block = create_genesis_block(difficulty);
    blockchain.blocks[blockchain.size] = genesis_block;
    blockchain.size++;

    add_block(&blockchain, "Data of Block 1");
    add_block(&blockchain, "Data of Block 2");
    add_block(&blockchain, "Data of Block 3");

    print_blockchain(&blockchain);

    free(blockchain.blocks);

    return 0;
}
