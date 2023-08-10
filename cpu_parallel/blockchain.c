#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <omp.h>

#include "blockchain.h"
#include "sha256.h"

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
    strncpy(new_block.data, data, MAX_DATA_SIZE - 1);
    new_block.data[MAX_DATA_SIZE - 1] = '\0'; // Ensure null termination
    strncpy(new_block.previous_hash, previous_block.hash, MAX_HASH_SIZE);
    new_block.difficulty = previous_block.difficulty;

    calculate_hash(&new_block);

    blockchain->blocks[blockchain->size] = new_block;
    blockchain->size++;
}

void calculate_hash(Block* block) {
    char combined_data[MAX_DATA_SIZE + MAX_HASH_SIZE + 20];
    unsigned char hash_result[SHA256_DIGEST_LENGTH];
    char hash_hex[SHA256_DIGEST_LENGTH * 2 + 1];
    int valid_hash = 0;
    int found_nonce = 0;

    #pragma omp parallel private(combined_data, hash_result, hash_hex)
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int max_nonce = INT_MAX / num_threads;
        int start_nonce = thread_id * max_nonce;
        int end_nonce = start_nonce + max_nonce;

        for (int nonce = start_nonce; nonce < end_nonce; nonce++) {
            if (valid_hash) break; // Check if a valid hash has been found by another thread

            snprintf(combined_data, sizeof(combined_data), "%d%lld%s%s%d", block->index, block->timestamp, block->data, block->previous_hash, nonce);
            sha256((unsigned char*)combined_data, strlen(combined_data), hash_result);

            for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
                sprintf(&hash_hex[i * 2], "%02x", hash_result[i]);
            }

            int local_valid_hash = 1;
            for (int i = 0; i < block->difficulty; i++) {
                if (hash_hex[i] != '0') {
                    local_valid_hash = 0;
                    break;
                }
            }

            #pragma omp critical // Protect writing to the shared variable
            if (local_valid_hash && !valid_hash) {
                valid_hash = local_valid_hash;
                found_nonce = nonce;
                strncpy(block->hash, hash_hex, MAX_HASH_SIZE);
            }
        }
    }

    printf("Nonce: %d\n", found_nonce); // Printing the nonce for verification
}

void print_block(const Block* block) {
    printf("Index: %d\n", block->index);
    printf("Timestamp: %lld\n", block->timestamp);
    printf("Data: %s\n", block->data);
    printf("Previous Hash: %s\n", block->previous_hash);
    printf("Hash: %s\n", block->hash);
    printf("====================\n");
}

void print_blockchain(const Blockchain* blockchain) {
    printf("====================\n");
    for (int i = 0; i < blockchain->size; i++) {
        const Block* block = &blockchain->blocks[i];
        print_block(block);
    }
}
