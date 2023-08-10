#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>

#include "sha256.cuh"
#include "blockchain.cuh"
#include "utilities.cuh"

__global__ void calculate_hash_kernel(int index, time_t timestamp, char* data, char* previous_hash, int difficulty, char* result, int* found_nonce) {
    int nonce = blockIdx.x * blockDim.x + threadIdx.x;
    SHA256_CTX sha256;
    BYTE hash[SHA256_DIGEST_LENGTH];
    char input[MAX_DATA_SIZE + MAX_HASH_SIZE + sizeof(int) * 3];
    char hash_hex[MAX_HASH_SIZE];

    char buffer[16];

    int offset = 0;
    int_to_str(index, buffer);
    custom_strncpy(input + offset, buffer, custom_strlen(buffer));
    offset += custom_strlen(buffer);
    input[offset++] = '\0'; // Adiciona caractere de terminação null

    int_to_str((int)timestamp, buffer);
    custom_strncpy(input + offset, buffer, custom_strlen(buffer));
    offset += custom_strlen(buffer);
    input[offset++] = '\0'; // Adiciona caractere de terminação null

    int_to_str(difficulty, buffer);
    custom_strncpy(input + offset, buffer, custom_strlen(buffer));
    offset += custom_strlen(buffer);
    input[offset++] = '\0'; // Adiciona caractere de terminação null

    int_to_str(nonce, buffer);
    custom_strncpy(input + offset, buffer, custom_strlen(buffer));
    offset += custom_strlen(buffer);
    input[offset++] = '\0'; // Adiciona caractere de terminação null

    custom_strncpy(input + offset, data, custom_strlen(data)); // Adiciona o data ao input
    offset += custom_strlen(data);
    input[offset++] = '\0'; // Adiciona caractere de terminação null

    custom_strncpy(input + offset, previous_hash, custom_strlen(previous_hash)); // Adiciona o previous_hash ao input
    offset += custom_strlen(previous_hash);
    input[offset++] = '\0'; // Adiciona caractere de terminação null


    sha256_init(&sha256);
    sha256_update(&sha256, (BYTE*)input, offset);
    sha256_final(&sha256, hash);

    to_hex_string(hash, hash_hex, SHA256_DIGEST_LENGTH);

    int valid_hash = 1;
    for (int i = 0; i < difficulty; i++) {
        if (hash_hex[i] != '0') {
            valid_hash = 0;
            break;
        }
    }

    if (valid_hash) {
        if (atomicExch(found_nonce, nonce) == -1) {
            custom_strncpy(result, hash_hex, MAX_HASH_SIZE);
        }
    }
}


void calculate_hash(Block* block) {
    char* d_data;
    char* d_previous_hash;
    char* d_result;
    int* d_found_nonce;
    int found_nonce = -1;

    cudaMalloc((void**)&d_data, MAX_DATA_SIZE);
    cudaMalloc((void**)&d_previous_hash, MAX_HASH_SIZE);
    cudaMalloc((void**)&d_result, MAX_HASH_SIZE);
    cudaMalloc((void**)&d_found_nonce, sizeof(int));

    cudaMemcpy(d_data, block->data, MAX_DATA_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_previous_hash, block->previous_hash, MAX_HASH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_nonce, &found_nonce, sizeof(int), cudaMemcpyHostToDevice);

    char result[MAX_HASH_SIZE] = {0};
    cudaMemcpy(d_result, result, MAX_HASH_SIZE, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(NUM_THREADS);
    dim3 numBlocks((int)ceil(1.0 * 1024 * 1024 * 1024 / threadsPerBlock.x));

    int attempts = 0;
    while(found_nonce == -1) {  // Keep trying until a nonce is found
        calculate_hash_kernel<<<numBlocks, threadsPerBlock>>>(block->index, block->timestamp, d_data, d_previous_hash, block->difficulty, d_result, d_found_nonce);
        cudaDeviceSynchronize();

        cudaMemcpy(&found_nonce, d_found_nonce, sizeof(int), cudaMemcpyDeviceToHost);
        attempts++;

        if (attempts > 10) { // Just an arbitrary number to limit the number of attempts.
            printf("Failed to find a valid nonce after %d attempts\n", attempts);
            break;
        }
    }

    cudaMemcpy(block->hash, d_result, MAX_HASH_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_previous_hash);
    cudaFree(d_result);
    cudaFree(d_found_nonce);

    printf("Found nonce: %d\n", found_nonce);
}


// main, etc.
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
        blockchain->blocks = (Block*)realloc(blockchain->blocks, blockchain->capacity * sizeof(Block));
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
