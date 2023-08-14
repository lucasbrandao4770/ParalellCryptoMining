#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "blockchain.h"
#include "sha256.h"

Block create_genesis_block(int difficulty) {
    /**< Creates and initializes the genesis block of the blockchain with given difficulty. */
    Block genesis_block;
    genesis_block.index = 0; /**< Index of the genesis block (first block in the chain). */
    genesis_block.timestamp = time(NULL); /**< Timestamp of the genesis block (current time). */
    strncpy(genesis_block.data, "Genesis Block", MAX_DATA_SIZE); /**< Data content of the genesis block. */
    strncpy(genesis_block.previous_hash, "0", MAX_HASH_SIZE);
    genesis_block.difficulty = difficulty; /**< Difficulty level for mining the genesis block. */

    calculate_hash(&genesis_block);

    return genesis_block;
}

void add_block(Blockchain* blockchain, const char* data) {
    /**< Adds a new block to the blockchain with the given data. */
    if (blockchain->size == blockchain->capacity) {
        /**< Checking and expanding the capacity of the blockchain if necessary. */
        blockchain->capacity *= 2;
        blockchain->blocks = realloc(blockchain->blocks, blockchain->capacity * sizeof(Block));
    }

    Block previous_block = blockchain->blocks[blockchain->size - 1];

    Block new_block;
    new_block.index = blockchain->size; /**< Index of the new block. */
    new_block.timestamp = time(NULL); /**< Timestamp of the new block (current time). */
    strncpy(new_block.data, data, MAX_DATA_SIZE - 1);
    new_block.data[MAX_DATA_SIZE - 1] = '\0'; /**< Ensure null termination of the data string. */
    strncpy(new_block.previous_hash, previous_block.hash, MAX_HASH_SIZE);
    new_block.difficulty = previous_block.difficulty;

    calculate_hash(&new_block);

    blockchain->blocks[blockchain->size] = new_block;
    blockchain->size++;
}

void calculate_hash(Block* block) {
    /**< Calculates the hash for a given block based on its content and difficulty level. */
    char combined_data[MAX_DATA_SIZE + MAX_HASH_SIZE + 20];
    unsigned char hash_result[SHA256_DIGEST_LENGTH];
    char hash_hex[SHA256_DIGEST_LENGTH * 2 + 1];
    int nonce = 0;

    while (1) {
        snprintf(combined_data, sizeof(combined_data), "%d%lld%s%s%d", block->index, block->timestamp, block->data, block->previous_hash, nonce);
        /**< Combining the block's data, index, timestamp, previous hash, and nonce into a single string for hashing. */
        sha256((unsigned char*)combined_data, strlen(combined_data), hash_result);

        for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        /**< Converting the hash into a hexadecimal string for further processing. */
            sprintf(&hash_hex[i * 2], "%02x", hash_result[i]);
        }

        int valid_hash = 1; /**< Flag to check if the hash meets the difficulty criteria (leading zeros). */
        for (int i = 0; i < block->difficulty; i++) {
            if (hash_hex[i] != '0') {
                valid_hash = 0;
                break;
            }
        }

        if (valid_hash) {
            /**< If the hash meets the difficulty criteria, copy it to the block's hash and exit the loop. */
            strncpy(block->hash, hash_hex, MAX_HASH_SIZE);
            break;
        }

        nonce++;
    }
}

void print_block(const Block* block) {
    /**< Prints the details of a given block, including index, timestamp, data, previous hash, and hash. */
    printf("Index: %d\n", block->index);
    printf("Timestamp: %lld\n", block->timestamp);
    printf("Data: %s\n", block->data);
    printf("Previous Hash: %s\n", block->previous_hash);
    printf("Hash: %s\n", block->hash);
    printf("====================\n");
}

void print_blockchain(const Blockchain* blockchain) {
    /**< Prints the details of the entire blockchain. */
    printf("====================\n");
    for (int i = 0; i < blockchain->size; i++) {
        const Block* block = &blockchain->blocks[i];
        print_block(block);
    }
}
