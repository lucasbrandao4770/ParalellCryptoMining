
#ifndef BLOCKCHAIN_H
#define BLOCKCHAIN_H

#include <time.h>

#define MAX_DATA_SIZE 100 /**< Maximum size of the data field in a block, defining how much information can be stored. */
#define MAX_HASH_SIZE 65 /**< Maximum size of the hash in a block, accommodating the SHA-256 hash plus null termination. */

/**
 * @brief Structure representing a block in the blockchain.
 *
 * A block consists of the following components:
 * - Index: Position of the block within the blockchain.
 * - Timestamp: Time when the block was created.
 * - Data: Content or transactions stored in the block.
 * - Previous Hash: Ensures integrity by linking to the previous block.
 * - Difficulty: Determines how hard it is to mine the block.
 * - Hash: Unique identifier and verification of the block's content.
 */
typedef struct {
    int index; /**< Index of the block in the blockchain. */
    time_t timestamp; /**< Time when the block was created. */
    char data[MAX_DATA_SIZE]; /**< Content or transactions stored in the block. */
    char previous_hash[MAX_HASH_SIZE]; /**< Hash of the previous block, creating a chain. */
    int difficulty; /**< Number of leading zeros required in the block's hash. */
    char hash[MAX_HASH_SIZE]; /**< SHA-256 hash representing the block's content. */
} Block;

/**
 * @brief Structure representing the entire blockchain.
 *
 * A blockchain consists of:
 * - Blocks: Dynamic array of blocks forming the chain.
 * - Size: Number of blocks currently in the blockchain.
 * - Capacity: Allocated space, allowing for efficient addition of new blocks.
 */
typedef struct {
    Block* blocks; /**< Dynamic array of blocks. */
    int size; /**< Current number of blocks in the chain. */
    int capacity; /**< Allocated space, used for efficient management of memory. */
} Blockchain;

/**
 * @brief Creates and initializes the genesis block of the blockchain.
 *
 * The genesis block is the first block in the chain and has a predefined content.
 *
 * @param difficulty Difficulty level for mining the genesis block.
 * @return The created genesis block.
 */
Block create_genesis_block(int difficulty);

/**
 * @brief Adds a new block to the blockchain with the given data.
 *
 * Ensures that the blockchain's capacity is sufficient and initializes the new block's properties.
 *
 * @param blockchain Pointer to the blockchain where the block will be added.
 * @param data Data content for the new block.
 */
void add_block(Blockchain* blockchain, const char* data);

/**
 * @brief Calculates the hash for a given block based on its content and difficulty level.
 *
 * Combines the block's components and uses the SHA-256 algorithm to calculate the hash.
 *
 * @param block Pointer to the block whose hash will be calculated.
 */
void calculate_hash(Block* block);

/**
 * @brief Prints the details of a given block in a human-readable format.
 *
 * Useful for debugging or inspecting individual blocks.
 *
 * @param block Pointer to the block to be printed.
 */
void print_block(const Block* block);

/**
 * @brief Prints the details of the entire blockchain, showing the chain of blocks.
 *
 * This function provides an overview of the blockchain's structure and content.
 * It can be useful for debugging, analysis, or visual representation of the blockchain.
 *
 * @param blockchain Pointer to the blockchain to be printed.
 */
void print_blockchain(const Blockchain* blockchain);

#endif // BLOCKCHAIN_H
