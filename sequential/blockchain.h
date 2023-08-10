#ifndef BLOCKCHAIN_H
#define BLOCKCHAIN_H

#include <time.h>

#define MAX_DATA_SIZE 100
#define MAX_HASH_SIZE 65

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

Block create_genesis_block(int difficulty);
void add_block(Blockchain* blockchain, const char* data);
void calculate_hash(Block* block);
void print_block(const Block* block);
void print_blockchain(const Blockchain* blockchain);

#endif // BLOCKCHAIN_H
