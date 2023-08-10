#ifndef BLOCK_H
#define BLOCK_H

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

void calculate_hash(Block* block);
Block create_genesis_block(int difficulty);
void print_block(const Block* block);

#endif // BLOCK_H
