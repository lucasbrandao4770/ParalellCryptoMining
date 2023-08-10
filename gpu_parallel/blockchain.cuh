#ifndef BLOCKCHAIN_CUH
#define BLOCKCHAIN_CUH

#define MAX_DATA_SIZE 100
#define MAX_HASH_SIZE 65
#define NUM_THREADS 256

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

__global__ void calculate_hash_kernel(int index, time_t timestamp, char* data, char* previous_hash, int difficulty, char* result, int* found_nonce);
void calculate_hash(Block* block);
Block create_genesis_block(int difficulty);
void add_block(Blockchain* blockchain, const char* data);
void print_block(const Block* block);
void print_blockchain(const Blockchain* blockchain);

#endif   // BLOCKCHAIN_CUH
