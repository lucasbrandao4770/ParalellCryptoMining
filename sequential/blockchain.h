#ifndef BLOCKCHAIN_H
#define BLOCKCHAIN_H

typedef struct {
    Block* blocks;
    int size;
    int capacity;
} Blockchain;

void add_block(Blockchain* blockchain, const char* data);
void print_blockchain(const Blockchain* blockchain);

#endif // BLOCKCHAIN_H
