#include <stdio.h>
#include <stdlib.h>

#include "gpu_parallel/blockchain.cuh"
#include "gpu_parallel/utilities.cuh"

int main(int argc, char *argv[]) {
    int difficulty = 4;
    int size = 0;
    int capacity = 10;
    int num_blocks = 5;

    KeyValue keyValues[] = {
        {"difficulty", &difficulty},
        {"size", &size},
        {"capacity", &capacity},
        {"num_blocks", &num_blocks}
    };

    parse_arguments(argc, argv, keyValues, sizeof(keyValues) / sizeof(KeyValue));

    Blockchain blockchain;
    blockchain.size = size;
    blockchain.capacity = capacity;
    blockchain.blocks = (Block*)malloc(blockchain.capacity * sizeof(Block));
    Block genesis_block = create_genesis_block(difficulty);
    blockchain.blocks[blockchain.size] = genesis_block;
    blockchain.size++;

    for (int i = 1; i <= num_blocks; i++) {
        char data[50];
        sprintf(data, "Data of Block %%d", i);
        add_block(&blockchain, data);
    }

    print_blockchain(&blockchain);

    free(blockchain.blocks);

    return 0;
}
