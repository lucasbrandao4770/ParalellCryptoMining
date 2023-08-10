#include <stdlib.h>

#include "sequential/blockchain.h"
#include "utils/arg_parser.h"

int main(int argc, char *argv[]) {
    int difficulty = 4;
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
    blockchain.blocks = (Block*)malloc(blockchain.capacity * sizeof(Block));
    Block genesis_block = create_genesis_block(difficulty);
    blockchain.blocks[blockchain.size] = genesis_block;
    blockchain.size++;

    add_block(&blockchain, "Data of Block 1");
    add_block(&blockchain, "Data of Block 2");
    add_block(&blockchain, "Data of Block 3");
    add_block(&blockchain, "Data of Block 4");
    add_block(&blockchain, "Data of Block 5");

    print_blockchain(&blockchain);

    free(blockchain.blocks);

    return 0;
}
