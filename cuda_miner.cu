#include <stdio.h>
#include <stdlib.h>

#include "gpu_parallel/blockchain.cuh"
#include "gpu_parallel/utilities.cuh"

int main(int argc, char *argv[]) {
    /**< Main function for the sequential Bitcoin miner. Initializes blockchain and starts mining. */
    int difficulty = 4; /**< Default mining difficulty level. */
    int size = 0; /**< Default size of the blockchain (number of blocks). */
    int capacity = 10; /**< Default capacity allocated for blocks in the blockchain. */
    int num_blocks = 5; /**< Default number of blocks to be added to the blockchain. */

    KeyValue keyValues[] = {
        /**< Mapping command-line arguments to their corresponding variables. */
        {"difficulty", &difficulty},
        {"size", &size},
        {"capacity", &capacity},
        {"num_blocks", &num_blocks}
    };

    parse_arguments(argc, argv, keyValues, sizeof(keyValues) / sizeof(KeyValue));
    /**< Parse command-line arguments to set difficulty, size, and capacity of the blockchain. */

    Blockchain blockchain;
    blockchain.size = size;
    blockchain.capacity = capacity;
    blockchain.blocks = (Block*)malloc(blockchain.capacity * sizeof(Block));
    /**< Allocate memory for the blocks in the blockchain based on the specified capacity. */
    Block genesis_block = create_genesis_block(difficulty);
    blockchain.blocks[blockchain.size] = genesis_block;
    blockchain.size++;
    /**< Create the genesis block and add it to the blockchain. */

    for (int i = 1; i <= num_blocks; i++) {
        char data[50];
        sprintf(data, "Data of Block %%d", i);
        add_block(&blockchain, data);
    }
    /**< Add blocks to the blockchain with predefined data using a loop based on "num_blocks". */

    print_blockchain(&blockchain);
    /**< Print the details of the entire blockchain, showing the chain of blocks. */

    free(blockchain.blocks);
    /**< Free the allocated memory for the blocks in the blockchain. */

    return 0;
}
