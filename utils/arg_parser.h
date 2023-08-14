#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <stddef.h>

/**
 * @struct KeyValue
 * @brief Structure representing a key-value pair.
 *
 * The key is a string representing the argument name, and the value is a pointer to an integer to store the value of the argument.
 */
typedef struct {
    const char *key; /**< The key or name of the argument. */
    int *value;      /**< Pointer to the integer value of the argument. */
} KeyValue;

/**
 * @brief Parses the command-line arguments and populates the key-value pairs.
 *
 * The function iterates through the command-line arguments, looks for keys and values separated by an equal sign ('='), and populates the provided key-value pairs.
 * If a match is found between the provided key and the command-line key, the corresponding value is updated.
 *
 * @param argc The count of command-line arguments.
 * @param argv Array of pointers to the command-line arguments.
 * @param keyValues Pointer to an array of key-value pairs to be populated.
 * @param keyValueCount The number of key-value pairs in the array.
 */
void parse_arguments(int argc, char *argv[], KeyValue *keyValues, size_t keyValueCount);

#endif // ARG_PARSER_H
