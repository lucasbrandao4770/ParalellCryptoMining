#include "arg_parser.h"
#include <string.h>
#include <stdlib.h>

void parse_arguments(int argc, char *argv[], KeyValue *keyValues, size_t keyValueCount) {
    for (int i = 1; i < argc; i++) {
        char *equalSign = strchr(argv[i], '='); /**< Pointer to the location of the equal sign in the argument. */
        if (equalSign != NULL) {
            *equalSign = '\0'; /**< Replace '=' with null terminator to split key and value */
            const char *key = argv[i]; /**< Key extracted from the argument. */
            const char *value = equalSign + 1; /**< Value extracted from the argument. */

            /**< Iterate through the key-value pairs and update the matching key's value */
            for (size_t j = 0; j < keyValueCount; j++) {
                if (strcmp(key, keyValues[j].key) == 0) {
                    *keyValues[j].value = atoi(value); /**< Convert the value to an integer and update */
                    break;
                }
            }
        }
    }
}
