#include "arg_parser.h"
#include <string.h>
#include <stdlib.h>

void parse_arguments(int argc, char *argv[], KeyValue *keyValues, size_t keyValueCount) {
    for (int i = 1; i < argc; i++) {
        char *equalSign = strchr(argv[i], '=');
        if (equalSign != NULL) {
            *equalSign = '\0'; // Replace '=' with null terminator to split key and value
            const char *key = argv[i];
            const char *value = equalSign + 1;

            for (size_t j = 0; j < keyValueCount; j++) {
                if (strcmp(key, keyValues[j].key) == 0) {
                    *keyValues[j].value = atoi(value);
                    break;
                }
            }
        }
    }
}
