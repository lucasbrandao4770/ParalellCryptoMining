#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <stddef.h>

typedef struct {
    const char *key;
    int *value;
} KeyValue;

void parse_arguments(int argc, char *argv[], KeyValue *keyValues, size_t keyValueCount);

#endif // ARG_PARSER_H
