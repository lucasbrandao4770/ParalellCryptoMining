CC = gcc
CFLAGS = -Wall -O2
TARGET = sequential_miner
SOURCES = ../sequential_miner.c ../sequential/block.c blockchain.c sha256.c ../utils/arg_parser.c
INCLUDES = -I../utils
OBJECTS = $(SOURCES:.c=.o)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJECTS) -o $(TARGET)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	-del /f /q ..\sequential_miner.o block.o blockchain.o sha256.o sequential_miner.exe ..\utils\arg_parser.o
