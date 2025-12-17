CC = gcc
CFLAGS = -O3 -Wall -I.
LDFLAGS = -lm

OBJS = lenet_cnn_fixed.o conv_fixed.o pool_fixed.o fc_fixed.o utils.o softmax.o

all: lenet_cnn_fixed

lenet_cnn_fixed: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

lenet_cnn_fixed.o: lenet_cnn_fixed.c lenet_cnn_fixed.h weights.h fixed_Point.h
	$(CC) $(CFLAGS) -c $< -o $@

conv_fixed.o: conv_fixed.c lenet_cnn_fixed.h
	$(CC) $(CFLAGS) -c $< -o $@

pool_fixed.o: pool_fixed.c lenet_cnn_fixed.h
	$(CC) $(CFLAGS) -c $< -o $@

fc_fixed.o: fc_fixed.c lenet_cnn_fixed.h
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: utils.c lenet_cnn_fixed.h weights.h fixed_Point.h
	$(CC) $(CFLAGS) -c $< -o $@

softmax.o: softmax.c lenet_cnn_fixed.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) lenet_cnn_fixed
