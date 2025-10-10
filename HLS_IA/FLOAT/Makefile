CC = gcc
CFLAGS = -I/usr/include/hdf5/serial/ -O3
LDFLAGS = -lhdf5_serial -lm

OBJS = lenet_cnn_float.o conv.o fc.o pool.o utils.o softmax.o

all: lenet_cnn_float

lenet_cnn_float: $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)

lenet_cnn_float.o: lenet_cnn_float.c lenet_cnn_float.h
	$(CC) $(CFLAGS) -c $< -o $@

conv.o: conv.c lenet_cnn_float.h
	$(CC) $(CFLAGS) -c $< -o $@

fc.o: fc.c lenet_cnn_float.h
	$(CC) $(CFLAGS) -c $< -o $@

pool.o: pool.c lenet_cnn_float.h
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: utils.c lenet_cnn_float.h
	$(CC) $(CFLAGS) -c $< -o $@

softmax.o: softmax.c lenet_cnn_float.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) lenet_cnn_float
