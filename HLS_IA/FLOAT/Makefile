CC = gcc
CFLAGS = -I/usr/include/hdf5/serial/ -O3
LDFLAGS = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_serial -lm

OBJS = lenet_cnn_float.o fc.o conv.o pooling.o utils.o softmax.o

all: lenet_cnn_float

lenet_cnn_float: $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)

lenet_cnn_float.o: lenet_cnn_float.c lenet_cnn_float.h
	$(CC) $(CFLAGS) -c $< -o $@

fc.o: fc.c lenet_cnn_float.h
	$(CC) $(CFLAGS) -c $< -o $@

pooling.o: pooling.c pooling.h lenet_cnn_float.h
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: utils.c lenet_cnn_float.h
	$(CC) $(CFLAGS) -c $< -o $@

softmax.o: softmax.c lenet_cnn_float.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) lenet_cnn_float

