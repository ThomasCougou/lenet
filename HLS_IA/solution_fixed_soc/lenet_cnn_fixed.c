/**
 * lenet_cnn_fixed.c — top-level forward + main()
 * - Float I/O, int8 core inside kernels (HLS friendly)
 * - Uses embedded weights from weights.h (no HDF5)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "lenet_cnn_fixed.h"
#include "weights.h"

/* Leaky-ReLU slope (must match conv/fc if you use it here too) */
#ifndef LRELU_SLOPE
#define LRELU_SLOPE 0.05f
#endif

static inline float lrelu(float x){
    return (x >= 0.0f) ? x : (LRELU_SLOPE * x);
}

/* ====== Forward wrapper using embedded weights ====== */
static void lenet_forward(
    float input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    float output[FC2_NBOUTPUT]
){
    static float conv1_out[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH];
    static float pool1_out[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH];
    static float conv2_out[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH];
    static float pool2_out[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
    static float fc1_out[FC1_NBOUTPUT];

    Conv1_28x28x1_5x5x20_1_0(input, CONV1_KERNEL, CONV1_BIAS, conv1_out);
    for (int c = 0; c < CONV1_NBOUTPUT; c++)
        for (int y = 0; y < CONV1_HEIGHT; y++)
            for (int x = 0; x < CONV1_WIDTH; x++)
                conv1_out[c][y][x] = lrelu(conv1_out[c][y][x]);

    Pool1_24x24x20_2x2x20_2_0(conv1_out, pool1_out);

    Conv2_12x12x20_5x5x40_1_0(pool1_out, CONV2_KERNEL, CONV2_BIAS, conv2_out);
    for (int c = 0; c < CONV2_NBOUTPUT; c++)
        for (int y = 0; y < CONV2_HEIGHT; y++)
            for (int x = 0; x < CONV2_WIDTH; x++)
                conv2_out[c][y][x] = lrelu(conv2_out[c][y][x]);

    Pool2_8x8x40_2x2x40_2_0(conv2_out, pool2_out);

    Fc1_40_400(pool2_out, FC1_KERNEL, FC1_BIAS, fc1_out);
    for (int i = 0; i < FC1_NBOUTPUT; i++)
        fc1_out[i] = lrelu(fc1_out[i]);

    Fc2_400_10(fc1_out, FC2_KERNEL, FC2_BIAS, output);
}

int main(void)
{
    /* MNIST test set files (same behavior as original code) */
    char mnist_test_labels_filename[] = "../FLOAT/mnist/t10k-images-idx3-ubyte";

    FILE *label_file = fopen(mnist_test_labels_filename, "rb");
    if (!label_file){
        printf("\nERROR: could not open label file %s\n", mnist_test_labels_filename);
        return 1;
    }

    /* Skip IDX header (8 bytes for labels) */
    for (int kk = 0; kk < 8; kk++) (void)fgetc(label_file);

    printf("\nUsing embedded weights from weights.h (no HDF5)\n");
    printf("Processing test set...\n");

    struct timeval start, end;
    gettimeofday(&start, NULL);

    unsigned int error = 0;
    int m = 0;

    unsigned char ref_img[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
    float input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
    float logits[FC2_NBOUTPUT];
    float prob[FC2_NBOUTPUT];

    while (m < 10000){
    	int lab = fgetc(label_file);
    	if (lab == EOF) break;

        unsigned char label = (unsigned char)lab;

        /* Original code reads samples/<index>.pgm */
        char img_filename[120];
	char img_count[10];

	sprintf(img_count, "%05d", m + 1);

	strcpy(img_filename, "../FLOAT/mnist/t10k-images-idx3-ubyte[");
	strcat(img_filename, img_count);
	strcat(img_filename, "].pgm");

        for (int y = 0; y < IMG_HEIGHT; y++)
            for (int x = 0; x < IMG_WIDTH; x++)
        NormalizeImg((unsigned char *)ref_img, (float *)input, IMG_WIDTH, IMG_WIDTH);

        lenet_forward(input, logits);

        Softmax(logits, prob);

        int best = 0;
        for (int i = 1; i < FC2_NBOUTPUT; i++)
            if (prob[i] > prob[best]) best = i;

        if ((unsigned char)best != label) error++;

        m++;
        if ((m % 100) == 0){
            printf("\rProgress: %d%%", (m * 100) / 10000);
            fflush(stdout);
        }
    }

    gettimeofday(&end, NULL);
    double tdiff = (double)(end.tv_sec - start.tv_sec)
                 + (double)(end.tv_usec - start.tv_usec) / 1000000.0;

    printf("\rProgress: 100%%\n");
    printf("\nTOTAL TIME: %.3f s", tdiff);
    printf("\nErrors: %u / %d", error, m);
    printf("\nAccuracy: %.2f%%\n", (1.0f - ((float)error / (float)m)) * 100.0f);

    fclose(label_file);
    return 0;
}
