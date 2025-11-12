/**
 * lenet_cnn_fixed.c — top-level forward + main()
 * - Float I/O, fixed-point interne (dans les kernels)
 * - Compatible avec softmax.c (float,float)
 * - Affiche la progression en %
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "lenet_cnn_fixed.h"

// ====== Activation ======
#ifndef LRELU_SLOPE
#define LRELU_SLOPE 0.05f
#endif
static inline float lrelu(float x) {
#pragma HLS INLINE
    return (x >= 0.0f) ? x : (LRELU_SLOPE * x);
}

// ====== CNN core ======
void lenet_cnn(
    float  input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    float  conv1_kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    float  conv1_bias[CONV1_NBOUTPUT],
    float  conv2_kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    float  conv2_bias[CONV2_NBOUTPUT],
    float  fc1_kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    float  fc1_bias[FC1_NBOUTPUT],
    float  fc2_kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
    float  fc2_bias[FC2_NBOUTPUT],
    float  output[FC2_NBOUTPUT]
) {
    static float conv1_out[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH];
    static float pool1_out[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH];
    static float conv2_out[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH];
    static float pool2_out[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
    static float fc1_out[FC1_NBOUTPUT];

    Conv1_28x28x1_5x5x20_1_0(input, conv1_kernel, conv1_bias, conv1_out);
    for (int c = 0; c < CONV1_NBOUTPUT; c++)
        for (int y = 0; y < CONV1_HEIGHT; y++)
            for (int x = 0; x < CONV1_WIDTH; x++)
                conv1_out[c][y][x] = lrelu(conv1_out[c][y][x]);

    Pool1_24x24x20_2x2x20_2_0(conv1_out, pool1_out);

    Conv2_12x12x20_5x5x40_1_0(pool1_out, conv2_kernel, conv2_bias, conv2_out);
    for (int c = 0; c < CONV2_NBOUTPUT; c++)
        for (int y = 0; y < CONV2_HEIGHT; y++)
            for (int x = 0; x < CONV2_WIDTH; x++)
                conv2_out[c][y][x] = lrelu(conv2_out[c][y][x]);

    Pool2_8x8x40_2x2x40_2_0(conv2_out, pool2_out);

    Fc1_40_400(pool2_out, fc1_kernel, fc1_bias, fc1_out);
    for (int i = 0; i < FC1_NBOUTPUT; i++)
        fc1_out[i] = lrelu(fc1_out[i]);

    Fc2_400_10(fc1_out, fc2_kernel, fc2_bias, output);
}

// ====== Globals ======
unsigned char  REF_IMG[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
float          INPUT_NORM[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
float          CONV1_KERNEL[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM];
float          CONV1_BIAS[CONV1_NBOUTPUT];
float          CONV2_KERNEL[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM];
float          CONV2_BIAS[CONV2_NBOUTPUT];
float          FC1_KERNEL[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
float          FC1_BIAS[FC1_NBOUTPUT];
float          FC2_KERNEL[FC2_NBOUTPUT][FC1_NBOUTPUT];
float          FC2_BIAS[FC2_NBOUTPUT];
float          FC2_OUTPUT[FC2_NBOUTPUT];
float          SOFTMAX_OUTPUT[FC2_NBOUTPUT];

// ====== Main ======
int main(void) {
    char *hdf5_filename        = "../FLOAT/lenet_weights.weights.h5";
    char *conv1_weights_path   = "/layers/conv2d/vars/0";
    char *conv1_bias_path      = "/layers/conv2d/vars/1";
    char *conv2_weights_path   = "/layers/conv2d_1/vars/0";
    char *conv2_bias_path      = "/layers/conv2d_1/vars/1";
    char *fc1_weights_path     = "/layers/dense/vars/0";
    char *fc1_bias_path        = "/layers/dense/vars/1";
    char *fc2_weights_path     = "/layers/dense_1/vars/0";
    char *fc2_bias_path        = "/layers/dense_1/vars/1";
    char *test_labels_filename = "../FLOAT/mnist/t10k-labels-idx1-ubyte";

    FILE *label_file;
    int ret;
    unsigned char label, number;
    unsigned int error = 0;
    unsigned char labels_legend[10] = {0,1,2,3,4,5,6,7,8,9};
    char img_filename[120], img_count[10];
    float maxv;
    struct timeval start, end;
    double tdiff;
    int m = 0;

    printf("\e[1;1H\e[2J");
    printf("\nReading weights...\n");
    ReadConv1Weights(hdf5_filename, conv1_weights_path, CONV1_KERNEL);
    ReadConv1Bias   (hdf5_filename, conv1_bias_path,   CONV1_BIAS);
    ReadConv2Weights(hdf5_filename, conv2_weights_path, CONV2_KERNEL);
    ReadConv2Bias   (hdf5_filename, conv2_bias_path,    CONV2_BIAS);
    ReadFc1Weights  (hdf5_filename, fc1_weights_path,   FC1_KERNEL);
    ReadFc1Bias     (hdf5_filename, fc1_bias_path,      FC1_BIAS);
    ReadFc2Weights  (hdf5_filename, fc2_weights_path,   FC2_KERNEL);
    ReadFc2Bias     (hdf5_filename, fc2_bias_path,      FC2_BIAS);

    label_file = fopen(test_labels_filename, "rb");
    if (!label_file) {
        printf("Error: Unable to open file %s.\n", test_labels_filename);
        return 1;
    }
    for (int k = 0; k < 8; k++) (void)fgetc(label_file); // skip header

    printf("\nProcessing test set...\n");
    gettimeofday(&start, NULL);

    while (1) {
        int lab = fgetc(label_file);
        if (lab == EOF) break;
        label = (unsigned char)lab;

        strcpy(img_filename, "../FLOAT/mnist/t10k-images-idx3-ubyte[");
        sprintf(img_count, "%d", m);
        if      (m < 10)    strcat(img_filename, "0000");
        else if (m < 100)   strcat(img_filename, "000");
        else if (m < 1000)  strcat(img_filename, "00");
        else if (m < 10000) strcat(img_filename, "0");
        strcat(img_filename, img_count);
        strcat(img_filename, "].pgm");

        ReadPgmFile(img_filename, (unsigned char *)REF_IMG);
        NormalizeImg((unsigned char *)REF_IMG, (float *)INPUT_NORM, IMG_WIDTH, IMG_WIDTH);

        lenet_cnn(INPUT_NORM,
                  CONV1_KERNEL, CONV1_BIAS,
                  CONV2_KERNEL, CONV2_BIAS,
                  FC1_KERNEL, FC1_BIAS,
                  FC2_KERNEL, FC2_BIAS,
                  FC2_OUTPUT);

        Softmax(FC2_OUTPUT, SOFTMAX_OUTPUT);

        maxv = 0.0f;
        number = 0;
        for (int k = 0; k < FC2_NBOUTPUT; k++) {
            if (SOFTMAX_OUTPUT[k] > maxv) {
                maxv = SOFTMAX_OUTPUT[k];
                number = (unsigned char)k;
            }
        }
        if (labels_legend[number] != label) error++;

        // progress
        if ((m % 100) == 0)
            printf("\rProgress: %d / 10000 (%.1f%%)", m, (100.0f * m) / 10000.0f);

        m++;
    }

    gettimeofday(&end, NULL);
    tdiff = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec)/1000000.0;

    printf("\rProgress: 100%%\n");
    printf("\nTOTAL TIME: %.3f s", tdiff);
    printf("\nErrors: %u / %d", error, m);
    printf("\nAccuracy: %.2f%%\n", (1.0f - ((float)error / (float)m)) * 100.0f);

    fclose(label_file);
    return 0;
}
