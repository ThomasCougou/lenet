/**
  ******************************************************************************
  * @file    lenet_cnn_fixed.h
  * @brief   LeNet fixed-point version (HLS friendly)
  * @note    Same structure as lenet_cnn_float.h (float I/O, int8 core)
  ******************************************************************************
  */

#ifndef __LENET_CNN_FIXED_H__
#define __LENET_CNN_FIXED_H__

#include <stdint.h>

/* ===================== Layer dimensions ===================== */

// Image input
#define IMG_HEIGHT          28
#define IMG_WIDTH           28
#define IMG_DEPTH           1

// Conv1
#define CONV1_NBOUTPUT      20
#define CONV1_DIM           5
#define CONV1_STRIDE        1
#define CONV1_PAD           0
#define CONV1_HEIGHT        24
#define CONV1_WIDTH         24

// Pool1
#define POOL1_NBOUTPUT      CONV1_NBOUTPUT
#define POOL1_HEIGHT        12
#define POOL1_WIDTH         12

// Conv2
#define CONV2_NBOUTPUT      40
#define CONV2_DIM           5
#define CONV2_STRIDE        1
#define CONV2_PAD           0
#define CONV2_HEIGHT        8
#define CONV2_WIDTH         8

// Pool2
#define POOL2_NBOUTPUT      CONV2_NBOUTPUT
#define POOL2_HEIGHT        4
#define POOL2_WIDTH         4

// FC layers
#define FC1_NBOUTPUT        400
#define FC2_NBOUTPUT        10

// Softmax
#define SM_INPUT_LEN        FC2_NBOUTPUT
#define SM_OUTPUT_LEN       FC2_NBOUTPUT


/* ===================== Function prototypes ===================== */

// ---- Convolution layers ----
void Conv1_28x28x1_5x5x20_1_0(
    float input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    float kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    float bias[CONV1_NBOUTPUT],
    float output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]
);

void Conv2_12x12x20_5x5x40_1_0(
    float input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
    float kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    float bias[CONV2_NBOUTPUT],
    float output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]
);


// ---- Pooling layers ----
void Pool1_24x24x20_2x2x20_2_0(
    float input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
    float output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]
);

void Pool2_8x8x40_2x2x40_2_0(
    float input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
    float output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]
);


// ---- Fully Connected layers ----
void Fc1_40_400(
    float input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    float weight[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    float bias[FC1_NBOUTPUT],
    float output[FC1_NBOUTPUT]
);

void Fc2_400_10(
    float input[FC1_NBOUTPUT],
    float weight[FC2_NBOUTPUT][FC1_NBOUTPUT],
    float bias[FC2_NBOUTPUT],
    float output[FC2_NBOUTPUT]
);


// ---- Softmax ----
void Softmax_10(float input[SM_INPUT_LEN], float output[SM_OUTPUT_LEN], int *pred_class);
void Softmax(float input[SM_INPUT_LEN], float output[SM_OUTPUT_LEN]);


// ---- Top-level network ----
void lenet_cnn_q(
    float input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    float conv1_kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    float conv1_bias[CONV1_NBOUTPUT],
    float conv2_kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    float conv2_bias[CONV2_NBOUTPUT],
    float fc1_kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    float fc1_bias[FC1_NBOUTPUT],
    float fc2_kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
    float fc2_bias[FC2_NBOUTPUT],
    float output[FC2_NBOUTPUT]
);

#endif // __LENET_CNN_FIXED_H__
