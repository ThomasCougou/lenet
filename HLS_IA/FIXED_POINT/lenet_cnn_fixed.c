/**
 * lenet_cnn_fixed.c — Top-level forward for LeNet (fixed-point cores)
 * - Input float image -> MNIST z-score normalization
 * - Conv1 -> LeakyReLU -> Pool1 -> Conv2 -> LeakyReLU -> Pool2 -> Fc1 (ReLU inside) -> Fc2 (logits)
 * - Logits (float) -> fixed-point short -> Softmax(short*, float*)
 */

#include <math.h>
#include <stddef.h>
#include "lenet_cnn_fixed.h"

// ==== activation: Leaky-ReLU ====
#ifndef LRELU_SLOPE
#define LRELU_SLOPE 0.05f
#endif
static inline float lrelu(float x){ return (x >= 0.0f) ? x : (LRELU_SLOPE * x); }

// ==== softmax prototype (fixed-point input) ====
#ifndef FIXED_POINT
#define FIXED_POINT 7  // Q7. change if your project uses another frac bits count
#endif
void Softmax(short vector_in[FC2_NBOUTPUT], float vector_out[FC2_NBOUTPUT]);

// ==== main forward (no I/O here) ====
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
    float output[FC2_NBOUTPUT]               // softmax probabilities
){
    // -------- 0) input normalization (MNIST mean/std) --------
    // expects input in [0,1]; if you pass 0..255, divide first in your loader.
    static float in_norm[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
    for (int c = 0; c < IMG_DEPTH; c++){
        for (int y = 0; y < IMG_HEIGHT; y++){
            for (int x = 0; x < IMG_WIDTH; x++){
                float v01 = input[c][y][x];
                float v = (v01 - 0.1307f) / 0.3081f;
                in_norm[c][y][x] = v;
            }
        }
    }

    // -------- 1) Conv1 -> LeakyReLU -> Pool1 --------
    static float conv1_out[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH];
    static float pool1_out[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH];

    Conv1_28x28x1_5x5x20_1_0(in_norm, conv1_kernel, conv1_bias, conv1_out);

    for (int c = 0; c < CONV1_NBOUTPUT; c++)
        for (int y = 0; y < CONV1_HEIGHT; y++)
            for (int x = 0; x < CONV1_WIDTH; x++)
                conv1_out[c][y][x] = lrelu(conv1_out[c][y][x]);

    Pool1_24x24x20_2x2x20_2_0(conv1_out, pool1_out);

    // -------- 2) Conv2 -> LeakyReLU -> Pool2 --------
    static float conv2_out[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH];
    static float pool2_out[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];

    Conv2_12x12x20_5x5x40_1_0(pool1_out, conv2_kernel, conv2_bias, conv2_out);

    for (int c = 0; c < CONV2_NBOUTPUT; c++)
        for (int y = 0; y < CONV2_HEIGHT; y++)
            for (int x = 0; x < CONV2_WIDTH; x++)
                conv2_out[c][y][x] = lrelu(conv2_out[c][y][x]);

    Pool2_8x8x40_2x2x40_2_0(conv2_out, pool2_out);

    // -------- 3) Fc1 (ReLU inside in your fixed core) --------
    static float fc1_out[FC1_NBOUTPUT];
    Fc1_40_400(pool2_out, fc1_kernel, fc1_bias, fc1_out);

    // -------- 4) Fc2 (logits) --------
    static float logits[FC2_NBOUTPUT];
    Fc2_400_10(fc1_out, fc2_kernel, fc2_bias, logits);

    // -------- 5) logits(float) -> fixed-point(short) -> Softmax --------
    // vector_in[k] = round( logits[k] * (1<<FIXED_POINT) )
    short vec_in[FC2_NBOUTPUT];
    for (int k = 0; k < FC2_NBOUTPUT; k++){
        float scaled = logits[k] * (float)(1 << FIXED_POINT);
        int v = (int)(scaled >= 0.0f ? (scaled + 0.5f) : (scaled - 0.5f));
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        vec_in[k] = (short)v;
    }

    Softmax(vec_in, output);
}
