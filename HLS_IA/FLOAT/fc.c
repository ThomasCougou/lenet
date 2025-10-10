// fc.c — Fully connected layers matching lenet_cnn_float.h
#include "lenet_cnn_float.h"

static inline float relu(float x){
#pragma HLS INLINE
    return (x > 0.0f) ? x : 0.0f;
}

// Fc1_40_400: input is 4x4x40 (POOL2 dims), output is 400
void Fc1_40_400(
    float input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],                               // IN  [40][4][4]
    float weight[400][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],                         // IN  [400][40][4][4]
    float bias[400],                                                                       // IN  [400]
    float output[400]                                                                      // OUT [400]
){
#pragma HLS INLINE off
    for (int o = 0; o < 400; o++){
        float acc = bias[o];
        for (int c = 0; c < POOL2_NBOUTPUT; c++){
#pragma HLS PIPELINE II=1
            for (int y = 0; y < POOL2_HEIGHT; y++){
#pragma HLS UNROLL
                for (int x = 0; x < POOL2_WIDTH; x++){
#pragma HLS UNROLL
                    acc += input[c][y][x] * weight[o][c][y][x];
                }
            }
        }
        output[o] = relu(acc);
    }
}

// Fc2_400_10: classic fully connected
void Fc2_400_10(
    float input[400],            // IN
    float weight[10][400],       // IN
    float bias[10],              // IN
    float output[10]             // OUT
){
#pragma HLS INLINE off
    for (int o = 0; o < 10; o++){
        float acc = bias[o];
        for (int i = 0; i < 400; i++){
#pragma HLS PIPELINE II=1
            acc += input[i] * weight[o][i];
        }
        output[o] = acc; // softmax applied later
    }
}
