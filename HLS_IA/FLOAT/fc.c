// fc.c - Reference (baseline) implementations only
// Fully connected layers for LeNet

#include "lenet_cnn_float.h"

// Fc1: input = POOL2_OUT (40 maps 4x4), output = 400 neurons
void Fc1_40_400(
    float input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    float kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    float bias[FC1_NBOUTPUT],
    float output[FC1_NBOUTPUT]
){
    for (int o = 0; o < FC1_NBOUTPUT; ++o){
        float acc = bias[o];
        for (int c = 0; c < POOL2_NBOUTPUT; ++c){
            for (int y = 0; y < POOL2_HEIGHT; ++y){
                for (int x = 0; x < POOL2_WIDTH; ++x){
                    acc += kernel[o][c][y][x] * input[c][y][x];
                }
            }
        }
        output[o] = acc;
    }
}

// Fc2: input = 400, output = 10 classes
void Fc2_400_10(
    float input[FC1_NBOUTPUT],
    float kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
    float bias[FC2_NBOUTPUT],
    float output[FC2_NBOUTPUT]
){
    for (int o = 0; o < FC2_NBOUTPUT; ++o){
        float acc = bias[o];
        for (int i = 0; i < FC1_NBOUTPUT; ++i){
            acc += kernel[o][i] * input[i];
        }
        output[o] = acc;
    }
}
