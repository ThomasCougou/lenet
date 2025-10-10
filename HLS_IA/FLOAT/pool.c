// pool.c — MaxPool 2x2 stride 2
#include "lenet_cnn_float.h"
#include <float.h>

void Pool1_24x24x20_2x2x20_2_0(
    float input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
    float output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]
){
#pragma HLS INLINE off
    for (int c = 0; c < CONV1_NBOUTPUT; c++){
        for (int y = 0; y < POOL1_HEIGHT; y++){
            for (int x = 0; x < POOL1_WIDTH; x++){
#pragma HLS PIPELINE II=1
                float m = -FLT_MAX;
                const int y0 = y * 2;
                const int x0 = x * 2;
                for (int ky = 0; ky < 2; ky++){
#pragma HLS UNROLL
                    for (int kx = 0; kx < 2; kx++){
#pragma HLS UNROLL
                        const int iy = y0 + ky;
                        const int ix = x0 + kx;
                        if (iy < CONV1_HEIGHT && ix < CONV1_WIDTH){
                            float v = input[c][iy][ix];
                            m = (v > m) ? v : m;
                        }
                    }
                }
                output[c][y][x] = m;
            }
        }
    }
}

void Pool2_8x8x40_2x2x40_2_0(
    float input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
    float output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]
){
#pragma HLS INLINE off
    for (int c = 0; c < CONV2_NBOUTPUT; c++){
        for (int y = 0; y < POOL2_HEIGHT; y++){
            for (int x = 0; x < POOL2_WIDTH; x++){
#pragma HLS PIPELINE II=1
                float m = -FLT_MAX;
                const int y0 = y * 2;
                const int x0 = x * 2;
                for (int ky = 0; ky < 2; ky++){
#pragma HLS UNROLL
                    for (int kx = 0; kx < 2; kx++){
#pragma HLS UNROLL
                        const int iy = y0 + ky;
                        const int ix = x0 + kx;
                        if (iy < CONV2_HEIGHT && ix < CONV2_WIDTH){
                            float v = input[c][iy][ix];
                            m = (v > m) ? v : m;
                        }
                    }
                }
                output[c][y][x] = m;
            }
        }
    }
}
