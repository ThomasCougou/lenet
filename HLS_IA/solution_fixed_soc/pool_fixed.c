// pool_fixed.c — MaxPool 2x2 stride 2 with uniform activation scales
#include "lenet_cnn_fixed.h"
#include <stdint.h>

#define SY_CONV1  (1.0f/24.0f)
#define SY_CONV2  (1.0f/24.0f)

static inline int8_t clamp_i8(int v){
#pragma HLS INLINE
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}
static inline int qroundf(float x){
#pragma HLS INLINE
    return (int)(x + (x >= 0.0f ? 0.5f : -0.5f));
}

void Pool1_24x24x20_2x2x20_2_0(
    float input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
    float output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]
){
#pragma HLS INLINE off
    static int8_t in_q[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH];
#pragma HLS ARRAY_PARTITION variable=in_q complete dim=1

    // quantize in with SX = SY_CONV1
    {
        float inv_sx = 1.0f / SY_CONV1;
        for (int c = 0; c < CONV1_NBOUTPUT; c++)
            for (int y = 0; y < CONV1_HEIGHT; y++)
                for (int x = 0; x < CONV1_WIDTH; x++){
#pragma HLS PIPELINE II=1
                    int v = qroundf(input[c][y][x] * inv_sx);
                    in_q[c][y][x] = clamp_i8(v);
                }
    }

    for (int c = 0; c < CONV1_NBOUTPUT; c++){
        for (int y = 0; y < POOL1_HEIGHT; y++){
            for (int x = 0; x < POOL1_WIDTH; x++){
#pragma HLS PIPELINE II=1
                int y0 = y*2, x0 = x*2;
                int m = -128;
                for (int ky=0; ky<2; ky++){
#pragma HLS UNROLL
                    for (int kx=0; kx<2; kx++){
#pragma HLS UNROLL
                        int iy = y0 + ky, ix = x0 + kx;
                        int val = (int)in_q[c][iy][ix];
                        if (val > m) m = val;
                    }
                }
                output[c][y][x] = (float)((int8_t)m) * SY_CONV1;
            }
        }
    }
}

void Pool2_8x8x40_2x2x40_2_0(
    float input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
    float output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]
){
#pragma HLS INLINE off
    static int8_t in_q[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH];
#pragma HLS ARRAY_PARTITION variable=in_q complete dim=1

    {
        float inv_sx = 1.0f / SY_CONV2;
        for (int c = 0; c < CONV2_NBOUTPUT; c++)
            for (int y = 0; y < CONV2_HEIGHT; y++)
                for (int x = 0; x < CONV2_WIDTH; x++){
#pragma HLS PIPELINE II=1
                    int v = qroundf(input[c][y][x] * inv_sx);
                    in_q[c][y][x] = clamp_i8(v);
                }
    }

    for (int c = 0; c < CONV2_NBOUTPUT; c++){
        for (int y = 0; y < POOL2_HEIGHT; y++){
            for (int x = 0; x < POOL2_WIDTH; x++){
#pragma HLS PIPELINE II=1
                int y0 = y*2, x0 = x*2;
                int m = -128;
                for (int ky=0; ky<2; ky++){
#pragma HLS UNROLL
                    for (int kx=0; kx<2; kx++){
#pragma HLS UNROLL
                        int iy = y0 + ky, ix = x0 + kx;
                        int val = (int)in_q[c][iy][ix];
                        if (val > m) m = val;
                    }
                }
                output[c][y][x] = (float)((int8_t)m) * SY_CONV2;
            }
        }
    }
}
