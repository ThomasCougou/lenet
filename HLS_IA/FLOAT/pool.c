// pool.c — MaxPool 2x2 stride 2 (fixed-point core, float I/O)
#include "lenet_cnn_float.h"
#include <stdint.h>
#include <float.h>
#include <math.h>

static inline int8_t clamp_i8(int v){
#pragma HLS INLINE
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

static inline float maxabs_f(const float *p, int n){
#pragma HLS INLINE
    float m = 0.0f;
    for (int i=0;i<n;i++){
        float a = p[i]; if (a < 0) a = -a;
        if (a > m) m = a;
    }
    if (m < 1e-8f) m = 1e-8f;
    return m;
}

// Pool1: input [20][24][24] -> output [20][12][12]
void Pool1_24x24x20_2x2x20_2_0(
    float input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
    float output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]
){
#pragma HLS INLINE off

    // per-tensor scale (positive, so max is order-preserving)
    const int n_in = CONV1_NBOUTPUT*CONV1_HEIGHT*CONV1_WIDTH;
    const float sx = maxabs_f(&input[0][0][0], n_in) / 127.0f;
    const float inv_sx = 1.0f / sx;

    static int8_t in_q[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH];
#pragma HLS ARRAY_PARTITION variable=in_q complete dim=1

    // quantize input to int8
    for (int c = 0; c < CONV1_NBOUTPUT; c++){
        for (int y = 0; y < CONV1_HEIGHT; y++){
            for (int x = 0; x < CONV1_WIDTH; x++){
#pragma HLS PIPELINE II=1
                int v = (int)lrintf(input[c][y][x] * inv_sx);
                in_q[c][y][x] = clamp_i8(v);
            }
        }
    }

    // max-pool in int8, then dequantize to float
    for (int c = 0; c < CONV1_NBOUTPUT; c++){
        for (int y = 0; y < POOL1_HEIGHT; y++){
            for (int x = 0; x < POOL1_WIDTH; x++){
#pragma HLS PIPELINE II=1
                const int y0 = y * 2;
                const int x0 = x * 2;
                int m = -128;
                for (int ky = 0; ky < 2; ky++){
#pragma HLS UNROLL
                    for (int kx = 0; kx < 2; kx++){
#pragma HLS UNROLL
                        const int iy = y0 + ky;
                        const int ix = x0 + kx;
                        if (iy < CONV1_HEIGHT && ix < CONV1_WIDTH){
                            int v = (int)in_q[c][iy][ix];
                            if (v > m) m = v;
                        }
                    }
                }
                output[c][y][x] = (float)((int8_t)m) * sx;
            }
        }
    }
}

// Pool2: input [40][8][8] -> output [40][4][4]
void Pool2_8x8x40_2x2x40_2_0(
    float input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
    float output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]
){
#pragma HLS INLINE off

    const int n_in = CONV2_NBOUTPUT*CONV2_HEIGHT*CONV2_WIDTH;
    const float sx = maxabs_f(&input[0][0][0], n_in) / 127.0f;
    const float inv_sx = 1.0f / sx;

    static int8_t in_q[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH];
#pragma HLS ARRAY_PARTITION variable=in_q complete dim=1

    // quantize input to int8
    for (int c = 0; c < CONV2_NBOUTPUT; c++){
        for (int y = 0; y < CONV2_HEIGHT; y++){
            for (int x = 0; x < CONV2_WIDTH; x++){
#pragma HLS PIPELINE II=1
                int v = (int)lrintf(input[c][y][x] * inv_sx);
                in_q[c][y][x] = clamp_i8(v);
            }
        }
    }

    // max-pool in int8, then dequantize to float
    for (int c = 0; c < CONV2_NBOUTPUT; c++){
        for (int y = 0; y < POOL2_HEIGHT; y++){
            for (int x = 0; x < POOL2_WIDTH; x++){
#pragma HLS PIPELINE II=1
                const int y0 = y * 2;
                const int x0 = x * 2;
                int m = -128;
                for (int ky = 0; ky < 2; ky++){
#pragma HLS UNROLL
                    for (int kx = 0; kx < 2; kx++){
#pragma HLS UNROLL
                        const int iy = y0 + ky;
                        const int ix = x0 + kx;
                        if (iy < CONV2_HEIGHT && ix < CONV2_WIDTH){
                            int v = (int)in_q[c][iy][ix];
                            if (v > m) m = v;
                        }
                    }
                }
                output[c][y][x] = (float)((int8_t)m) * sx;
            }
        }
    }
}
