// conv.c — HLS-compliant convolution layers for LeNet
// Fixes: bias applied, bounds checks, optional SAME padding knob, stable loops
// Notes: keep prototypes identical to lenet_cnn_float.h

#include "lenet_cnn_float.h"

// knobs
#ifndef CONV1_UNROLL_C
#define CONV1_UNROLL_C IMG_DEPTH
#endif
#ifndef CONV2_UNROLL_C
#define CONV2_UNROLL_C CONV1_NBOUTPUT
#endif
// keep UNROLL_K at 1 unless you target higher area
#ifndef CONV1_UNROLL_K
#define CONV1_UNROLL_K 1
#endif
#ifndef CONV2_UNROLL_K
#define CONV2_UNROLL_K 1
#endif

// optional SAME padding (set to 0 to keep legacy behavior)
#ifndef CONV1_SAME
#define CONV1_SAME 0
#endif
#ifndef CONV2_SAME
#define CONV2_SAME 0
#endif

#if (CONV1_UNROLL_K > 1)
#define PRAGMA_UNROLL_K1 _Pragma("HLS UNROLL")
#else
#define PRAGMA_UNROLL_K1
#endif
#if (CONV2_UNROLL_K > 1)
#define PRAGMA_UNROLL_K2 _Pragma("HLS UNROLL")
#else
#define PRAGMA_UNROLL_K2
#endif

static inline int clampi(int v, int lo, int hi){
#pragma HLS INLINE
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

void Conv1_28x28x1_5x5x20_1_0(
    float input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    float kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    float bias[CONV1_NBOUTPUT],
    float output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]
){
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=2
#if (CONV1_UNROLL_K > 1)
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=3
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=4
#endif

    const int pad = CONV1_SAME ? (CONV1_DIM - 1) / 2 : CONV1_PAD;

    for (int m = 0; m < CONV1_NBOUTPUT; m++){
        for (int y = 0; y < CONV1_HEIGHT; y++){
            for (int x = 0; x < CONV1_WIDTH; x++){
#pragma HLS PIPELINE II=1
                float acc = bias[m];

                for (int c = 0; c < IMG_DEPTH; c++){
#if (CONV1_UNROLL_C > 1)
#pragma HLS UNROLL
#endif
                    for (int ky = 0; ky < CONV1_DIM; ky++){
                        PRAGMA_UNROLL_K1;
                        const int in_y = y * CONV1_STRIDE + ky - pad;
                        if ((in_y < 0) || (in_y >= IMG_HEIGHT)) continue;
                        for (int kx = 0; kx < CONV1_DIM; kx++){
                            PRAGMA_UNROLL_K1;
                            const int in_x = x * CONV1_STRIDE + kx - pad;
                            if ((in_x < 0) || (in_x >= IMG_WIDTH)) continue;
                            acc += input[c][in_y][in_x] * kernel[m][c][ky][kx];
                        }
                    }
                }
                output[m][y][x] = acc; // activation is handled outside
            }
        }
    }
}

void Conv2_12x12x20_5x5x40_1_0(
    float input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
    float kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    float bias[CONV2_NBOUTPUT],
    float output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]
){
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=2
#if (CONV2_UNROLL_K > 1)
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=3
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=4
#endif

    const int pad = CONV2_SAME ? (CONV2_DIM - 1) / 2 : CONV2_PAD;

    for (int m = 0; m < CONV2_NBOUTPUT; m++){
        for (int y = 0; y < CONV2_HEIGHT; y++){
            for (int x = 0; x < CONV2_WIDTH; x++){
#pragma HLS PIPELINE II=1
                float acc = bias[m];

                for (int c = 0; c < POOL1_NBOUTPUT; c++){
#if (CONV2_UNROLL_C > 1)
#pragma HLS UNROLL
#endif
                    for (int ky = 0; ky < CONV2_DIM; ky++){
                        PRAGMA_UNROLL_K2;
                        const int in_y = y * CONV2_STRIDE + ky - pad;
                        if ((in_y < 0) || (in_y >= POOL1_HEIGHT)) continue;
                        for (int kx = 0; kx < CONV2_DIM; kx++){
                            PRAGMA_UNROLL_K2;
                            const int in_x = x * CONV2_STRIDE + kx - pad;
                            if ((in_x < 0) || (in_x >= POOL1_WIDTH)) continue;
                            acc += input[c][in_y][in_x] * kernel[m][c][ky][kx];
                        }
                    }
                }
                output[m][y][x] = acc;
            }
        }
    }
}
