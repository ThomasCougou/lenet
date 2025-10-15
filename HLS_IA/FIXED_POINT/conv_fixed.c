// conv.c — HLS-compliant convolution layers for LeNet (fixed-point core, float I/O)
// Strategy:
//   - Quantize float input/weights/bias to int8/int8/int32
//   - Compute MAC in int32
//   - Requantize to int8 (mul/shift), then dequantize to float for output
//   - Keep same prototypes as lenet_cnn_float.h (do not break other files)

#include "lenet_cnn_float.h"
#include <stdint.h>
#include <math.h>

// knobs
#ifndef CONV1_UNROLL_C
#define CONV1_UNROLL_C IMG_DEPTH
#endif
#ifndef CONV2_UNROLL_C
#define CONV2_UNROLL_C CONV1_NBOUTPUT
#endif
#ifndef CONV1_UNROLL_K
#define CONV1_UNROLL_K 1
#endif
#ifndef CONV2_UNROLL_K
#define CONV2_UNROLL_K 1
#endif

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

// -------- Fixed-point helpers (local to this TU) --------

static inline int8_t clamp_i8(int v){
#pragma HLS INLINE
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

static inline int32_t mul_shift_round(int32_t x, int32_t mul, int shift){
#pragma HLS INLINE
    long long t = (long long)x * (long long)mul;
    long long add = (t >= 0) ? (1LL<<(shift-1)) : -(1LL<<(shift-1));
    return (int32_t)((t + add) >> shift);
}

// choose mul/shift so that scale ~= mul / 2^shift
static inline void choose_mul_shift(float M, int32_t *mul, int *shift){
#pragma HLS INLINE
    // fixed shift = 24 is a good tradeoff for precision
    const int s = 24;
    long long m = (long long)llroundf(M * (float)(1<<s));
    if (m == 0) { *mul = 0; *shift = 0; return; }
    *mul = (int32_t)m; *shift = s;
}

static inline float maxabs_f(const float *p, int n){
#pragma HLS INLINE
    float m = 0.0f;
    for (int i=0;i<n;i++){
        float a = p[i]; if (a < 0) a = -a;
        if (a > m) m = a;
    }
    if (m < 1e-8f) m = 1e-8f; // avoid zero scale
    return m;
}

static inline int clampi(int v, int lo, int hi){
#pragma HLS INLINE
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// --------------- Conv1 (fixed core, float I/O) ---------------

void Conv1_28x28x1_5x5x20_1_0(
    float input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    float kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    float bias[CONV1_NBOUTPUT],
    float output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]
){
#pragma HLS INLINE off

    // Choose padding mode
    const int pad = CONV1_SAME ? (CONV1_DIM - 1) / 2 : CONV1_PAD;

    // ---- Build per-tensor scales (simple calibration on the fly) ----
    const int n_in  = IMG_DEPTH*IMG_HEIGHT*IMG_WIDTH;
    const int n_w1  = CONV1_NBOUTPUT*IMG_DEPTH*CONV1_DIM*CONV1_DIM;

    const float sx  = maxabs_f(&input[0][0][0], n_in) / 127.0f;      // input scale
    const float sw  = maxabs_f(&kernel[0][0][0][0], n_w1) / 127.0f;  // weight scale
    const float sy  = sx; // keep same dynamic as input for simplicity

    int32_t rq_mul; int rq_shift;
    // M = (sx * sw) / sy
    choose_mul_shift((sx * sw) / sy, &rq_mul, &rq_shift);

    // ---- Quantize input/weights/bias once (int8/int32) ----
    static int8_t in_q[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
#pragma HLS ARRAY_PARTITION variable=in_q complete dim=1
    static int8_t w_q[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM];
#pragma HLS ARRAY_PARTITION variable=w_q complete dim=2
    static int32_t b_q[CONV1_NBOUTPUT];

    // quantize input
    {
        const float inv_sx = 1.0f / sx;
        for (int c=0;c<IMG_DEPTH;c++)
            for (int y=0;y<IMG_HEIGHT;y++)
                for (int x=0;x<IMG_WIDTH;x++){
#pragma HLS PIPELINE II=1
                    int v = (int)lrintf(input[c][y][x] * inv_sx);
                    in_q[c][y][x] = clamp_i8(v);
                }
    }
    // quantize weights
    {
        const float inv_sw = 1.0f / sw;
        for (int m=0;m<CONV1_NBOUTPUT;m++)
            for (int c=0;c<IMG_DEPTH;c++)
                for (int ky=0;ky<CONV1_DIM;ky++)
                    for (int kx=0;kx<CONV1_DIM;kx++){
#pragma HLS PIPELINE II=1
                        int v = (int)lrintf(kernel[m][c][ky][kx] * inv_sw);
                        w_q[m][c][ky][kx] = clamp_i8(v);
                    }
    }
    // quantize bias: b_q = round(b / (sx*sw))
    {
        const float inv_b = 1.0f / (sx * sw);
        for (int m=0;m<CONV1_NBOUTPUT;m++){
#pragma HLS PIPELINE II=1
            b_q[m] = (int32_t)lrintf(bias[m] * inv_b);
        }
    }

    // ---- Int8 conv core (pad + stride) -> int32 acc -> int8 via requant ----
    for (int m = 0; m < CONV1_NBOUTPUT; m++){
        for (int y = 0; y < CONV1_HEIGHT; y++){
            for (int x = 0; x < CONV1_WIDTH; x++){
#pragma HLS PIPELINE II=1
                int32_t acc = b_q[m];

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
                            acc += (int32_t)in_q[c][in_y][in_x] * (int32_t)w_q[m][c][ky][kx];
                        }
                    }
                }

                // Requantize to int8 (no activation here)
                int32_t z_q = (rq_shift>0) ? mul_shift_round(acc, rq_mul, rq_shift) : acc;
                int8_t  y_q = clamp_i8(z_q);

                // Dequantize to float for output tensor
                output[m][y][x] = (float)y_q * sy;
            }
        }
    }
}

// --------------- Conv2 (fixed core, float I/O) ---------------

void Conv2_12x12x20_5x5x40_1_0(
    float input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
    float kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    float bias[CONV2_NBOUTPUT],
    float output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]
){
#pragma HLS INLINE off

    const int pad = CONV2_SAME ? (CONV2_DIM - 1) / 2 : CONV2_PAD;

    // ---- Per-tensor scales (simple, fast) ----
    const int n_in  = POOL1_NBOUTPUT*POOL1_HEIGHT*POOL1_WIDTH;
    const int n_w2  = CONV2_NBOUTPUT*POOL1_NBOUTPUT*CONV2_DIM*CONV2_DIM;

    const float sx  = maxabs_f(&input[0][0][0], n_in) / 127.0f;
    const float sw  = maxabs_f(&kernel[0][0][0][0], n_w2) / 127.0f;
    const float sy  = sx; // keep same dynamic across layers

    int32_t rq_mul; int rq_shift;
    choose_mul_shift((sx * sw) / sy, &rq_mul, &rq_shift);

    // ---- Quantize operands ----
    static int8_t in_q[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH];
#pragma HLS ARRAY_PARTITION variable=in_q complete dim=1
    static int8_t w_q[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM];
#pragma HLS ARRAY_PARTITION variable=w_q complete dim=2
    static int32_t b_q[CONV2_NBOUTPUT];

    {
        const float inv_sx = 1.0f / sx;
        for (int c=0;c<POOL1_NBOUTPUT;c++)
            for (int y=0;y<POOL1_HEIGHT;y++)
                for (int x=0;x<POOL1_WIDTH;x++){
#pragma HLS PIPELINE II=1
                    int v = (int)lrintf(input[c][y][x] * inv_sx);
                    in_q[c][y][x] = clamp_i8(v);
                }
    }
    {
        const float inv_sw = 1.0f / sw;
        for (int m=0;m<CONV2_NBOUTPUT;m++)
            for (int c=0;c<POOL1_NBOUTPUT;c++)
                for (int ky=0;ky<CONV2_DIM;ky++)
                    for (int kx=0;kx<CONV2_DIM;kx++){
#pragma HLS PIPELINE II=1
                        int v = (int)lrintf(kernel[m][c][ky][kx] * inv_sw);
                        w_q[m][c][ky][kx] = clamp_i8(v);
                    }
    }
    {
        const float inv_b = 1.0f / (sx * sw);
        for (int m=0;m<CONV2_NBOUTPUT;m++){
#pragma HLS PIPELINE II=1
            b_q[m] = (int32_t)lrintf(bias[m] * inv_b);
        }
    }

    // ---- Int8 conv core ----
    for (int m = 0; m < CONV2_NBOUTPUT; m++){
        for (int y = 0; y < CONV2_HEIGHT; y++){
            for (int x = 0; x < CONV2_WIDTH; x++){
#pragma HLS PIPELINE II=1
                int32_t acc = b_q[m];

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
                            acc += (int32_t)in_q[c][in_y][in_x] * (int32_t)w_q[m][c][ky][kx];
                        }
                    }
                }

                int32_t z_q = (rq_shift>0) ? mul_shift_round(acc, rq_mul, rq_shift) : acc;
                int8_t  y_q = clamp_i8(z_q);
                output[m][y][x] = (float)y_q * sy; // back to float
            }
        }
    }
}
