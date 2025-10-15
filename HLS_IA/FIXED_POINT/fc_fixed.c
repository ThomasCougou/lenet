// fc.c — Fully connected layers (fixed-point core, float I/O)
// Notes:
//   - Same prototypes as lenet_cnn_float.h
//   - Quantize input/weights/bias to int8/int32
//   - Accumulate in int32
//   - Requantize + dequantize to float
//   - Keep HLS-friendly loop structure

#include "lenet_cnn_float.h"
#include <stdint.h>
#include <math.h>

static inline float relu(float x){
#pragma HLS INLINE
    return (x > 0.0f) ? x : 0.0f;
}

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

static inline void choose_mul_shift(float M, int32_t *mul, int *shift){
#pragma HLS INLINE
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
    if (m < 1e-8f) m = 1e-8f;
    return m;
}

// ================== Fc1_40_400 ==================

void Fc1_40_400(
    float input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    float weight[400][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    float bias[400],
    float output[400]
){
#pragma HLS INLINE off

    const int n_in = POOL2_NBOUTPUT*POOL2_HEIGHT*POOL2_WIDTH;
    const int n_w  = 400*POOL2_NBOUTPUT*POOL2_HEIGHT*POOL2_WIDTH;

    const float sx = maxabs_f(&input[0][0][0], n_in) / 127.0f;
    const float sw = maxabs_f(&weight[0][0][0][0], n_w) / 127.0f;
    const float sy = sx;

    int32_t rq_mul; int rq_shift;
    choose_mul_shift((sx * sw) / sy, &rq_mul, &rq_shift);

    static int8_t in_q[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
#pragma HLS ARRAY_PARTITION variable=in_q complete dim=1
    static int8_t w_q[400][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
#pragma HLS ARRAY_PARTITION variable=w_q complete dim=2
    static int32_t b_q[400];

    // quantize input
    {
        const float inv_sx = 1.0f / sx;
        for (int c=0;c<POOL2_NBOUTPUT;c++)
            for (int y=0;y<POOL2_HEIGHT;y++)
                for (int x=0;x<POOL2_WIDTH;x++){
#pragma HLS PIPELINE II=1
                    int v = (int)lrintf(input[c][y][x] * inv_sx);
                    in_q[c][y][x] = clamp_i8(v);
                }
    }

    // quantize weights
    {
        const float inv_sw = 1.0f / sw;
        for (int o=0;o<400;o++)
            for (int c=0;c<POOL2_NBOUTPUT;c++)
                for (int y=0;y<POOL2_HEIGHT;y++)
                    for (int x=0;x<POOL2_WIDTH;x++){
#pragma HLS PIPELINE II=1
                        int v = (int)lrintf(weight[o][c][y][x] * inv_sw);
                        w_q[o][c][y][x] = clamp_i8(v);
                    }
    }

    // quantize bias
    {
        const float inv_b = 1.0f / (sx * sw);
        for (int o=0;o<400;o++){
#pragma HLS PIPELINE II=1
            b_q[o] = (int32_t)lrintf(bias[o] * inv_b);
        }
    }

    // compute MAC in int32
    for (int o = 0; o < 400; o++){
        int32_t acc = b_q[o];
        for (int c = 0; c < POOL2_NBOUTPUT; c++){
#pragma HLS PIPELINE II=1
            for (int y = 0; y < POOL2_HEIGHT; y++){
#pragma HLS UNROLL
                for (int x = 0; x < POOL2_WIDTH; x++){
#pragma HLS UNROLL
                    acc += (int32_t)in_q[c][y][x] * (int32_t)w_q[o][c][y][x];
                }
            }
        }
        int32_t z_q = (rq_shift>0) ? mul_shift_round(acc, rq_mul, rq_shift) : acc;
        int8_t y_q = clamp_i8(z_q);
        output[o] = relu((float)y_q * sy);
    }
}

// ================== Fc2_400_10 ==================

void Fc2_400_10(
    float input[400],
    float weight[10][400],
    float bias[10],
    float output[10]
){
#pragma HLS INLINE off

    const int n_in = 400;
    const int n_w  = 10*400;

    const float sx = maxabs_f(input, n_in) / 127.0f;
    const float sw = maxabs_f(&weight[0][0], n_w) / 127.0f;
    const float sy = sx;

    int32_t rq_mul; int rq_shift;
    choose_mul_shift((sx * sw) / sy, &rq_mul, &rq_shift);

    static int8_t in_q[400];
    static int8_t w_q[10][400];
    static int32_t b_q[10];

    {
        const float inv_sx = 1.0f / sx;
        for (int i=0;i<400;i++){
#pragma HLS PIPELINE II=1
            int v = (int)lrintf(input[i] * inv_sx);
            in_q[i] = clamp_i8(v);
        }
    }

    {
        const float inv_sw = 1.0f / sw;
        for (int o=0;o<10;o++)
            for (int i=0;i<400;i++){
#pragma HLS PIPELINE II=1
                int v = (int)lrintf(weight[o][i] * inv_sw);
                w_q[o][i] = clamp_i8(v);
            }
    }

    {
        const float inv_b = 1.0f / (sx * sw);
        for (int o=0;o<10;o++){
#pragma HLS PIPELINE II=1
            b_q[o] = (int32_t)lrintf(bias[o] * inv_b);
        }
    }

    for (int o=0;o<10;o++){
        int32_t acc = b_q[o];
        for (int i=0;i<400;i++){
#pragma HLS PIPELINE II=1
            acc += (int32_t)in_q[i] * (int32_t)w_q[o][i];
        }
        int32_t z_q = (rq_shift>0) ? mul_shift_round(acc, rq_mul, rq_shift) : acc;
        int8_t y_q = clamp_i8(z_q);
        output[o] = (float)y_q * sy; // no activation, softmax later
    }
}
