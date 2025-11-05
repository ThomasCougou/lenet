// fc_fixed.c — int8 fixed point (no calib, no per-channel act scales)
// - Per-output-channel for weights/bias/requant
// - Uniform activation scales; ReLU in Fc1

#include "lenet_cnn_fixed.h"
#include <stdint.h>

#define SY_CONV2  (1.0f/24.0f)
#define SY_FC1    (1.0f/24.0f)
#define SY_FC2    (1.0f/12.0f)

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
static inline int32_t mul_shift_round(int32_t x, int32_t mul, int shift){
#pragma HLS INLINE
    long long t = (long long)x * (long long)mul;
    long long add = (t >= 0) ? (1LL<<(shift-1)) : -(1LL<<(shift-1));
    return (int32_t)((t + add) >> shift);
}
static inline int32_t choose_mul_from_scale(float M){
#pragma HLS INLINE
    const int s = 24;
    float t = M * (float)(1 << s);
    return (int32_t)(t + (t >= 0.0f ? 0.5f : -0.5f));
}
static inline float fabsf_fast(float a){
#pragma HLS INLINE
    return (a >= 0.0f) ? a : -a;
}
static inline float relu_f(float x){
#pragma HLS INLINE
    return (x > 0.0f) ? x : 0.0f;
}

// caches
static int fc1_inited = 0, fc2_inited = 0;
static int8_t  FC1_W_Q[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
static int32_t FC1_B_Q[FC1_NBOUTPUT];
static int32_t FC1_RQ_MUL[FC1_NBOUTPUT];
static int     FC1_RQ_SHIFT = 24;
static float   FC1_SW[FC1_NBOUTPUT];

static int8_t  FC2_W_Q[FC2_NBOUTPUT][FC1_NBOUTPUT];
static int32_t FC2_B_Q[FC2_NBOUTPUT];
static int32_t FC2_RQ_MUL[FC2_NBOUTPUT];
static int     FC2_RQ_SHIFT = 24;
static float   FC2_SW[FC2_NBOUTPUT];

static void init_fc1_once(
    float weight[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    float bias  [FC1_NBOUTPUT]
){
    if (fc1_inited) return;

    for (int o=0;o<FC1_NBOUTPUT;o++){
        float maxw = 0.0f;
        for (int c=0;c<POOL2_NBOUTPUT;c++)
            for (int y=0;y<POOL2_HEIGHT;y++)
                for (int x=0;x<POOL2_WIDTH;x++){
                    float a = fabsf_fast(weight[o][c][y][x]);
                    if (a > maxw) maxw = a;
                }
        FC1_SW[o] = (maxw > 1e-8f) ? (maxw/127.0f) : (1.0f/127.0f);

        float inv_sw = 1.0f / FC1_SW[o];
        for (int c=0;c<POOL2_NBOUTPUT;c++)
            for (int y=0;y<POOL2_HEIGHT;y++)
                for (int x=0;x<POOL2_WIDTH;x++){
                    int v = qroundf(weight[o][c][y][x] * inv_sw);
                    FC1_W_Q[o][c][y][x] = clamp_i8(v);
                }

        float inv_b = 1.0f / (SY_CONV2 * FC1_SW[o]);
        FC1_B_Q[o] = (int32_t)qroundf(bias[o] * inv_b);

        float M = (SY_CONV2 * FC1_SW[o]) / SY_FC1;
        FC1_RQ_MUL[o] = choose_mul_from_scale(M);
    }
    fc1_inited = 1;
}

static void init_fc2_once(
    float weight[FC2_NBOUTPUT][FC1_NBOUTPUT],
    float bias  [FC2_NBOUTPUT]
){
    if (fc2_inited) return;

    for (int o=0;o<FC2_NBOUTPUT;o++){
        float maxw = 0.0f;
        for (int i=0;i<FC1_NBOUTPUT;i++){
            float a = fabsf_fast(weight[o][i]);
            if (a > maxw) maxw = a;
        }
        FC2_SW[o] = (maxw > 1e-8f) ? (maxw/127.0f) : (1.0f/127.0f);

        float inv_sw = 1.0f / FC2_SW[o];
        for (int i=0;i<FC1_NBOUTPUT;i++){
            int v = qroundf(weight[o][i] * inv_sw);
            FC2_W_Q[o][i] = clamp_i8(v);
        }

        float inv_b = 1.0f / (SY_FC1 * FC2_SW[o]);
        FC2_B_Q[o] = (int32_t)qroundf(bias[o] * inv_b);

        float M = (SY_FC1 * FC2_SW[o]) / SY_FC2;
        FC2_RQ_MUL[o] = choose_mul_from_scale(M);
    }
    fc2_inited = 1;
}

void Fc1_40_400(
    float input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    float weight[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    float bias[FC1_NBOUTPUT],
    float output[FC1_NBOUTPUT]
){
#pragma HLS INLINE off
    init_fc1_once(weight, bias);

    // quantize input with SX = SY_CONV2
    static int8_t in_q[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
#pragma HLS ARRAY_PARTITION variable=in_q complete dim=1
    {
        float inv_sx = 1.0f / SY_CONV2;
        for (int c=0;c<POOL2_NBOUTPUT;c++)
            for (int y=0;y<POOL2_HEIGHT;y++)
                for (int x=0;x<POOL2_WIDTH;x++){
#pragma HLS PIPELINE II=1
                    int v = qroundf(input[c][y][x] * inv_sx);
                    in_q[c][y][x] = clamp_i8(v);
                }
    }

    for (int o=0;o<FC1_NBOUTPUT;o++){
        int32_t acc = FC1_B_Q[o];
        for (int c=0;c<POOL2_NBOUTPUT;c++){
#pragma HLS PIPELINE II=1
            for (int y=0;y<POOL2_HEIGHT;y++){
#pragma HLS UNROLL
                for (int x=0;x<POOL2_WIDTH;x++){
#pragma HLS UNROLL
                    acc += (int32_t)in_q[c][y][x] * (int32_t)FC1_W_Q[o][c][y][x];
                }
            }
        }
        int32_t z_q = mul_shift_round(acc, FC1_RQ_MUL[o], FC1_RQ_SHIFT);
        if (z_q > 126) z_q = 126; if (z_q < -127) z_q = -127;
        int8_t  y_q = clamp_i8(z_q);
        output[o] = relu_f((float)y_q * SY_FC1);
    }
}

void Fc2_400_10(
    float input[FC1_NBOUTPUT],
    float weight[FC2_NBOUTPUT][FC1_NBOUTPUT],
    float bias[FC2_NBOUTPUT],
    float output[FC2_NBOUTPUT]
){
#pragma HLS INLINE off
    init_fc2_once(weight, bias);

    static int8_t in_q[FC1_NBOUTPUT];
    {
        float inv_sx = 1.0f / SY_FC1;
        for (int i=0;i<FC1_NBOUTPUT;i++){
#pragma HLS PIPELINE II=1
            int v = qroundf(input[i] * inv_sx);
            in_q[i] = clamp_i8(v);
        }
    }

    for (int o=0;o<FC2_NBOUTPUT;o++){
        int32_t acc = FC2_B_Q[o];
        for (int i=0;i<FC1_NBOUTPUT;i++){
#pragma HLS PIPELINE II=1
            acc += (int32_t)in_q[i] * (int32_t)FC2_W_Q[o][i];
        }
        int32_t z_q = mul_shift_round(acc, FC2_RQ_MUL[o], FC2_RQ_SHIFT);
        if (z_q > 126) z_q = 126; if (z_q < -127) z_q = -127;
        int8_t  y_q = clamp_i8(z_q);
        output[o] = (float)y_q * SY_FC2; // logits
    }
}
