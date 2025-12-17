// conv_fixed.c — int8 fixed point for LeNet (no calib, no per-channel act scales)
// - Int8 inputs/weights, int32 accumulators
// - Per-output-channel quant for weights & bias (no runtime calib)
// - Uniform activation scales per layer (tuned to reduce clipping)
// - Leaky-ReLU on conv outputs (helps recover neg info lost by int8)

#include "lenet_cnn_fixed.h"
#include <stdint.h>

// ---------------- config ----------------
// If you pre-normalize MNIST as (x/255 - 0.1307)/0.3081 in your input loader,
// define USE_MNIST_STD_NORM=1 at compile time; else leave it 0.
// With z-score inputs (std~1), SX_CONV1_IN=1/32 maps ~±4σ to ±127.
#ifndef USE_MNIST_STD_NORM
#define USE_MNIST_STD_NORM 1
#endif

// Activation scales (uniform per layer)
#define SY_CONV1  (1.0f/24.0f)
#define SY_CONV2  (1.0f/24.0f)
#define SY_FC1    (1.0f/24.0f)
#define SY_FC2    (1.0f/12.0f)   // give logits more headroom

#if USE_MNIST_STD_NORM
  #define SX_CONV1_IN (1.0f/32.0f)  // z-score input (std~1)
#else
  #define SX_CONV1_IN (1.0f/64.0f)  // centered in [-0.5,0.5]
#endif

// Leaky-ReLU slope
#ifndef LRELU_SLOPE
#define LRELU_SLOPE 0.05f
#endif

// ---------------- helpers ----------------
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
static inline float lrelu(float x){
#pragma HLS INLINE
    return (x >= 0.0f) ? x : (LRELU_SLOPE * x);
}

// ---------------- caches ----------------
static int conv1_inited = 0, conv2_inited = 0;

// per-channel weight scales, bias quant, requant mul
static int8_t  CONV1_W_Q[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM];
static int32_t CONV1_B_Q[CONV1_NBOUTPUT];
static int32_t CONV1_RQ_MUL[CONV1_NBOUTPUT];
static int     CONV1_RQ_SHIFT = 24;
static float   CONV1_SW[CONV1_NBOUTPUT];

static int8_t  CONV2_W_Q[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM];
static int32_t CONV2_B_Q[CONV2_NBOUTPUT];
static int32_t CONV2_RQ_MUL[CONV2_NBOUTPUT];
static int     CONV2_RQ_SHIFT = 24;
static float   CONV2_SW[CONV2_NBOUTPUT];

static void init_conv1_once(
    float kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    float bias  [CONV1_NBOUTPUT]
){
    if (conv1_inited) return;

    for (int m=0;m<CONV1_NBOUTPUT;m++){
        float maxw = 0.0f;
        for (int c=0;c<IMG_DEPTH;c++)
            for (int ky=0;ky<CONV1_DIM;ky++)
                for (int kx=0;kx<CONV1_DIM;kx++){
                    float a = fabsf_fast(kernel[m][c][ky][kx]);
                    if (a > maxw) maxw = a;
                }
        CONV1_SW[m] = (maxw > 1e-8f) ? (maxw/127.0f) : (1.0f/127.0f);

        float inv_sw = 1.0f / CONV1_SW[m];
        for (int c=0;c<IMG_DEPTH;c++)
            for (int ky=0;ky<CONV1_DIM;ky++)
                for (int kx=0;kx<CONV1_DIM;kx++){
                    int v = qroundf(kernel[m][c][ky][kx] * inv_sw);
                    CONV1_W_Q[m][c][ky][kx] = clamp_i8(v);
                }
        float inv_b = 1.0f / (SX_CONV1_IN * CONV1_SW[m]);
        CONV1_B_Q[m] = (int32_t)qroundf(bias[m] * inv_b);

        // requant to uniform SY_CONV1
        float M = (SX_CONV1_IN * CONV1_SW[m]) / SY_CONV1;
        CONV1_RQ_MUL[m] = choose_mul_from_scale(M);
    }
    conv1_inited = 1;
}

static void init_conv2_once(
    float kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    float bias  [CONV2_NBOUTPUT]
){
    if (conv2_inited) return;

    for (int m=0;m<CONV2_NBOUTPUT;m++){
        float maxw = 0.0f;
        for (int c=0;c<POOL1_NBOUTPUT;c++)
            for (int ky=0;ky<CONV2_DIM;ky++)
                for (int kx=0;kx<CONV2_DIM;kx++){
                    float a = fabsf_fast(kernel[m][c][ky][kx]);
                    if (a > maxw) maxw = a;
                }
        CONV2_SW[m] = (maxw > 1e-8f) ? (maxw/127.0f) : (1.0f/127.0f);

        float inv_sw = 1.0f / CONV2_SW[m];
        for (int c=0;c<POOL1_NBOUTPUT;c++)
            for (int ky=0;ky<CONV2_DIM;ky++)
                for (int kx=0;kx<CONV2_DIM;kx++){
                    int v = qroundf(kernel[m][c][ky][kx] * inv_sw);
                    CONV2_W_Q[m][c][ky][kx] = clamp_i8(v);
                }

        // input scale to Conv2 equals SY_CONV1 (uniform)
        float inv_b = 1.0f / (SY_CONV1 * CONV2_SW[m]);
        CONV2_B_Q[m] = (int32_t)qroundf(bias[m] * inv_b);

        float M = (SY_CONV1 * CONV2_SW[m]) / SY_CONV2;
        CONV2_RQ_MUL[m] = choose_mul_from_scale(M);
    }
    conv2_inited = 1;
}

void Conv1_28x28x1_5x5x20_1_0(
    float input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    float kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    float bias[CONV1_NBOUTPUT],
    float output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]
){
#pragma HLS INLINE off
    const int pad = CONV1_PAD;
    init_conv1_once(kernel, bias);

    // quantize input with SX_CONV1_IN
    static int8_t in_q[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
#pragma HLS ARRAY_PARTITION variable=in_q complete dim=1
    {
        float inv_sx = 1.0f / SX_CONV1_IN;
        for (int c=0;c<IMG_DEPTH;c++)
            for (int y=0;y<IMG_HEIGHT;y++)
                for (int x=0;x<IMG_WIDTH;x++){
#pragma HLS PIPELINE II=1
                    int v = qroundf(input[c][y][x] * inv_sx);
                    in_q[c][y][x] = clamp_i8(v);
                }
    }

    for (int m=0; m<CONV1_NBOUTPUT; m++){
        for (int y=0; y<CONV1_HEIGHT; y++){
            for (int x=0; x<CONV1_WIDTH; x++){
#pragma HLS PIPELINE II=1
                int32_t acc = CONV1_B_Q[m];
                for (int c=0; c<IMG_DEPTH; c++){
                    for (int ky=0; ky<CONV1_DIM; ky++){
                        int in_y = y * CONV1_STRIDE + ky - pad;
                        if ((in_y < 0) || (in_y >= IMG_HEIGHT)) continue;
                        for (int kx=0; kx<CONV1_DIM; kx++){
                            int in_x = x * CONV1_STRIDE + kx - pad;
                            if ((in_x < 0) || (in_x >= IMG_WIDTH)) continue;
                            acc += (int32_t)in_q[c][in_y][in_x] * (int32_t)CONV1_W_Q[m][c][ky][kx];
                        }
                    }
                }
                int32_t z_q = mul_shift_round(acc, CONV1_RQ_MUL[m], CONV1_RQ_SHIFT);
                if (z_q > 126) z_q = 126; if (z_q < -127) z_q = -127;
                int8_t  y_q = clamp_i8(z_q);
                output[m][y][x] = lrelu((float)y_q * SY_CONV1);
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
    const int pad = CONV2_PAD;
    init_conv2_once(kernel, bias);

    // quantize input with uniform SX = SY_CONV1
    static int8_t in_q[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH];
#pragma HLS ARRAY_PARTITION variable=in_q complete dim=1
    {
        float inv_sx = 1.0f / SY_CONV1;
        for (int c=0;c<POOL1_NBOUTPUT;c++)
            for (int y=0;y<POOL1_HEIGHT;y++)
                for (int x=0;x<POOL1_WIDTH;x++){
#pragma HLS PIPELINE II=1
                    int v = qroundf(input[c][y][x] * inv_sx);
                    in_q[c][y][x] = clamp_i8(v);
                }
    }

    for (int m=0; m<CONV2_NBOUTPUT; m++){
        for (int y=0; y<CONV2_HEIGHT; y++){
            for (int x=0; x<CONV2_WIDTH; x++){
#pragma HLS PIPELINE II=1
                int32_t acc = CONV2_B_Q[m];
                for (int c=0; c<POOL1_NBOUTPUT; c++){
                    for (int ky=0; ky<CONV2_DIM; ky++){
                        int in_y = y * CONV2_STRIDE + ky - pad;
                        if ((in_y < 0) || (in_y >= POOL1_HEIGHT)) continue;
                        for (int kx=0; kx<CONV2_DIM; kx++){
                            int in_x = x * CONV2_STRIDE + kx - pad;
                            if ((in_x < 0) || (in_x >= POOL1_WIDTH)) continue;
                            acc += (int32_t)in_q[c][in_y][in_x] * (int32_t)CONV2_W_Q[m][c][ky][kx];
                        }
                    }
                }
                int32_t z_q = mul_shift_round(acc, CONV2_RQ_MUL[m], CONV2_RQ_SHIFT);
                if (z_q > 126) z_q = 126; if (z_q < -127) z_q = -127;
                int8_t  y_q = clamp_i8(z_q);
                output[m][y][x] = lrelu((float)y_q * SY_CONV2);
            }
        }
    }
}
