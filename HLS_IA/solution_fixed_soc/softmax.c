// softmax.c — stable softmax with Kahan summation and optional temperature
// - Works with lenet_cnn_fixed.h
// - Subtracts max logit for stability (log-sum-exp trick)
// - Clamps exponent range to avoid underflow/overflow
// - Kahan summation to reduce rounding error on the denominator
// - Argmax is computed directly from raw logits (most stable for top-1)
// - Optional temperature scaling (compile-time)

#include "lenet_cnn_fixed.h"
#include <float.h>
#include <math.h>

// ---------------------- Config ----------------------
// Temperature = NUM / DEN. Default = 1.0f
#ifndef SM_TEMPERATURE_NUM
#define SM_TEMPERATURE_NUM 1.0f
#endif
#ifndef SM_TEMPERATURE_DEN
#define SM_TEMPERATURE_DEN 1.0f
#endif

// Exponential clamp range after (x - max) / T
// Values below EXP_LO are effectively 0, above EXP_HI saturate to exp(EXP_HI).
// With float32, exp(-20) ~ 2.06e-9; exp(20) ~ 4.85e8 (not expected here after max subtraction).
#ifndef EXP_LO
#define EXP_LO (-20.0f)
#endif
#ifndef EXP_HI
#define EXP_HI (20.0f)
#endif

static inline float fast_clampf(float x, float lo, float hi){
#pragma HLS INLINE
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

// Optionally use a cheap exp approximation for very small negatives to speed up.
// Keep correctness first: default uses expf.
static inline float safe_expf(float x){
#pragma HLS INLINE
    // Clamp to stay in a well-behaved range
    float z = fast_clampf(x, EXP_LO, EXP_HI);
    return expf(z);
}

// Kahan summation: improves precision of sum of positive terms.
static inline void kahan_add(float *sum, float *c, float y){
#pragma HLS INLINE
    float t  = y - *c;
    float nt = *sum + t;
    *c = (nt - *sum) - t;
    *sum = nt;
}

// Compute softmax with optional temperature and return argmax.
// input:  logits (length SM_INPUT_LEN)
// output: probabilities (length SM_OUTPUT_LEN)
// pred_class: best class id (can be NULL)
void Softmax_10(float input[SM_INPUT_LEN], float output[SM_OUTPUT_LEN], int *pred_class){
#pragma HLS INLINE off

    // 1) Find max logit and argmax on raw logits (no temperature here)
    float max_logit = -FLT_MAX;
    int   max_idx   = 0;
    for (int i = 0; i < SM_INPUT_LEN; i++){
#pragma HLS PIPELINE II=1
        float v = input[i];
        if (v > max_logit){
            max_logit = v;
            max_idx   = i;
        }
    }

    // 2) Apply temperature (T = NUM/DEN) only in the exponent path
    const float T_num = (float)SM_TEMPERATURE_NUM;
    const float T_den = (float)SM_TEMPERATURE_DEN;
    const float invT  = T_num / T_den; // if you want T<1 -> sharper, set NUM<DEN or DEN>NUM

    // 3) Compute unnormalized exp scores: exp((x - max)/T), with clamp; sum with Kahan
    float sum = 0.0f;
    float c   = 0.0f; // Kahan compensation
    for (int i = 0; i < SM_INPUT_LEN; i++){
#pragma HLS PIPELINE II=1
        float s = (input[i] - max_logit) / invT;
        float e = safe_expf(s);
        output[i] = e;         // store temp exp scores first
        kahan_add(&sum, &c, e);
    }

    // 4) Normalize and pick best by probability too (tie-breaker consistent with logits argmax)
    float best_p = -1.0f;
    int   best_i = 0;
    if (sum <= 0.0f){
        // pathological (should not happen); fall back to uniform
        const float u = 1.0f / (float)SM_OUTPUT_LEN;
        for (int i = 0; i < SM_OUTPUT_LEN; i++){
#pragma HLS PIPELINE II=1
            output[i] = u;
        }
        best_i = max_idx; // stay consistent with logits argmax
    } else {
        const float inv_sum = 1.0f / sum;
        for (int i = 0; i < SM_OUTPUT_LEN; i++){
#pragma HLS PIPELINE II=1
            float p = output[i] * inv_sum;
            output[i] = p;
            if (p > best_p){
                best_p = p;
                best_i = i;
            }
        }
    }

    // Prefer the class that wins on raw logits; if someone downstream
    // reads only pred_class, they get the most stable top-1.
    // If you want to prefer prob-based tie-breaks, comment the override line.
    if (pred_class) *pred_class = max_idx;
}

// Wrapper without argmax (kept for compatibility)
void Softmax(float input[SM_INPUT_LEN], float output[SM_OUTPUT_LEN]){
#pragma HLS INLINE off
    Softmax_10(input, output, (int*)0);
}
