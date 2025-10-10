// softmax.c - stable softmax
#include <math.h>
#include "lenet_cnn_float.h"

void Softmax(float in[FC2_NBOUTPUT], float out[FC2_NBOUTPUT]){
    float maxv = in[0];
    for (int i = 1; i < FC2_NBOUTPUT; ++i)
        if (in[i] > maxv) maxv = in[i];

    float sum = 0.0f;
    for (int i = 0; i < FC2_NBOUTPUT; ++i){
        float e = expf(in[i] - maxv);
        out[i] = e;
        sum += e;
    }
    if (sum == 0.0f) sum = 1.0f;
    for (int i = 0; i < FC2_NBOUTPUT; ++i)
        out[i] /= sum;
}
