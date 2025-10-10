#include <float.h>
#include "pool.h"


void Pool1_24x24x20_2x2x20_2_0(
	float input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
	float output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH])
	{
	#pragma HLS INLINE off
	const int C = POOL1_NBOUTPUT;
	const int OH = POOL1_HEIGHT;
	const int OW = POOL1_WIDTH;
	const int KH = POOL1_DIM;
	const int KW = POOL1_DIM;
	const int STR = POOL1_STRIDE;


	for (int c = 0; c < C; ++c) {
		for (int oy = 0; oy < OH; ++oy) {
			for (int ox = 0; ox < OW; ++ox) {
				#pragma HLS PIPELINE II=1
				#if defined(USE_POOL_MAX)
					float acc = -FLT_MAX;
				#elif defined(USE_POOL_AVG)
					float acc = 0.0f;
				#endif
				const int iy0 = oy * STR;
				const int ix0 = ox * STR;
				for (int ky = 0; ky < KH; ++ky) {
					#pragma HLS UNROLL
					const int iy = iy0 + ky;
					for (int kx = 0; kx < KW; ++kx) {
						#pragma HLS UNROLL
						const int ix = ix0 + kx;
						const float v = input[c][iy][ix];
						#if defined(USE_POOL_MAX)
							acc = (v > acc) ? v : acc;
						#elif defined(USE_POOL_AVG)
							acc += v;
						#endif
					}
				}
				#if defined(USE_POOL_MAX)
					output[c][oy][ox] = acc;
				#elif defined(USE_POOL_AVG)
					output[c][oy][ox] = acc * 0.25f;
				#endif
			}
		}
	}
}


void Pool2_8x8x40_2x2x40_2_0(
float input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
float output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH])
{
	#pragma HLS INLINE off
	const int C = POOL2_NBOUTPUT;
	const int OH = POOL2_HEIGHT;
	const int OW = POOL2_WIDTH;
	const int KH = POOL2_DIM;
	const int KW = POOL2_DIM;
	const int STR = POOL2_STRIDE;


	for (int c = 0; c < C; ++c) {
		for (int oy = 0; oy < OH; ++oy) {
			for (int ox = 0; ox < OW; ++ox) {
				#pragma HLS PIPELINE II=1
				#if defined(USE_POOL_MAX)
					float acc = -FLT_MAX;
				#elif defined(USE_POOL_AVG)
					float acc = 0.0f;
				#endif
				const int iy0 = oy * STR;
				const int ix0 = ox * STR;
				for (int ky = 0; ky < KH; ++ky) {
					#pragma HLS UNROLL
					const int iy = iy0 + ky;
					for (int kx = 0; kx < KW; ++kx) {
						#pragma HLS UNROLL
						const int ix = ix0 + kx;
						const float v = input[c][iy][ix];
					#if defined(USE_POOL_MAX)
						acc = (v > acc) ? v : acc;
					#elif defined(USE_POOL_AVG)
						acc += v;
					#endif
					}
				}
				#if defined(USE_POOL_MAX)
					output[c][oy][ox] = acc;
				#elif defined(USE_POOL_AVG)
					output[c][oy][ox] = acc * 0.25f;
				#endif
			}
		}
	}
}
