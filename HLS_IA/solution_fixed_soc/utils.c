/**
  ******************************************************************************
  * @file    utils.c
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @version V1.0
  * @date    04 february 2019
  * @brief   Plain C code for the implementation of Convolutional Neural Networks on FPGA
  * @brief   Designed to support Vivado HLS synthesis
  */


#include <stdio.h>
#include <stdlib.h>

#include "lenet_cnn_fixed.h"
void ReadPgmFile(char *filename, unsigned char *pix) {
  FILE* pgm_file; 
  int i, width, height, max, ret; 
  char readChars[256]; 

  pgm_file = fopen( filename, "rb" );
  if (!pgm_file) {
    printf("Error: Unable to open file %s.\n", filename);
    exit(1);
  }

  ret = fscanf (pgm_file, "%s", readChars); 
  ret = fscanf (pgm_file, "%d", &width);
  ret = fscanf (pgm_file, "%d", &height);
  ret = fscanf (pgm_file, "%d", &max);
//  printf("Reading PGM file %s \t -> Type %s, width %d, height %d, max %d\n", filename, readChars, width, height, max);
//  if (width != IMG_WIDTH) printf("Warning: Image width mismatch (%d, expecting %d) \t -> Consider rescaling\n", width, IMG_WIDTH); 
//  if (height != IMG_HEIGHT) printf("Warning: Image height mismatch(%d, expecting %d) \t -> Consider rescaling\n", height, IMG_HEIGHT); 

  for (i = 0; i < width*height; i++) // DEBUG IF IMG_DEPTH > 1 ??
    ret = fscanf(pgm_file, "%c", &pix[i]); 

  fclose(pgm_file); 
}


void WritePgmFile(char *filename, float *pix, short width, short height) {
  FILE* pgm_file; 
  short i; 

  pgm_file = fopen( filename, "w" );
  if (!pgm_file) {
    printf("Error: Unable to open file %s.\n", filename);
    exit(1);
  }
  fprintf (pgm_file, "P2\n"); 
  fprintf (pgm_file, "%d %d\n", width, height);
  fprintf (pgm_file, "255\n");

  for (i = 0; i < width*height; i++) { // DEBUG IF IMG_DEPTH > 1 ??
    fprintf(pgm_file, "%d ", (unsigned char)(pix[i]*64)); // *64 because pix values are too small
    if ( i%width == width-1 ) fprintf(pgm_file, "\n");  
  }

  fclose(pgm_file); 
}


void ReadTestLabels(char *filename, short size) {
  FILE* label_file; 
  int ret; 
  short k; 
  unsigned char label; 

  label_file = fopen( filename, "r" );
  if (!label_file) {
    printf("Error: Unable to open file %s.\n", filename);
    exit(1);
  }

  for (k = 0; k < size; k++) {
    ret = fscanf(label_file, "%c", &label); 
    if (k >= 8) printf("img%d -> 0x%x \n" , k - 8, label); 
  }
  printf("\n"); 

  fclose(label_file); 
}


// Nearest neighbor, linear interpolation
// Based on 
// http://courses.cs.vt.edu/~masc1044/L17-Rotation/ScalingNN.html
#define min(a,b) ( (a) < (b) ? (a) : (b) )
void RescaleImg(unsigned char *input, short width,short height, float *output, short new_width, short new_height) {
  short x, y; 
  short interpol_x, interpol_y; 

  for (y=0; y<new_height; y++) {
    for (x=0; x<new_width; x++) {
      interpol_x = (short)( ((float)x/(float)new_width)*(float)width + 0.5 ); 
      interpol_x = min( interpol_x, width-1); 
      interpol_y = (short)( ((float)y/(float)new_height)*(float)height + 0.5 ); // MOVE TO Y LOOP
      interpol_y = min( interpol_y, height-1); 
      output[(y*new_width)+x] = input[(interpol_y*width)+interpol_x]; 
    }
  }
}

void NormalizeImg(unsigned char *input, float *output, short width, short height) {
  short x, y; 

  for (y=0; y<height; y++) 
    for (x=0; x<width; x++) 
      output[(y*width)+x] = ( (float)input[(y*width)+x] / 255 ); 

}


/* Used to generate weights */
void WriteWeights(char *filename, short weight[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM]) {
  FILE* 	weight_file; 
  short 	i, j, k, l; 

  weight_file = fopen( filename, "w" );
  if (!weight_file) {
    printf("Error: Unable to open file %s.\n", filename);
    exit(1);
  }

  fprintf (weight_file, "short CONV1_KERNEL[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM] = { \n");
  for (i = 0; i < CONV1_NBOUTPUT; i++) {
    fprintf (weight_file, "{ \n");
    for (j = 0; j < IMG_DEPTH; j++) {
      fprintf (weight_file, "{ \n");
      for (k = 0; k < CONV1_DIM; k++) {
		fprintf (weight_file, "{ "); 
        for (l = 0; l < CONV1_DIM; l++)
	  	  fprintf(weight_file, "%d, ", weight[i][j][k][l]); 
	    fprintf (weight_file, "}, ");
      }
   	  fprintf (weight_file, "}, \n");
 	}
    fprintf (weight_file, "}, \n");
  }
  fprintf (weight_file, "}; \n");

  fclose(weight_file); 
}



// Flatten layer impacts reading order: 
// Keras / Tensorflow uses NHWC channels last
// so the 800 (50*4*4) flatten values are in order NHWC channels last


/* ===================== No-HDF5 weight readers =====================
 * Kept only for compatibility with older code paths.
 * They copy from the embedded arrays in weights.h.
 */
#include "weights.h"
#include <string.h>

void ReadConv1Weights(char* filename, char* datasetname,
    float kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM])
{
    (void)filename; (void)datasetname;
    for (int m=0;m<CONV1_NBOUTPUT;m++)
        for (int c=0;c<IMG_DEPTH;c++)
            for (int y=0;y<CONV1_DIM;y++)
                for (int x=0;x<CONV1_DIM;x++)
                    kernel[m][c][y][x] = (float)CONV1_KERNEL[m][c][y][x];
}

void ReadConv1Bias(char* filename, char* datasetname,
    float bias[CONV1_NBOUTPUT])
{
    (void)filename; (void)datasetname;
    for (int m=0;m<CONV1_NBOUTPUT;m++)
        bias[m] = (float)CONV1_BIAS[m];
}

void ReadConv2Weights(char* filename, char* datasetname,
    float kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM])
{
    (void)filename; (void)datasetname;
    for (int m=0;m<CONV2_NBOUTPUT;m++)
        for (int c=0;c<POOL1_NBOUTPUT;c++)
            for (int y=0;y<CONV2_DIM;y++)
                for (int x=0;x<CONV2_DIM;x++)
                    kernel[m][c][y][x] = (float)CONV2_KERNEL[m][c][y][x];
}

void ReadConv2Bias(char* filename, char* datasetname,
    float bias[CONV2_NBOUTPUT])
{
    (void)filename; (void)datasetname;
    for (int m=0;m<CONV2_NBOUTPUT;m++)
        bias[m] = (float)CONV2_BIAS[m];
}

void ReadFc1Weights(char* filename, char* datasetname,
    float kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH])
{
    (void)filename; (void)datasetname;
    for (int o=0;o<FC1_NBOUTPUT;o++)
        for (int c=0;c<POOL2_NBOUTPUT;c++)
            for (int y=0;y<POOL2_HEIGHT;y++)
                for (int x=0;x<POOL2_WIDTH;x++)
                    kernel[o][c][y][x] = (float)FC1_KERNEL[o][c][y][x];
}

void ReadFc1Bias(char* filename, char* datasetname,
    float bias[FC1_NBOUTPUT])
{
    (void)filename; (void)datasetname;
    for (int o=0;o<FC1_NBOUTPUT;o++)
        bias[o] = (float)FC1_BIAS[o];
}

void ReadFc2Weights(char* filename, char* datasetname,
    float kernel[FC2_NBOUTPUT][FC1_NBOUTPUT])
{
    (void)filename; (void)datasetname;
    for (int o=0;o<FC2_NBOUTPUT;o++)
        for (int i=0;i<FC1_NBOUTPUT;i++)
            kernel[o][i] = (float)FC2_KERNEL[o][i];
}

void ReadFc2Bias(char* filename, char* datasetname,
    float bias[FC2_NBOUTPUT])
{
    (void)filename; (void)datasetname;
    for (int o=0;o<FC2_NBOUTPUT;o++)
        bias[o] = (float)FC2_BIAS[o];
}
