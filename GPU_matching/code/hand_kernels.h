#ifndef __HAND_KERNELS_H_
#define __HAND_KERNELS_H_

typedef unsigned char Pixel;

// global determines which filter to invoke
enum SobelDisplayMode
{
    SOBELDISPLAY_IMAGE = 0,
    SOBELDISPLAY_SOBELTEX,
    SOBELDISPLAY_SOBELSHARED
};


extern enum SobelDisplayMode g_SobelDisplayMode;

extern "C" void sobelFilter(Pixel * idata, Pixel *odata,  int iw, int ih, enum SobelDisplayMode mode, float fScale);
extern "C" void orOperation(unsigned char * in, unsigned char * out, int width, int height);
extern "C" void rgba_to_greyscale( cudaArray * rgbaImage,
                                       unsigned char* const greyImage,
                                       int numRows,
                                       int numCols);

#endif
