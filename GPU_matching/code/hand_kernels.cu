#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "hand_kernels.h"

#define RADIUS 1

// Texture reference for reading image
texture<uchar4, 2> tex;

extern __shared__ unsigned char LocalBlock[];

__global__
void rgba_to_greyscale_kernel(cudaArray * rgbaImage,
                                  unsigned char* const greyImage,
                                  int numRows,
                                  int numCols)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (y >= numRows && x >= numCols)
  {
      return;
  }

   int i = y * numCols + x;
   uchar4 rgba = tex2D( tex, (float) x, (float) y ) ;
   greyImage[i] =  .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;

}

extern "C" void rgba_to_greyscale( cudaArray * rgbaImage,
                                       unsigned char* const greyImage,
                                       int numRows,
                                       int numCols)
{
   checkCudaErrors(cudaBindTextureToArray(tex, rgbaImage));

   dim3 block(16, 16, 1);
   dim3 grid(numCols / block.x, numRows / block.y, 1);
   rgba_to_greyscale_kernel<<< grid, block >>>( rgbaImage, greyImage, numRows, numCols);

   checkCudaErrors(cudaUnbindTexture(tex));
}

//performs a 5x5 OR filter operation
__global__ void orOperationKernel(unsigned char * g_idata, unsigned char * g_odata, int width, int height, const short filterWidth)
{
	__shared__ unsigned char smem[ 20 * 20 ];

	int R = filterWidth/2 ;
	int tile_w = blockDim.x - 2*R;


	int x = blockIdx.x*tile_w + threadIdx.x - R;
	int y = blockIdx.y*tile_w + threadIdx.y - R;

	// clamp to edge of image
	x = max(0, x);
	x = min(x, width-1);
	y = max(y, 0);
	y = min(y, height-1);

	unsigned int index = y*width + x;
	unsigned int bindex = threadIdx.y*blockDim.y+threadIdx.x;

	// each thread copies its pixel of the block to shared memory
	smem[bindex] = g_idata[index];
	__syncthreads();

	// only threads inside the apron will write results
	if ((threadIdx.x >= R) && (threadIdx.x < (blockDim.x - R)) && (threadIdx.y >= R) && (threadIdx.y < (blockDim.y - R)))
	{
		unsigned char result = 0;
		for(int dy=-R; dy<=R; dy++)
		{
			for(int dx=-R; dx<=R; dx++)
			{
				unsigned char i = smem[bindex + (dy*blockDim.x) + dx];
				result |= i;
			}
		}

		g_odata[index] = result;
	}
}

//wrapper for the orOperation kernel
extern "C" void orOperation(unsigned char * in, unsigned char * out, int width, int height)
{
	const short filterWidth = 5;
	// calculate grid size
	dim3 block(16, 16, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	orOperationKernel<<< grid, block >>>( in, out, width, height, filterWidth );

}

__device__ unsigned char
ComputeSobel(unsigned char ul, // upper left
             unsigned char um, // upper middle
             unsigned char ur, // upper right
             unsigned char ml, // middle left
             unsigned char mm, // middle (unused)
             unsigned char mr, // middle right
             unsigned char ll, // lower left
             unsigned char lm, // lower middle
             unsigned char lr, // lower right
             float fScale)
{
    short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
    short Vert = ul + 2*um + ur - ll - 2*lm - lr;
    short Sum = (short)(fScale*(abs((int)Horz)+abs((int)Vert)));
//    float angle = (atan2f( (float) Vert, (float) Horz) * (180.0f/ 3.1415f)) / 2.0f ;
    float angle = (atan2f( (float) Vert, (float) Horz) * (180.0f/ 3.1415f));
    angle = abs(angle);

    if (Sum == 0)
    {   return 0;

    }

    //compute bin
    unsigned int bin = angle/22.5;
    bin = bin % 8;
    bin = 1 << bin;
    return bin;

}

__global__ void
SobelShared( unsigned char * idata, uchar4 *pSobelOriginal, unsigned short SobelPitch,
#ifndef FIXED_BLOCKWIDTH
            short BlockWidth, short SharedPitch,
#endif
            short w, short h, float fScale)
{
    short u = 4*blockIdx.x*BlockWidth;
    short v = blockIdx.y*blockDim.y + threadIdx.y;
    short ib;

    int SharedIdx = threadIdx.y * SharedPitch;

    for (ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x)
    {
        LocalBlock[SharedIdx+4*ib+0] = idata [ (u+4*ib-RADIUS+0) + (v-RADIUS) * w ];
        LocalBlock[SharedIdx+4*ib+1] = idata [ (u+4*ib-RADIUS+1) + (v-RADIUS) * w ];
        LocalBlock[SharedIdx+4*ib+2] = idata [ (u+4*ib-RADIUS+2) + (v-RADIUS) * w ];
        LocalBlock[SharedIdx+4*ib+3] = idata [ (u+4*ib-RADIUS+3) + (v-RADIUS) * w ];
    }

    if (threadIdx.y < RADIUS*2)
    {
        //
        // copy trailing RADIUS*2 rows of pixels into shared
        //
        SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;

        for (ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x)
        {
            LocalBlock[SharedIdx+4*ib+0] =  idata [ (u+4*ib-RADIUS+0) + (v+blockDim.y-RADIUS) * w ];
            LocalBlock[SharedIdx+4*ib+1] =  idata [ (u+4*ib-RADIUS+1) + (v+blockDim.y-RADIUS) * w ];
            LocalBlock[SharedIdx+4*ib+2] =  idata [ (u+4*ib-RADIUS+2) + (v+blockDim.y-RADIUS) * w ];
            LocalBlock[SharedIdx+4*ib+3] =  idata [ (u+4*ib-RADIUS+3) + (v+blockDim.y-RADIUS) * w ];
        }
    }

    __syncthreads();

    u >>= 2;    // index as uchar4 from here
    uchar4 *pSobel = (uchar4 *)(((char *) pSobelOriginal)+v*SobelPitch);
    SharedIdx = threadIdx.y * SharedPitch;

    for (ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x)
    {

        unsigned char pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+0];
        unsigned char pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+1];
        unsigned char pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+2];
        unsigned char pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+0];
        unsigned char pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+1];
        unsigned char pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+2];
        unsigned char pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+0];
        unsigned char pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+1];
        unsigned char pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+2];

        uchar4 out;

        out.x = ComputeSobel(pix00, pix01, pix02,
                             pix10, pix11, pix12,
                             pix20, pix21, pix22, fScale);

        pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+3];
        pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+3];
        pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+3];
        out.y = ComputeSobel(pix01, pix02, pix00,
                             pix11, pix12, pix10,
                             pix21, pix22, pix20, fScale);

        pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+4];
        pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+4];
        pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+4];
        out.z = ComputeSobel(pix02, pix00, pix01,
                             pix12, pix10, pix11,
                             pix22, pix20, pix21, fScale);

        pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+5];
        pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+5];
        pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+5];
        out.w = ComputeSobel(pix00, pix01, pix02,
                             pix10, pix11, pix12,
                             pix20, pix21, pix22, fScale);

        if (u+ib < w/4 && v < h)
        {
        	pSobel[u+ib] = out;
        }
    }

    __syncthreads();
}

// Wrapper for the __global__ call that sets up the texture and threads
extern "C" void sobelFilter(Pixel * idata, Pixel *odata,  int iw, int ih, enum SobelDisplayMode mode, float fScale)
{

    //checkCudaErrors(cudaBindTextureToArray(tex, array));

	dim3 threads(16,4);
#ifndef FIXED_BLOCKWIDTH
	int BlockWidth = 80; // must be divisible by 16 for coalescing
#endif
	dim3 blocks = dim3(iw/(4*BlockWidth)+(0!=iw%(4*BlockWidth)),
			ih/threads.y+(0!=ih%threads.y));
	int SharedPitch = ~0x3f&(4*(BlockWidth+2*RADIUS)+0x3f);
	int sharedMem = SharedPitch*(threads.y+2*RADIUS);

	// for the shared kernel, width must be divisible by 4
	iw &= ~3;

	SobelShared<<<blocks, threads, sharedMem>>>( idata, (uchar4 *) odata,
			iw,
#ifndef FIXED_BLOCKWIDTH
			BlockWidth, SharedPitch,
#endif
                                                            iw, ih, fScale);

    //checkCudaErrors(cudaUnbindTexture(tex));
}
