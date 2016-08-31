#include <cstdio>
#include <fstream>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include "CudaHostTimer.h"
#include "cudaMatch.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////

// Define this to enable error checking
#define CUDA_CHECK_ERROR
#define CudaCheckError()        __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
    do
    {
        cudaError err = cudaGetLastError();
        if ( cudaSuccess != err )
        {
                        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }

                // More careful checking. However, this will affect performance.
                // Comment away if needed.
        err = cudaDeviceSynchronize();
        if( cudaSuccess != err )
        {
                        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
    } while ( 0 );
#endif // CUDA_CHECK_ERROR
}

////////////////////////////////////////////////////////////////////////////////


__global__ void computeScore(
        unsigned char * d_input,
        unsigned char * d_ref,
        int*   d_score,
        int*   d_mark,
        int    inputNum,
        int    refNum
        )
{
    const int tx       = blockDim.x * blockIdx.x + threadIdx.x;
    const int totThNum = gridDim.x * blockDim.x;

    for ( int idx = tx; idx < refNum; idx += totThNum )
    {
        const int refIdx = idx;
        const int inpIdx = idx % inputNum;

        const int dif = d_input[ inpIdx ] & d_ref[ refIdx ];

        if ( dif )
        	d_score[ refIdx ] = 1 ;

        else
        	d_score[ refIdx ] = 0;

        const int segIdx = idx / inputNum;
        d_mark[ refIdx ] = segIdx;
    }
}

///// DEPENDENT ON SIZE OF THE IMAGES ////////////
struct markLastElem
{
    __device__ int operator() ( const int inVal )
    {
        // If last element of segment
        if ( 0 == ( inVal + 1 ) % ( 512 * 512 ) )
            return 1;
        else
            return 0;
    }
};
/////////////////////////////

struct isLastElem
{
    __device__ bool operator() ( const int inVal )
    {
        return 1 == inVal;
    }
};


//computes the image with the minimum score
extern "C" unsigned int findMatch(  unsigned char* d_input,
                           unsigned char* d_ref,
                           int imgNum,
                           int rows,
                           int cols
                         )
{
	const int scoreSize = sizeof( int ) * rows * cols * imgNum;
	const int markSize  = sizeof( int ) * rows * cols * imgNum;
	const int resSize   = sizeof( int ) * imgNum;

	int*   d_score;
	int*   d_mark;
	int*   d_res;


	cudaMalloc( &d_score, scoreSize );
	cudaMalloc( &d_mark,  markSize );
	cudaMalloc( &d_res,   resSize );


	thrust::device_ptr<unsigned char> t_input = thrust::device_pointer_cast(d_input);
	thrust::device_ptr<unsigned char> t_ref   = thrust::device_pointer_cast(d_ref);
	thrust::device_ptr<int> t_score   = thrust::device_pointer_cast(d_score);
	thrust::device_ptr<int> t_mark    = thrust::device_pointer_cast(d_mark);
	thrust::device_ptr<int> t_res     = thrust::device_pointer_cast(d_res);

	///////////////////////////////////////////////////////////////////////////
	    // Do computation

	    CudaHostTimer timer;
	    timer.start();

	    // Compute score for each *pixel*
	    computeScore<<< 1000, 32 >>>(
	            d_input,
	            d_ref,
	            d_score,
	            d_mark,
	            rows * cols,
	            rows * cols * imgNum );
	    CudaCheckError();

	    // Segmented scan
	    thrust::inclusive_scan_by_key(
	            t_mark,
	            t_mark + ( rows * cols * imgNum ),
	            t_score,
	            t_score);
	    CudaCheckError();

	    // Create 0,1,2,...
	    thrust::sequence(
	            t_mark,
	            t_mark + ( rows * cols * imgNum ) );
	    CudaCheckError();

	    // Mark last element of each segment
	    thrust::transform(
	            t_mark,
	            t_mark + ( rows * cols * imgNum ),
	            t_mark,
	            markLastElem() );
	    CudaCheckError();

	    // Compact scores from each segment
	    thrust::copy_if(
	            t_score,
	            t_score + ( rows * cols * imgNum ),
	            t_mark,
	            t_res,
	            isLastElem() );
	    CudaCheckError();

	    // Find min score
	    thrust::device_ptr<int> minPtr =
	    thrust::min_element(
	            t_res,
	            t_res + imgNum );
	    CudaCheckError();

	    // Get min val and index
	    const int minIdx = thrust::raw_pointer_cast( minPtr ) - d_res;
	    unsigned int minScore;
	    cudaMemcpy( &minScore, d_res + minIdx, sizeof( int ), cudaMemcpyDeviceToHost );
	    CudaCheckError();

	    timer.stop();
	    //printf( "Match time %f\n", timer.value() );
	    ///////////////////////////////////////////////////////////////////////////

	    cudaFree( d_res );
	    cudaFree( d_mark );
	    cudaFree( d_score );

	    return minScore;
}

//int main()
//{
//    // Read from files
//
//    const int rows = 160, cols = 160;
//	short input[rows * cols];
//	readmat(input, "../output/input.txt", rows, cols);
//
//    short* ref = new short[ 100 * rows * cols];
//	for (int i = 0; i < 100; i++)
//	{
//		char filename[128];
//		sprintf(filename, "../output/ref%03d.txt", i);
//		readmat(ref + i * ( rows * cols ), filename, rows, cols);
//	}
//
//    // Copy to device
//
//    const int imgNum    = 100;
//    const int inputSize = sizeof( short ) * rows * cols;
//    const int refSize   = inputSize * imgNum;
//    const int scoreSize = sizeof( int ) * rows * cols * imgNum;
//    const int markSize  = sizeof( int ) * rows * cols * imgNum;
//    const int resSize   = sizeof( int ) * imgNum;
//
//    short* d_input;
//    short* d_ref;
//    int*   d_score;
//    int*   d_mark;
//    int*   d_res;
//
//    cudaMalloc( &d_input, inputSize );
//    cudaMalloc( &d_ref,   refSize );
//    cudaMalloc( &d_score, scoreSize );
//    cudaMalloc( &d_mark,  markSize );
//    cudaMalloc( &d_res,   resSize );
//
//    thrust::device_ptr<short> t_input = thrust::device_pointer_cast(d_input);
//    thrust::device_ptr<short> t_ref   = thrust::device_pointer_cast(d_ref);
//    thrust::device_ptr<int> t_score   = thrust::device_pointer_cast(d_score);
//    thrust::device_ptr<int> t_mark    = thrust::device_pointer_cast(d_mark);
//    thrust::device_ptr<int> t_res     = thrust::device_pointer_cast(d_res);
//
//    cudaMemcpy( d_input, input, inputSize, cudaMemcpyHostToDevice );
//    cudaMemcpy( d_ref, ref, refSize, cudaMemcpyHostToDevice );
//
//    CudaCheckError();
//
//    ///////////////////////////////////////////////////////////////////////////
//    // Do computation
//
//    CudaHostTimer timer;
//    timer.start();
//
//    // Compute score for each *pixel*
//    computeScore<<< 256, 256 >>>(
//            d_input,
//            d_ref,
//            d_score,
//            d_mark,
//            rows * cols,
//            rows * cols * imgNum );
//    CudaCheckError();
//
//    // Segmented scan
//    thrust::inclusive_scan_by_key(
//            t_mark,
//            t_mark + ( rows * cols * imgNum ),
//            t_score,
//            t_score);
//    CudaCheckError();
//
//    // Create 0,1,2,...
//    thrust::sequence(
//            t_mark,
//            t_mark + ( rows * cols * imgNum ) );
//    CudaCheckError();
//
//    // Mark last element of each segment
//    thrust::transform(
//            t_mark,
//            t_mark + ( rows * cols * imgNum ),
//            t_mark,
//            markLastElem() );
//    CudaCheckError();
//
//    // Compact scores from each segment
//    thrust::copy_if(
//            t_score,
//            t_score + ( rows * cols * imgNum ),
//            t_mark,
//            t_res,
//            isLastElem() );
//    CudaCheckError();
//
//    // Find min score
//    thrust::device_ptr<int> minPtr =
//    thrust::min_element(
//            t_res,
//            t_res + imgNum );
//    CudaCheckError();
//
//    // Get min val and index
//    const int minIdx = thrust::raw_pointer_cast( minPtr ) - d_res;
//    int minScore;
//    cudaMemcpy( &minScore, d_res + minIdx, sizeof( int ), cudaMemcpyDeviceToHost );
//    CudaCheckError();
//
//    timer.stop();
//    printf( "Time(s): %f\n", timer.value() );
//    ///////////////////////////////////////////////////////////////////////////
//
//    printf( "MinIdx: %d MinScore: %d\n", minIdx, minScore );
//
//    // Free memory
//    cudaFree( d_ref );
//    cudaFree( d_input );
//    cudaFree( d_res );
//    cudaFree( d_mark );
//
//    delete ref;
//
//    return 0;
//}
