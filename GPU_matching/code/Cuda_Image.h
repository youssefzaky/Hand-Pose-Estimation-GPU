#ifndef CUDA_IMAGE_H
#define CUDA_IMAGE_H

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>

template < class Format >
class Cuda_Image
{

public:

	Cuda_Image( unsigned int image_width , unsigned int image_height ) ;

	~Cuda_Image();

	void get_From_Texture( struct cudaGraphicsResource * texture ) ;

private:
	cudaChannelFormatDesc channel;

protected:
	cudaArray * image_data ;
	unsigned int image_width ;
    unsigned int image_height ;

};

template < class Format >
Cuda_Image< Format >::Cuda_Image ( unsigned int image_width , unsigned int image_height )
:image_width( image_width ), image_height( image_height )
{
	channel = cudaCreateChannelDesc<Format>();
	checkCudaErrors( cudaMallocArray( &image_data, &channel, this->image_width , this->image_height,  0 ) );
}

template < class Format >
Cuda_Image< Format >::~Cuda_Image()
{
	checkCudaErrors( cudaFreeArray( image_data ) ) ;
}

template < class Format >
void Cuda_Image< Format >::get_From_Texture( struct cudaGraphicsResource * texture  )
{
	checkCudaErrors(cudaGraphicsMapResources( 1, &texture, 0 ) ) ;

	cudaArray * in_array;

	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray( &in_array, texture, 0, 0 ) );

	checkCudaErrors(cudaMemcpyArrayToArray(image_data, 0, 0, in_array, 0 , 0 , image_width * image_height * sizeof(Format) ));

	checkCudaErrors(cudaGraphicsUnmapResources( 1, &texture, 0) ) ;

}

#endif










