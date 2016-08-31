#ifndef CUDA_IMAGE_HAND_H
#define CUDA_IMAGE_HAND_H

#include "Cuda_Image.h"
#include "hand_kernels.h"

template < class Format >
class Cuda_Image_Hand : public Cuda_Image< Format >
{

public:

	Cuda_Image_Hand( unsigned int image_width, unsigned int image_height ) ;
	~Cuda_Image_Hand() ;
	unsigned char * get_Processed_Image_Pointer() ;

private:
    unsigned char *cuda_final ;
    unsigned char *cuda_temp ;
    unsigned char *cuda_grey ;
    unsigned int size_Of_Data ;

    void convert_To_greyScale() ;
    void sobel_Filter() ;
    void or_Operation() ;

};

template <class Format >
Cuda_Image_Hand< Format > :: Cuda_Image_Hand( unsigned int image_width, unsigned int image_height )
:Cuda_Image< Format >( image_width , image_height )
{
	size_Of_Data = image_height * image_width * sizeof( Format ) ;
	checkCudaErrors( cudaMalloc( (void **) &cuda_final, size_Of_Data ) );
	checkCudaErrors( cudaMalloc( (void **) &cuda_temp, size_Of_Data ) );
	checkCudaErrors( cudaMalloc( (void **) &cuda_grey, size_Of_Data ) );
}

template <class Format >
Cuda_Image_Hand< Format > :: ~Cuda_Image_Hand()
{
	checkCudaErrors( cudaFree(cuda_final) );
	checkCudaErrors( cudaFree(cuda_temp) );
	checkCudaErrors( cudaFree(cuda_grey) );
}

template <class Format >
void Cuda_Image_Hand< Format > :: convert_To_greyScale()
{
	rgba_to_greyscale( this->image_data , cuda_grey, this->image_height, this->image_width );
}

template <class Format >
void Cuda_Image_Hand< Format > :: sobel_Filter()
{
	sobelFilter( cuda_grey, cuda_temp, this->image_width, this->image_height, SOBELDISPLAY_SOBELSHARED, 1.0f );
}

template <class Format >
void Cuda_Image_Hand< Format > :: or_Operation()
{
	orOperation( cuda_temp, cuda_final , this->image_width, this->image_height ) ;
}

template <class Format >
unsigned char  *  Cuda_Image_Hand< Format > ::  get_Processed_Image_Pointer()
{
	convert_To_greyScale() ;
	sobel_Filter() ;
	or_Operation() ;

	return cuda_final ;
}

#endif
