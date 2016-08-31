#include "Hand_Match.h"
#include "cudaMatch.h"

Hand_Match::Hand_Match( HandPose2_::HandPose2 & handpose,
		                   unsigned int image_width,
		                   unsigned int image_height,
		                   unsigned int num_Hyp_Images )
:generator( handpose ) ,
 temporary( image_width, image_height ),
 image_height( image_height ),
 image_width( image_width ),
 num_Hyp_Images( num_Hyp_Images ),
 is_Reference_Set( false ) ,
 is_Hypothesis_Set( false )
{
	checkCudaErrors( cudaMalloc( (void **) &reference,
			                       image_width * image_height * 4 ) );

	checkCudaErrors( cudaMalloc( (void **) &hypothesis_Images,
			                       image_width * image_height * 4 * num_Hyp_Images ) );
}

Hand_Match::~Hand_Match()
{
	checkCudaErrors( cudaFree(reference) );
	checkCudaErrors( cudaFree(hypothesis_Images) );
}

void Hand_Match::set_Reference_Image( std::vector<float> & param )
{
	generator.generate_From_Parameters( param, temporary ) ;

	checkCudaErrors(cudaMemcpy( reference ,
			                     temporary.get_Processed_Image_Pointer(),
			                     image_width * image_height,
			                     cudaMemcpyDeviceToDevice) );

	is_Reference_Set = true ;

}

void Hand_Match::set_Hypothesis_Images( std::vector< std::vector< float > > & param  )
{
	std::vector< std::vector< float > >::iterator iterator;

	for(iterator = param.begin();
	     iterator != param.end();
	     iterator++)
	{
	   generator.generate_From_Parameters( *iterator , temporary ) ;
	   int offset = image_height * image_width ;

	   checkCudaErrors(cudaMemcpy( reference + offset,
			                         temporary.get_Processed_Image_Pointer(),
			                         image_width * image_height,
			                         cudaMemcpyDeviceToDevice) );

//	   /// DISPLAY OUTPUT ////////
//	   	unsigned char * h_test = (unsigned char *) malloc(image_width * image_height);
//	   	checkCudaErrors(cudaMemcpy(h_test, temporary.get_Processed_Image_Pointer(), image_width*image_height, cudaMemcpyDeviceToHost) );
//
//	   	cv::Mat image(image_height, image_width, CV_8UC1, h_test);
//	   	cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
//	   	cv::imshow( "Display window", image );
//	   	cv::waitKey(0);
//	   	free (h_test);
//	   	////////////////

	}

	is_Hypothesis_Set = true ;
}

unsigned int Hand_Match::find_Top_N_Matches( )
{
	return findMatch( reference , hypothesis_Images , num_Hyp_Images, image_height, image_width) ;
}
