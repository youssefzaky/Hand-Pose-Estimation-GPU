#ifndef HAND_MATCH_H
#define HAND_MATCH_H

#include "Image_Generator.h"
#include "Cuda_Image_Hand.h"
#include "HandPose2.hpp"

class Hand_Match
{

public:
	Hand_Match( HandPose2_::HandPose2 & handpose,
			     unsigned int image_width,
			     unsigned int image_height,
			     unsigned int num_Hyp_Images ) ;
	~Hand_Match();
    void set_Reference_Image( std::vector<float> & param ) ;
    void set_Hypothesis_Images( std::vector< std::vector< float > > & param ) ;
    unsigned int find_Top_N_Matches(  ) ;

private:
	Image_Generator generator ;
	Cuda_Image_Hand<uchar4> temporary ;
	unsigned char * reference ;
	unsigned char * hypothesis_Images ;
	unsigned int image_height ;
	unsigned int image_width ;
	unsigned int num_Hyp_Images ;
	bool is_Reference_Set ;
	bool is_Hypothesis_Set ;

};

#endif
