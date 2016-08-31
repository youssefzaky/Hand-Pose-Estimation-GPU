#ifndef IMAGE_GENERATOR_H
#define IMAGE_GENERATOR_H

#include "HandPose2.hpp"
#include "Cuda_Image_Hand.h"
#include "Cuda_GL_Texture_Manager.h"

class Image_Generator
{

public:
	Image_Generator( HandPose2_::HandPose2 & handpose ) ;
	void generate_From_Parameters( std::vector<float> &param , Cuda_Image_Hand< uchar4 > & Image ) ;

private:
	HandPose2_::HandPose2  & handpose ;
	Cuda_GL_Texture_Manager  t_manager ;

};

#endif
