#include "Image_Generator.h"

Image_Generator::Image_Generator( HandPose2_::HandPose2 & handpose )
:t_manager( handpose.getTextureID() ) , handpose( handpose )
{
}

void Image_Generator::generate_From_Parameters( std::vector<float> &param, Cuda_Image_Hand< uchar4 > & Image )
{
	{
		using namespace Ogre;

		Matrix3 mat;
		mat.FromEulerAnglesXYZ(Degree( param[ 20 ] ), Degree( param[ 21 ]  ), Degree( param[ 22 ] ));
		handpose.hand_node->setOrientation(mat);

		handpose.hand_node->setPosition( param[ 23 ]  , param[ 24 ]  , param[ 25 ] );
	}

	// set hand pose 2
	handpose.SetPose( param );

	handpose.Render();

	Image.get_From_Texture( t_manager.get_Texture_Resource() ) ;

}
