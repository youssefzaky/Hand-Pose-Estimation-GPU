/* This class encapsulates registering a GL texture with Cuda.
 *

*/

#ifndef CUDA_GL_TEXTURE_MANAGER_H
#define CUDA_GL_TEXTURE_MANAGER_H

#include "Cuda_Image.h"
#include <cuda_gl_interop.h>

class Cuda_GL_Texture_Manager
{

public:

	Cuda_GL_Texture_Manager( GLuint gl_Texture );
	struct cudaGraphicsResource *  get_Texture_Resource( ) ;

private:

    struct cudaGraphicsResource * cuda_tex_resource;

};

#endif
