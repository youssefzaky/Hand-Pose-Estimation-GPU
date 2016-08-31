#include "Cuda_GL_Texture_Manager.h"
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>

Cuda_GL_Texture_Manager::Cuda_GL_Texture_Manager( GLuint gl_Texture )

{
	checkCudaErrors(cudaGraphicsGLRegisterImage( &cuda_tex_resource, gl_Texture,
				      GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly ) );
}

struct cudaGraphicsResource  *  Cuda_GL_Texture_Manager::get_Texture_Resource( )
{
	return cuda_tex_resource ;
}



