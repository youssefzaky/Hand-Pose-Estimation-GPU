#ifndef _CUDA_MATCH_KERNEL_H
#define _CUDA_MATCH_KERNEL_H

extern "C" unsigned int findMatch( unsigned char* d_input,
                           unsigned char* d_ref,
                           int imgNum,
                           int rows,
                           int cols
                         );

#endif
