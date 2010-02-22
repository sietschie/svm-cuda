#ifndef _CUDA_KERNEL_H_
#define _CUDA_KERNEL_H_

__global__ void
cuda_kernel( float* g_data )
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	g_data[tid] = tid; 
}

#endif // #ifndef _CUDA_KERNEL_H_
