#include <stdio.h>
#include <cuda_runtime.h>

#include "globals.h"

#include "cuda_kernel.cu"

extern "C" void run_cuda_kernel()
{
  float *h_x, *d_x;
  int   nblocks, nthreads, nsize, n; 

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  h_x = (float *)malloc(nsize*sizeof(float));
  cudaMalloc((void **)&d_x, nsize*sizeof(float));

  // execute kernel

  cuda_kernel<<<nblocks,nthreads>>>(d_x);

  // copy back results and print them out

  cudaMemcpy(h_x,d_x,nsize*sizeof(float),cudaMemcpyDeviceToHost);

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // free memory 

  cudaFree(d_x);
  free(h_x);

}
