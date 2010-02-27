#include <stdio.h>
#include <cuda_runtime.h>

#include "globals.h"

#include "cuda_kernel.cu"

extern "C" void run_cuda_kernel()
{

	printf(" %d \n", prob[0].l);

//  float *h_x, *d_x;
  int   nblocks, nthreads;//, nsize, n; 

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
//  nsize    = nblocks*nthreads ;

  // allocate memory for array


	float* h_data[2];
	float* d_data[2];
	float* d_weights[2];


	int i;
	for(i=0;i<2;i++) {
		h_data[i] =  (float*) malloc(sizeof(float) * prob[i].l * max_index);
		cudaMalloc((void **)&d_data[i], sizeof(float) * prob[i].l * max_index);
		cudaMalloc((void **)&d_weights[i], sizeof(float) * prob[i].l);

		memset( (void**) h_data[i] , 0, sizeof(float) * prob[i].l * max_index); //todo: int 0 == float 0?
		int j;
		for(j=0;j<prob[i].l; j++)
		{
			struct svm_node *p = prob[i].x[j];
			while(p->index != -1)
			{
				h_data[i][ max_index * j + p->index ] = p->value;
				p++;
			}
		}		

		cudaMemcpy(h_data[i],d_data[i],sizeof(float) * prob[i].l * max_index,cudaMemcpyHostToDevice);
	}

	float *d_dot_xi_x, *d_dot_yi_x;
	float *d_dot_xi_y, *d_dot_yi_y;

	cudaMalloc((void**) &d_dot_xi_x, prob[0].l * sizeof(float));
	cudaMalloc((void**) &d_dot_yi_x, prob[0].l * sizeof(float));

	cudaMalloc((void**) &d_dot_xi_y, prob[1].l * sizeof(float));
	cudaMalloc((void**) &d_dot_yi_y, prob[1].l * sizeof(float)); //todo: prob.l durch was schoeners  ersetzen?


  // execute kernel

  cuda_kernel<<<nblocks,nthreads>>>(h_data[0], h_data[1], max_index, prob[0].l, prob[1].l, d_weights[0], d_weights[1],
									d_dot_xi_x, d_dot_yi_x, d_dot_xi_y, d_dot_yi_y);

  // copy back results and print them out

//  cudaMemcpy(h_x,d_x,nsize*sizeof(float),cudaMemcpyDeviceToHost);

//  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // free memory 

//  cudaFree(d_x);
//  free(h_x);

}
