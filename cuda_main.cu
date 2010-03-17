/** 
* @file cuda_main.cu 
* 
* @brief Eine kurze Beschreibung der Datei - was enthält sie, wozu ist sie da, ...
* 
**/ 

#include <stdio.h>
#include <cuda_runtime.h>

#include "globals.h"

#include "cuda_kernel.cu"

#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() Runtime API error in file <%s>, line %i : %s.\n",
                file, line, cudaGetErrorString( err) );
        exit(-1);
    }
}


/** 
* @brief Eine kurze Beschreibung der Funktion 
* 
* Eine detailliertere Funktionsbeschreibung 
*
* @todo more work that needs to be done
*/ 
extern "C" void run_cuda_kernel()
{

	printf(" %d \n", prob[0].l);

//  float *h_x, *d_x;
  int   nblocks, nthreads;//, nsize, n; 

  // set number of blocks, and threads per block

//  nblocks  = (prob[0].l + prob[1].l + 1) / 256;
//  nthreads = 256;
  nblocks  = 1;
  nthreads = prob[0].l + prob[1].l;
//  nsize    = nblocks*nthreads ;

  // allocate memory for array


	float* h_data[2];
	float* d_data[2];
	float* d_weights[2];
	float* h_weights[2];


	int i;
	for(i=0;i<2;i++) {

		int size_of_data = sizeof(float) * prob[i].l * max_index; //sizeof(float) * prob[0].l * max_index;



		float* temp;
		//h_data[i] =  (float*) malloc(sizeof(float) * prob[i].l * max_index);
		float *h_data_temp =  (float*) malloc(size_of_data);
		//printf(" d_data[i] = %d ", temp);
		//cutilSafeCall(cudaMalloc((void **)&(d_data[i]), sizeof(float) * prob[i].l * max_index));
		cutilSafeCall(cudaMalloc((void **)& temp, size_of_data ));

		//printf(" danach: d_data[i] = %d  \n", temp);
		cudaMalloc((void **)&d_weights[i], sizeof(float) * prob[i].l);
		h_weights[i] = (float*) malloc(sizeof(float) * prob[i].l);
/*
* @todo : hier mal ein test-todo
*/																			
		//memset( (void**) h_data[i] , 0, sizeof(float) * prob[i].l * max_index); // 	 \todo : : int 0 == float 0?
		memset( (void**) h_data_temp , 0, size_of_data); // 	 \todo : : int 0 == float 0?
		int j;
		for(j=0;j<prob[i].l; j++)
		{
			struct svm_node *p = prob[i].x[j];
			while(p->index != -1)
			{
				h_data_temp[ max_index * j + (p->index - 1) ] = p->value;
				p++;
			}
		}		

		// copy host memory to device

		cutilSafeCall(cudaMemcpy(temp,h_data_temp,size_of_data,cudaMemcpyHostToDevice));
		h_data[i] = h_data_temp;
		d_data[i] = temp;
	}

	int j,k;
	for(i=0;i<2;i++)
	for(j=0;j<prob[i].l;j++)
	for(k=0;k<max_index;k++)
	{
		//printf(" i = %d,  j = %d  k = %d  value = %f \n ", i, j, k, h_data[i][ max_index * j + k ]);
	}

	for(i=0;i<2;i++)
	for(j=0;j<prob[i].l;j++)
	for(k=0;k<max_index;k++)
	{
		//printf(" device:  i = %d,  j = %d  k = %d  value = %f ( index = %d )\n ", i, j, k, d_data[i][ max_index * j + k ], max_index * j + k );
	}

	float *d_dot_xi_x, *d_dot_yi_x;
	float *d_dot_xi_y, *d_dot_yi_y;

	cudaMalloc((void**) &d_dot_xi_x, prob[0].l * sizeof(float));
	cudaMalloc((void**) &d_dot_yi_x, prob[0].l * sizeof(float));

	cudaMalloc((void**) &d_dot_xi_y, prob[1].l * sizeof(float));
	cudaMalloc((void**) &d_dot_yi_y, prob[1].l * sizeof(float)); // 	 \todo : : prob.l durch was schoeners  ersetzen?

	//cache parameters
	nr_of_cache_entries = 4;
	nr_of_elements = prob[0].l + prob[1].l;

    // allocate memory
    int *d_look_up_table;
	cudaMalloc((void**) &d_look_up_table, sizeof(int) * nr_of_elements );
    cudaMemset( d_look_up_table, -1, sizeof(int) * nr_of_elements );

    int *d_reverse_look_up_table;
	cudaMalloc( (void**) &d_reverse_look_up_table, sizeof(int) * nr_of_cache_entries );
    cudaMemset( d_reverse_look_up_table, -1, sizeof(int) * nr_of_cache_entries );

    int *d_circular_array;
	cudaMalloc( (void**) &d_circular_array, sizeof(int) * nr_of_cache_entries );
    cudaMemset( d_circular_array, -1, sizeof(int) * nr_of_cache_entries );

    float* d_data_cache;
	cudaMalloc( (void**) &d_data_cache, sizeof( float* ) * nr_of_cache_entries * nr_of_elements);
    
    int temp_size = 2;//prob[1].l ;
    float* d_temp;
	cudaMalloc( (void**) &d_temp, sizeof( float* ) * temp_size);




  // execute kernel

  cuda_kernel_init<<<nblocks,nthreads>>>(d_data[0], d_data[1], max_index, prob[0].l, prob[1].l, 
									d_weights[0], d_weights[1],
									d_dot_xi_x, d_dot_yi_x, d_dot_xi_y, d_dot_yi_y,
									nr_of_cache_entries, nr_of_elements,
									d_look_up_table, d_reverse_look_up_table, d_circular_array, d_data_cache, d_temp);

  // copy back results and print them 

  float h_temp[temp_size];

  cudaMemcpy( &h_temp, d_temp, sizeof(float) * temp_size, cudaMemcpyDeviceToHost);

  cudaMemcpy(h_weights[0],d_weights[0],sizeof(float) * prob[0].l,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_weights[1],d_weights[1],sizeof(float) * prob[1].l,cudaMemcpyDeviceToHost);

	for(int i=0;i<2;i++){
		printf("\n h_weights[ %d ]:", i);
		for(int j=0; j<prob[i].l; j++)
		{	
			if(h_weights[i][j] != 0.0)
				printf(" %d:%f \n", j, h_weights[i][j]);
		}
	}
//  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // free memory 

//  cudaFree(d_x);
//  free(h_x);

  //printf(" temp = %f  \n ", h_temp);
  for(int i = 0; i<temp_size; i++)
  {
    printf(" %d : %f \n", i, h_temp[i]);
  }

}
