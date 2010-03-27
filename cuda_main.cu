/**
 * @file cuda_main.cu
 *
 * @brief Eine kurze Beschreibung der Datei - was enthlt sie, wozu ist sie da, ...
 *
 **/

#include <stdio.h>
#include <cuda_runtime.h>

#include "globals.h"

#include "cuda_kernel.cu"

#define cutilCheckMsg(msg)           __cutilCheckMsg     (msg, __FILE__, __LINE__)

inline void __cutilCheckMsg( const char *errorMessage, const char *file, const int line )
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err)
	{
		fprintf(stderr, "cutilCheckMsg() CUTIL CUDA error: %s in file <%s>, line %i : %s.\n",
			errorMessage, file, line, cudaGetErrorString( err) );
		exit(-1);
	}
	#ifdef _DEBUG
	err = cudaThreadSynchronize();
	if( cudaSuccess != err)
	{
		fprintf(stderr, "cutilCheckMsg cudaThreadSynchronize error: %s in file <%s>, line %i : %s.\n",
			errorMessage, file, line, cudaGetErrorString( err) );
		exit(-1);
	}
	#endif
}


#  define CUDA_SAFE_CALL_NO_SYNC( call) { \
		cudaError err = call; \
		if( cudaSuccess != err) \
		{ \
			fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
			__FILE__, __LINE__, cudaGetErrorString( err) ); \
			exit(EXIT_FAILURE); \
		} \
	}

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);

#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
	if( cudaSuccess != err)
	{
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

	printf(" number of vectors in class 1 = %d and in class 2 = %d \n", prob[0].l, prob[1].l);

	int   nblocks, nthreads;	 //, nsize, n;

	// set number of blocks, and threads per block

	nblocks  = ((prob[0].l + prob[1].l) / 256) + 1;
	nthreads = 256;
	printf("blocks: %d \n", nblocks);

	// allocate memory for array

	float* d_data[2];
	float* d_weights[2];
	float* h_weights[2];

	int i;
	for(i=0;i<2;i++)
	{
		int size_of_data = sizeof(float) * prob[i].l * max_index;

		float* temp;
		float *h_data_temp =  (float*) malloc(size_of_data);
		cutilSafeCall(cudaMalloc((void **)& temp, size_of_data ));

		cutilSafeCall(cudaMalloc((void **)&d_weights[i], sizeof(float) * prob[i].l));
								 //todo: speicher wieder freigeben.
		h_weights[i] = (float*) malloc(sizeof(float) * prob[i].l);
		/*
		 * @todo : hier mal ein test-todo
		 */
		memset( (void**) h_data_temp , 0, size_of_data);
								 // 	 \todo : : int 0 == float 0?
		memset( (void**) h_weights[i] , 10, sizeof(float) * prob[i].l);
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
		d_data[i] = temp;
	}

	float *d_dot_xi_x, *d_dot_yi_x;
	float *d_dot_xi_y, *d_dot_yi_y;

	cutilSafeCall(cudaMalloc((void**) &d_dot_xi_x, prob[0].l * sizeof(float)));
	cutilSafeCall(cudaMalloc((void**) &d_dot_yi_x, prob[0].l * sizeof(float)));

	cutilSafeCall(cudaMalloc((void**) &d_dot_xi_y, prob[1].l * sizeof(float)));
								 // 	 \todo : : prob.l durch was schoeners  ersetzen?
	cutilSafeCall(cudaMalloc((void**) &d_dot_yi_y, prob[1].l * sizeof(float)));

	//cache parameters
	nr_of_cache_entries = 4;
	nr_of_elements = prob[0].l + prob[1].l;

	// allocate memory
	int *d_look_up_table;
	cutilSafeCall(cudaMalloc((void**) &d_look_up_table, sizeof(int) * nr_of_elements ));
	cutilSafeCall(cudaMemset( d_look_up_table, -1, sizeof(int) * nr_of_elements ));

	int *d_reverse_look_up_table;
	cutilSafeCall(cudaMalloc( (void**) &d_reverse_look_up_table, sizeof(int) * nr_of_cache_entries ));
	cutilSafeCall(cudaMemset( d_reverse_look_up_table, -1, sizeof(int) * nr_of_cache_entries ));

	int *d_circular_array;
	cutilSafeCall(cudaMalloc( (void**) &d_circular_array, sizeof(int) * nr_of_cache_entries ));
	cutilSafeCall(cudaMemset( d_circular_array, -1, sizeof(int) * nr_of_cache_entries ));

	float* d_data_cache;
	cutilSafeCall(cudaMalloc( (void**) &d_data_cache, sizeof( float* ) * nr_of_cache_entries * nr_of_elements));

	int temp_size = 2;			 //prob[1].l ;
	float* d_temp;
	cutilSafeCall(cudaMalloc( (void**) &d_temp, sizeof( float* ) * temp_size));

	// execute kernel

	cuda_kernel_init_pointer<<<1,1>>>(d_data[0], d_data[1], max_index, prob[0].l, prob[1].l,
		d_weights[0], d_weights[1],
		d_dot_xi_x, d_dot_yi_x, d_dot_xi_y, d_dot_yi_y,
		nr_of_cache_entries, nr_of_elements,
		d_look_up_table, d_reverse_look_up_table, d_circular_array, d_data_cache, d_temp);

	cudaThreadSynchronize();
	cuda_kernel_init_kernel<<<nblocks, nthreads>>>();
	cudaThreadSynchronize();
	cuda_kernel_init_findmax<<<1, 1>>>();

	cudaThreadSynchronize();
	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed");
	for(int i = 0; i<10; i++)
	{
		cuda_kernel_lambda<<<1, 1>>>();
		cudaThreadSynchronize();
		// check if kernel execution generated and error
		cutilCheckMsg("Kernel execution failed");
		cuda_kernel_computekernels<<<nblocks, nthreads>>>();

		cudaThreadSynchronize();
		// check if kernel execution generated and error
		cutilCheckMsg("Kernel execution failed");
		cuda_kernel_distance<<<1, 1>>>();
		cudaThreadSynchronize();
		// check if kernel execution generated and error
		cutilCheckMsg("Kernel execution failed");
	}

	// copy back results and print them

	float h_temp[temp_size];

	cutilSafeCall(cudaMemcpy( &h_temp, d_temp, sizeof(float) * temp_size, cudaMemcpyDeviceToHost));

	cutilSafeCall(cudaMemcpy(h_weights[0],d_weights[0],sizeof(float) * prob[0].l,cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_weights[1],d_weights[1],sizeof(float) * prob[1].l,cudaMemcpyDeviceToHost));

	for(int i=0;i<2;i++)
	{
		printf("\n h_weights[ %d ]:", i);
		for(int j=0; j<prob[i].l; j++)
		{
			if(h_weights[i][j] != 0.0)
				printf(" %d:%f \n", j, h_weights[i][j]);
		}
	}

	// free memory

	//  cudaFree(d_x);
	//  free(h_x);

	//printf(" temp = %f  \n ", h_temp);
	//for(int i = 0; i<temp_size; i++)
	//{
	//  printf(" %d : %f \n", i, h_temp[i]);
	//}

}
