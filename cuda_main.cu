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


void runFindMax(int dimGrid, int dimBlock, int* d_reduction_index, float* d_reduction_value, int set, int first, int data_size1)
{
	switch (dimBlock)
	{
     case 512:
        reduce6<512><<< dimGrid, dimBlock >>>(d_reduction_index, d_reduction_value, set, first, data_size1); break;
     case 256:
        reduce6<256><<< dimGrid, dimBlock >>>(d_reduction_index, d_reduction_value, set, first, data_size1); break;
     case 128:
        reduce6<128><<< dimGrid, dimBlock >>>(d_reduction_index, d_reduction_value, set, first, data_size1); break;
     case 64:
        reduce6< 64><<< dimGrid, dimBlock >>>(d_reduction_index, d_reduction_value, set, first, data_size1); break;
     case 32:
        reduce6< 32><<< dimGrid, dimBlock >>>(d_reduction_index, d_reduction_value, set, first, data_size1); break;
     case 16:
        reduce6< 16><<< dimGrid, dimBlock >>>(d_reduction_index, d_reduction_value, set, first, data_size1); break;
     case 8:
        reduce6< 8><<< dimGrid, dimBlock >>>(d_reduction_index, d_reduction_value, set, first, data_size1); break;
     case 4:
        reduce6< 4><<< dimGrid, dimBlock >>>(d_reduction_index, d_reduction_value, set, first, data_size1); break;
     case 2:
        reduce6< 2><<< dimGrid, dimBlock >>>(d_reduction_index, d_reduction_value, set, first, data_size1); break;
     case 1:
        reduce6< 1><<< dimGrid, dimBlock >>>(d_reduction_index, d_reduction_value, set, first, data_size1); break;
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

	// find max power of 2 that is smaller than prob.l
	int reduction_length[2];
	float *d_reduction_value[2];
	int *d_reduction_index[2];
	int reduction_nthreads[2];
	reduction_nthreads[0] = 256;
	reduction_nthreads[1] = 256; //todo: reduce.pdf sagt am besten 128

	for(i=0;i<2;i++){
		reduction_length[i] = (( prob[i].l - 1) / (2 * reduction_nthreads[i])) + 1;

		while(reduction_nthreads[i] > prob[i].l)
		{
			reduction_nthreads[i] /= 2;
		}

		//reduction_length[i] = 128; //test

		cutilSafeCall(cudaMalloc((void**) &d_reduction_value[i], reduction_length[i] * sizeof(float)));
		cutilSafeCall(cudaMalloc((void**) &d_reduction_index[i], reduction_length[i] * sizeof(int)));
	}


	printf(" rl: %d, thr: %d prob: %d and rl: %d thr: %d prob: %d \n", reduction_length[0], reduction_nthreads[0], prob[0].l, reduction_length[1], reduction_nthreads[1], prob[1].l);

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
	for(int i = 0; i<9; i++)
	{
		cuda_kernel_lambda<<<1, 1>>>();
		cudaThreadSynchronize();
		// check if kernel execution generated and error
		cutilCheckMsg("Kernel execution failed");
		cuda_kernel_computekernels<<<nblocks, nthreads>>>();

		cudaThreadSynchronize();
		// check if kernel execution generated and error
		cutilCheckMsg("Kernel execution failed");
//		cuda_kernel_distance<<<1, 1>>>();

		float *d_temp1;
		cutilSafeCall(cudaMalloc((void**) &d_temp1, 1024 * sizeof(float)));

		int *d_temp2;
		cutilSafeCall(cudaMalloc((void**) &d_temp2, 1024 * sizeof(int)));

		runFindMax(reduction_length[0], reduction_nthreads[0], d_reduction_index[0], d_reduction_value[0], 0, 1,0);
		runFindMax(reduction_length[1], reduction_nthreads[1], d_reduction_index[1], d_reduction_value[1], 1, 1,0);


		//reduce6<256><<<reduction_length[1], reduction_nthreads[1]>>>(d_reduction_index[1], d_reduction_value[1], 1, 1, 0);
		cudaThreadSynchronize();

		int tempSize = 512;
		float h_temp1[tempSize];
		cutilSafeCall(cudaMemcpy( &h_temp1, d_temp1, sizeof(float) * tempSize, cudaMemcpyDeviceToHost));
		int h_temp2[tempSize];
		cutilSafeCall(cudaMemcpy( &h_temp2, d_temp2, sizeof(int) * tempSize, cudaMemcpyDeviceToHost));

		int l;
		for(l=0;l<3;l++)
		{
			//printf("%d - %f (%d)\n", l, h_temp1[l], h_temp2[l]);
		}


		//reduce6<2><<<1, 2>>>(d_reduction_index[1], d_reduction_value[1], 1, 0, 3);
//		runFindMax(1, 2, d_reduction_index[1], d_reduction_value[1], 1, 0);

		float values[4];
		int indizes[4];
		
		cutilSafeCall(cudaMemcpy( &values, d_reduction_value[1], sizeof(float) * 4, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy( &indizes, d_reduction_index[1], sizeof(int) * 4, cudaMemcpyDeviceToHost));
		
		int k;
		for(k=0;k<1;k++)
		{
			//printf(" %d: %f (%d) \n", k, values[k], indizes[k]);
		}

		int temp_reduction_length[2];
		int temp_reduction_nthreads[2];

		for(k=0;k<2;k++) //todo: diesen hack entfernen
		{
			temp_reduction_length[k] = reduction_length[k];
			temp_reduction_nthreads[k] = reduction_nthreads[k];
		}

		int counter = 0;
		while( (reduction_length[0] != 1) |  (reduction_length[1] != 1) )
		{
			counter++;
			int j;
			for(j=0;j<2;j++)
			{
			  if(reduction_length[j] != 1) {
				int data_size = reduction_length[j];
				
				reduction_nthreads[j] = 256;
				reduction_length[j] = (( data_size - 1) / (2 * reduction_nthreads[j])) + 1;

				while(reduction_nthreads[j] > data_size)
				{
					reduction_nthreads[j] /= 2;
				}

				runFindMax(reduction_length[j], reduction_nthreads[j], d_reduction_index[j], d_reduction_value[j], j, 0, data_size);
				//printf(" search max, iteration %d, teil %d,  %d-%d \n", counter, j, reduction_length[j], reduction_nthreads[j]);
			  }
			}
		cudaThreadSynchronize();
		}

		for(k=0;k<2;k++)
		{
			reduction_length[k] = temp_reduction_length[k];
			reduction_nthreads[k] = temp_reduction_nthreads[k];
		}

		
/*		float reduction[2];
		int reduction_idx[2];
		cutilSafeCall(cudaMemcpy( &reduction, d_reduction_value[0], sizeof(float) * 1, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy( &reduction_idx, d_reduction_index[0], sizeof(int), cudaMemcpyDeviceToHost));
		int j;
		for(j=0;j<1;j++)
			printf(" %d : %f - %d \n", j, reduction[j], reduction_idx[j]);
*/
		float max1[1];
		float max2[1];	
		int max1_idx[1];
		int max2_idx[1];	
		cutilSafeCall(cudaMemcpy( &max1, d_reduction_value[0], sizeof(float), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy( &max2, d_reduction_value[1], sizeof(float), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy( &max1_idx, d_reduction_index[0], sizeof(int), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy( &max2_idx, d_reduction_index[1], sizeof(int), cudaMemcpyDeviceToHost));
	
		printf("max[0] = %d (%f)   max[1] = %d (%f)  \n", max1_idx[0], max1[0], max2_idx[0], max2[0]);

		//printf("aufruf zweite runde\n");

		//runFindMax(1, reduction_length[0] / 2, d_reduction_index[0], d_reduction_value[0], 0, 0);
		//runFindMax(1, reduction_length[1] / 2, d_reduction_index[1], d_reduction_value[1], 1, 0);

		cudaThreadSynchronize();

		//printf("d_reduction_index[0][0] = %d \n", d_reduction_index[0][0]);
		//printf("d_reduction_index[1][0] = %d \n", d_reduction_index[1][0]);
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
