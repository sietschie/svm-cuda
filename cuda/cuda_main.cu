/**
 * @file cuda_main.cu
 *
 * @brief Eine kurze Beschreibung der Datei - was enthlt sie, wozu ist sie da, ...
 *
 **/

#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

#include "globals.h"

#include "cuda_kernel.cu"
#include "cuda_utils.h"

const int maxNumThreadsPerBlock = 128;

// variables for reduction function
int reduction_length[2];
float *d_reduction_value[2];
int *d_reduction_index[2];
int reduction_nthreads[2];

void reduction_startKernel(int dimGrid, int dimBlock, int* d_reduction_index, float* d_reduction_value, int set, int first, int data_size1)
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


void reduction_init()
{
	reduction_nthreads[0] = maxNumThreadsPerBlock;
	reduction_nthreads[1] = maxNumThreadsPerBlock;

	int i;
	for(i=0;i<2;i++)
	{
		reduction_length[i] = (( prob[i].l - 1) / (2 * reduction_nthreads[i])) + 1;

		while(reduction_nthreads[i] > prob[i].l)
		{
			reduction_nthreads[i] /= 2;
		}

		cutilSafeCall(cudaMalloc((void**) &d_reduction_value[i], reduction_length[i] * sizeof(float)));
		cutilSafeCall(cudaMalloc((void**) &d_reduction_index[i], reduction_length[i] * sizeof(int)));
	}
}


void reduction_findMaximum()
{
	reduction_startKernel(reduction_length[0], reduction_nthreads[0], d_reduction_index[0], d_reduction_value[0], 0, 1,0);
	reduction_startKernel(reduction_length[1], reduction_nthreads[1], d_reduction_index[1], d_reduction_value[1], 1, 1,0);

	cudaThreadSynchronize();

	int temp_reduction_length[2];
	int temp_reduction_nthreads[2];

	int k;
	for(k=0;k<2;k++)			 //todo: diesen hack entfernen
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
			if(reduction_length[j] != 1)
			{
				int data_size = reduction_length[j];

				reduction_nthreads[j] = 256;
				reduction_length[j] = (( data_size - 1) / (2 * reduction_nthreads[j])) + 1;

				while(reduction_nthreads[j] > data_size)
				{
					reduction_nthreads[j] /= 2;
				}

				reduction_startKernel(reduction_length[j], reduction_nthreads[j], d_reduction_index[j], d_reduction_value[j], j, 0, data_size);
			}
		}
		cudaThreadSynchronize();
	}

	for(k=0;k<2;k++)
	{
		reduction_length[k] = temp_reduction_length[k];
		reduction_nthreads[k] = temp_reduction_nthreads[k];
	}
}


/**
 * @brief Eine kurze Beschreibung der Funktion
 *
 * Eine detailliertere Funktionsbeschreibung
 *
 * @todo more work that needs to be done
 */
extern "C" void run_cuda_kernel(struct svm_parameter param,	float** weights, float *rho)
{
	int   nblocks, nthreads;	 //, nsize, n;

	// set number of blocks, and threads per block

	nblocks  = ((prob[0].l + prob[1].l - 1) / maxNumThreadsPerBlock) + 1;
	nthreads = maxNumThreadsPerBlock;
	if(param.verbosity == 2)
		printf("blocks: %d \n", nblocks);

	// allocate memory for array

	float* d_data[2];
	float* d_weights[2];
	float* h_weights[2];
	float h_rho[1];

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
		memset( (void**) h_data_temp , 0, size_of_data);
								 // 	 \todo : : int 0 == float 0?

		// copy data from data structure to plain array
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
		free(h_data_temp);
	}

	float *d_dot_xi_x, *d_dot_yi_x;
	float *d_dot_xi_y, *d_dot_yi_y;
	float *d_distance, *d_rho;
	float *d_dot_same[2];

	cutilSafeCall(cudaMalloc((void**) &d_distance, sizeof(float)));
	cutilSafeCall(cudaMalloc((void**) &d_rho, sizeof(float)));
	cutilSafeCall(cudaMalloc((void**) &d_dot_xi_x, prob[0].l * sizeof(float)));
	cutilSafeCall(cudaMalloc((void**) &d_dot_yi_x, prob[0].l * sizeof(float)));
	cutilSafeCall(cudaMalloc((void**) &d_dot_same[0], prob[0].l * sizeof(float)));

	cutilSafeCall(cudaMalloc((void**) &d_dot_xi_y, prob[1].l * sizeof(float)));
								 // 	 \todo : : prob.l durch was schoeners  ersetzen?
	cutilSafeCall(cudaMalloc((void**) &d_dot_yi_y, prob[1].l * sizeof(float)));
	cutilSafeCall(cudaMalloc((void**) &d_dot_same[1], prob[1].l * sizeof(float)));

	reduction_init();

	//cache parameters
	nr_of_cache_entries = param.cache_size;
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

	// execute kernels

	cuda_kernel_init_pointer<<<1,1>>>(d_data[0], d_data[1], max_index, prob[0].l, prob[1].l,
		d_weights[0], d_weights[1],
		d_dot_xi_x, d_dot_yi_x, d_dot_xi_y, d_dot_yi_y, 
		d_dot_same[0], d_dot_same[1], d_distance, d_rho, param);

	cuda_cache_init<<<1,1>>>(nr_of_cache_entries, nr_of_elements,
		d_look_up_table, d_reverse_look_up_table, d_circular_array, d_data_cache);


	clock_t start, finish;
	start = clock();

	cudaThreadSynchronize();
	cuda_kernel_init_kernel<<<nblocks, nthreads>>>();
	cudaThreadSynchronize();
	cuda_kernel_init_findmax<<<1, 1>>>();

	cudaThreadSynchronize();
	reduction_findMaximum();

	cudaThreadSynchronize();
	// check if kernel execution generated and error
	cutilCheckMsg("Kernel execution failed");
	for(int i = 0; i<param.maximum_iterations; i++)
	{
		cuda_kernel_lambda<<<1, 1>>>();
		cudaThreadSynchronize();
								 //todo: nur soviele kernel starten wie wirklich noetig
		cutilCheckMsg("Kernel execution failed");
		cuda_kernel_updateWeights<<< nblocks, nthreads >>>();
		cudaThreadSynchronize();
		cutilCheckMsg("Kernel execution failed");
		cuda_cache_update<<<1,1>>>();
		cudaThreadSynchronize();
		cutilCheckMsg("Kernel execution failed");
		cuda_kernel_computekernels_cache<<<nblocks, nthreads>>>();
		cudaThreadSynchronize();
		// check if kernel execution generated and error
		cutilCheckMsg("Kernel execution failed");

		cudaThreadSynchronize();
		// check if kernel execution generated and error
		cutilCheckMsg("Kernel execution failed");

		reduction_findMaximum();

		float max1[1];
		float max2[1];
		int max1_idx[1];
		int max2_idx[1];
		cutilSafeCall(cudaMemcpy( &max1, d_reduction_value[0], sizeof(float), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy( &max2, d_reduction_value[1], sizeof(float), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy( &max1_idx, d_reduction_index[0], sizeof(int), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy( &max2_idx, d_reduction_index[1], sizeof(int), cudaMemcpyDeviceToHost));

		if(param.verbosity == 2) 
		{
			printf("max[0] = %d (%f)   max[1] = %d (%f)  \n", max1_idx[0], max1[0], max2_idx[0], max2[0]);
		}

		float h_distance[1];
		cutilSafeCall(cudaMemcpy( &h_distance, d_distance, sizeof(float), cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaMemcpy( &h_rho, d_rho, sizeof(float), cudaMemcpyDeviceToHost));

		cudaThreadSynchronize();
		// check if kernel execution generated and error
		cutilCheckMsg("Kernel execution failed");
		
        double adg = max1[0] + max2[0];
        double rdg_nenner = *h_distance - adg;
        double rdg;

        if (rdg_nenner <= 0)
        {
            rdg = HUGE_VAL;
        }
        else
        {
            rdg = adg / rdg_nenner;
        }

		if( param.verbosity >= 1 )
		{
			printf("iter = %d ", i);
			printf("dist = %e " , *h_distance);
			//printf("rho = %e " , *h_rho);
			printf("adg = %e " , adg);
			printf("rdg = %e \n", rdg);
		}
		
		if( rdg < param.eps )
			break;
	}

	finish = clock();

	double time = ((double)(finish - start))/CLOCKS_PER_SEC;
	printf("time: %f \n", time);

	// copy results back and print them

	cutilSafeCall(cudaMemcpy(h_weights[0],d_weights[0],sizeof(float) * prob[0].l,cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_weights[1],d_weights[1],sizeof(float) * prob[1].l,cudaMemcpyDeviceToHost));

	if(param.verbosity == 2)
	{
		for(int i=0;i<2;i++)
		{
			printf("\n h_weights[ %d ]:", i);
			for(int j=0; j<prob[i].l; j++)
			{
				if(h_weights[i][j] != 0.0)
					printf(" %d:%f \n", j, h_weights[i][j]);
			}
		}
	}

	// free memory
	cudaFree(d_data[0]);
	cudaFree(d_data[1]);
	cudaFree(d_weights[0]);
	cudaFree(d_weights[1]);

	cudaFree(d_distance);
	cudaFree(d_rho);
	cudaFree(d_dot_xi_x);
	cudaFree(d_dot_yi_x);
	cudaFree(d_dot_same[0]);
	cudaFree(d_dot_xi_y);
	cudaFree(d_dot_yi_y);
	cudaFree(d_dot_same[1]);
	cudaFree(d_reduction_index[0]);
	cudaFree(d_reduction_value[0]);
	cudaFree(d_reduction_index[1]);
	cudaFree(d_reduction_value[1]);
	cudaFree(d_look_up_table);
	cudaFree(d_reverse_look_up_table);
	cudaFree(d_circular_array);
	cudaFree(d_data_cache);
	
	// values that will be returned
	*rho = *h_rho;
	weights[0] = h_weights[0];
	weights[1] = h_weights[1];
}
