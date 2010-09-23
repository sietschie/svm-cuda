#ifndef _CUDA_KERNEL_H_
#define _CUDA_KERNEL_H_

#include "globals.h"

__device__ float* g_data[2];
__device__ int maximum_index;
__device__ int data_size[2];
__device__ float* g_weights[2];
__device__ float lambda;
__device__ int max_p_index;
__device__ float max_p;
__device__ int max_q_index;
__device__ float max_q;
__device__ float dot_xi_yi;		 // <x_i, y_i >
__device__ float dot_xi_xi;		 // <x_i, x_i >
__device__ float dot_yi_yi;		 // <y_i, y_i >
__device__ float* distance;
__device__ float* rho;
__device__ float* dot_xi_x;
__device__ float* dot_yi_x;
__device__ float* dot_xi_y;
__device__ float* dot_yi_y;
__device__ float* dot_same[2];

__device__ float* get_element(int id, int set);

__device__ struct svm_parameter param;

__device__ float dot(float* px, float *py)
{
	float sum = 0.0;
	int i;
	for(i=0; i< maximum_index; i++)
	{
		sum += px[i] * py[i];
	}
	return sum;
}


inline float powi(float base, int times)
{
	float tmp = base, ret = 1.0;

	int t;
	for(t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}

__device__ float kernel_linear(int set1, int element1, int set2, int element2) //todo: als template implementieren
{
	float* px = &(g_data[set1][ element1 * maximum_index ]);
	float* py = &(g_data[set2][ element2 * maximum_index ]);

	float ret = dot(px, py );
	if(set1 == set2 && element1 == element2)
		ret += 1.0/param.C;
	return ret;
}

__device__ float power(float base, int exponent) { //todo: effizienter berechnen? (squaring, bitshifts)
	int i;
	float res = base; 
	for(i=0;i<exponent;i++) 
	{
		res  *= base;
	}
	return res;
}

__device__ float kernel_poly(int set1, int element1, int set2, int element2)
{
	float* px = &(g_data[set1][ element1 * maximum_index ]);
	float* py = &(g_data[set2][ element2 * maximum_index ]);

	float ret = power(param.gamma*dot(px, py )+param.coef0,param.degree);
	if(set1 == set2 && element1 == element2)
		ret += 1.0/param.C;
	return ret;
}

__device__ float kernel_rbf(int set1, int element1, int set2, int element2)
{
	float* px = &(g_data[set1][ element1 * maximum_index ]);
	float* py = &(g_data[set2][ element2 * maximum_index ]);

	float dots = ( dot(px, px)+
						dot(py, py)-2*
						dot(px, py)); //todo: dot(x,x) vorberechnen??
	float wgamma = -param.gamma*dots;
	float wexp = exp(wgamma);

	if(set1 == set2 && element1 == element2)
		wexp += 1.0/param.C;
	return wexp;

}

__device__ float kernel_sigmoid(int set1, int element1, int set2, int element2)
{
	float* px = &(g_data[set1][ element1 * maximum_index ]);
	float* py = &(g_data[set2][ element2 * maximum_index ]);

	float ret = tanh(param.gamma*dot(px, py)+param.coef0);
	if(set1 == set2 && element1 == element2)
		ret += 1.0/param.C;
	return ret;
}

__device__ float kernel(int set1, int element1, int set2, int element2)
{
	switch(param.kernel_type)
	{
		case POLY:
			return kernel_poly(set1, element1, set2, element2);
		case RBF:
			return kernel_rbf(set1, element1, set2, element2);
		case SIGMOID:
			return kernel_sigmoid(set1, element1, set2, element2);
//		case PRECOMPUTED:
//			return kernel_precomputed(set1, element1, set2, element2);
		case LINEAR:
		default:
			return kernel_linear(set1, element1, set2, element2);
	}
}

__device__ int find_max(int p, float *dot_yi_x, float* dot_xi_x, float dot_xi_yi, float dot_xi_xi, float *max_p)
{
	// find max
	int max_p_index = -1;
	*max_p = -1000000000.0;		 //todo: HUGE_VAL fuer Cuda finden

	int i;
	for (i=0; i<data_size[p]; i++)
	{
		float sum = dot_yi_x[i] - dot_xi_x[i] - dot_xi_yi + dot_xi_xi;
		if(sum > *max_p)
		{
			*max_p = sum;
			max_p_index = i;
		}
	}
	return max_p_index;
}


template <unsigned int blockSize>
								 //todo: set und first als template parameter?
__global__ void reduce6(int *g_data_index, float *g_data_value, unsigned int set, int first, int data_size1)
{
	__shared__ int sdata_index[blockSize];
	__shared__ float sdata_value[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;

	if(first == 1)
	{
		float value1;
		if(set == 0)
			value1 = dot_yi_x[i] - dot_xi_x[i] - dot_xi_yi + dot_xi_xi;
		else
			value1 = dot_xi_y[i] - dot_yi_y[i] - dot_xi_yi + dot_yi_yi;

		if(i < data_size[set])
		{
			if( i + blockSize < data_size[set])
			{
	
				float value2;
				if(set == 0)
					value2 = dot_yi_x[i+blockSize] - dot_xi_x[i+blockSize] - dot_xi_yi + dot_xi_xi;
				else
					value2 = dot_xi_y[i+blockSize] - dot_yi_y[i+blockSize] - dot_xi_yi + dot_yi_yi;
	
				if(value1 > value2)
				{
					sdata_value[tid] = value1;
					sdata_index[tid] = i;
				}
				else
				{
					sdata_value[tid] = value2;
					sdata_index[tid] = i+blockSize;
				}
			} else
			{
				sdata_value[tid] = value1;
				sdata_index[tid] = i;
			}
		} else {
			sdata_value[tid] = -100000000000.0; //todo: max_val suchen
			sdata_index[tid] = -1;
		}
	} else
	{
		if( i + blockSize < data_size1)
		{
			float value1 = g_data_value[i];
			float value2 = g_data_value[i+blockSize];
			if(value1>value2)
			{
				sdata_value[tid] = value1;
				sdata_index[tid] = g_data_index[i];
			}
			else
			{
				sdata_value[tid] = value2;
				sdata_index[tid] = g_data_index[i+blockSize];
			}
		} else
		{
			sdata_value[tid] = g_data_value[i];
			sdata_index[tid] = g_data_index[i];
		}
	}

	__syncthreads();
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			if(sdata_value[tid] < sdata_value[tid + 256])
			{
				sdata_value[tid] = sdata_value[tid + 256];
				sdata_index[tid] = sdata_index[tid + 256];
			}
		}
		__syncthreads();
	}
	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			if(sdata_value[tid] < sdata_value[tid + 128])
			{
				sdata_value[tid] = sdata_value[tid + 128];
				sdata_index[tid] = sdata_index[tid + 128];
			}
		}
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (tid < 64)
		{
			if(sdata_value[tid] < sdata_value[tid + 64])
			{
				sdata_value[tid] = sdata_value[tid + 64];
				sdata_index[tid] = sdata_index[tid + 64];
			}
		}
		__syncthreads();
	}

	#ifdef __DEVICE_EMULATION__
	if (blockSize >= 64)
	{
		if (tid < 32)
		{
			if(sdata_value[tid] < sdata_value[tid + 32])
			{
				sdata_value[tid] = sdata_value[tid + 32];
				sdata_index[tid] = sdata_index[tid + 32];
			}
		}
		__syncthreads();
	}
	if (blockSize >= 32)
	{
		if (tid < 16)
		{
			if(sdata_value[tid] < sdata_value[tid + 16])
			{
				sdata_value[tid] = sdata_value[tid + 16];
				sdata_index[tid] = sdata_index[tid + 16];
			}
		}
		__syncthreads();
	}
	if (blockSize >= 16)
	{
		if (tid < 8)
		{
			if(sdata_value[tid] < sdata_value[tid + 8])
			{
				sdata_value[tid] = sdata_value[tid + 8];
				sdata_index[tid] = sdata_index[tid + 8];
			}
		}
		__syncthreads();
	}
	if (blockSize >= 8)
	{
		if (tid < 4)
		{
			if(sdata_value[tid] < sdata_value[tid + 4])
			{
				sdata_value[tid] = sdata_value[tid + 4];
				sdata_index[tid] = sdata_index[tid + 4];
			}
		}
		__syncthreads();
	}
	if (blockSize >= 4)
	{
		if (tid < 2)
		{
			if(sdata_value[tid] < sdata_value[tid + 2])
			{
				sdata_value[tid] = sdata_value[tid + 2];
				sdata_index[tid] = sdata_index[tid + 2];
			}
		}
		__syncthreads();
	}
	if (blockSize >= 2)
	{
		if (tid < 1)
		{
			if(sdata_value[tid] < sdata_value[tid + 1])
			{
				sdata_value[tid] = sdata_value[tid + 1];
				sdata_index[tid] = sdata_index[tid + 1];
			}
		}
		__syncthreads();
	}
	#else
	if (tid < 32)
	{
		if (blockSize >= 64)
		{
			if(sdata_value[tid] < sdata_value[tid + 32])
			{
				sdata_value[tid] = sdata_value[tid + 32];
				sdata_index[tid] = sdata_index[tid + 32];
			}

		}
		if (blockSize >= 32)
		{
			if(sdata_value[tid] < sdata_value[tid + 16])
			{
				sdata_value[tid] = sdata_value[tid + 16];
				sdata_index[tid] = sdata_index[tid + 16];
			}

		}
		if (blockSize >= 16)
		{
			if(sdata_value[tid] < sdata_value[tid + 8])
			{
				sdata_value[tid] = sdata_value[tid + 8];
				sdata_index[tid] = sdata_index[tid + 8];
			}

		}
		if (blockSize >= 8)
		{
			if(sdata_value[tid] < sdata_value[tid + 4])
			{
				sdata_value[tid] = sdata_value[tid + 4];
				sdata_index[tid] = sdata_index[tid + 4];
			}

		}
		if (blockSize >= 4)
		{
			if(sdata_value[tid] < sdata_value[tid + 2])
			{
				sdata_value[tid] = sdata_value[tid + 2];
				sdata_index[tid] = sdata_index[tid + 2];
			}

		}
		if (blockSize >= 2)
		{
			if(sdata_value[tid] < sdata_value[tid + 1])
			{
				sdata_value[tid] = sdata_value[tid + 1];
				sdata_index[tid] = sdata_index[tid + 1];
			}

		}
	}
	#endif
	if (tid == 0)
	{
		g_data_index[blockIdx.x] = sdata_index[0];
		g_data_value[blockIdx.x] = sdata_value[0];
		if(blockIdx.x==0)
		{
			if(set == 0)
			{
				max_p = sdata_value[0];
				max_p_index = sdata_index[0];
			}
			else
			{
				max_q = sdata_value[0];
				max_q_index = sdata_index[0];
			}
		}
	}
}


__device__ float compute_zaehler(float dot_xi_yi, float* dot_yi_x, float* dot_xi_x, int p, int max_p_index )
{
	//todo: samevector, kann vorberechnet werden.
	float zaehler = dot_xi_yi - dot_yi_x[max_p_index] - dot_xi_x[max_p_index] + dot_same[p][max_p_index];//kernel(p,max_p_index, p, max_p_index);
	return zaehler;
}


__device__ float compute_nenner(float dot_xi_xi, float* dot_xi_x, int p, int max_p_index)
{
	float nenner = dot_xi_xi - 2* dot_xi_x[max_p_index] +  dot_same[p][max_p_index];//kernel(p, max_p_index, p, max_p_index);
	return nenner;
}


__device__ void add_to_weights(float* weights, float lambda, int index, int set)
{
	int i;
	for (i=0; i<data_size[set]; i++)
	{
		if (i!=index)
			weights[i] *= lambda;
		else
			weights[i] = weights[i]*lambda + (1.0 - lambda)*1;
	}
}


__device__ float update_xi_xi(float dot_xi_xi, float* dot_xi_x, int p, int max_p_index, float lambda)
{
	dot_xi_xi = lambda * lambda * dot_xi_xi
		+ 2 * lambda * (1.0 - lambda) * dot_xi_x[max_p_index]
								 //todo: skalarprodukt von vector mit sich selbst zwischenspeichern
		+ (1.0 - lambda)*(1.0 - lambda)*dot_same[p][max_p_index];//kernel(p, max_p_index, p ,max_p_index );
	return dot_xi_xi;
}


__device__ float update_xi_yi(float dot_xi_yi, float* dot_yi_x, int max_p_index, float lambda)
{
	dot_xi_yi = lambda * dot_xi_yi + (1.0 - lambda) * dot_yi_x[max_p_index];
	return dot_xi_yi;
}


__device__ void update_xi_x(float* dot_xi_x, int p, int p2, int max_p_index, float lambda, float* computed_kernels , int tid)
{
	if( (tid < data_size[0] && p2 == 0) || ( tid >= data_size[0] && p2 == 1 ) )
	{
		int offset = p2 * data_size[0];
								 //(p, max_p_index, p2, i);
		dot_xi_x[tid - offset]= dot_xi_x[tid - offset] * lambda + (1.0 - lambda) * computed_kernels[ tid  ];
	}
}


// cache anfang

__device__ int nr_of_cache_entries;
__device__ int nr_of_elements;

__device__ float* data;

__device__ int* look_up_table;	 // translates data id to cache position
								 // translates cache positions to id
__device__ int* reverse_look_up_table;
__device__ int* circular_array;	 // safes order in which cache pos was inserted

__device__ int ca_first;
__device__ int ca_last;
__device__ int ca_free_pos;		 // safes which pos has no yet been occupied

__device__ bool ca_cachemiss;

__global__ void cuda_cache_init(int g_nr_of_cache_entries, int g_nr_of_elements,
int *g_look_up_table, int* g_reverse_look_up_table, int* g_circular_array, float* g_data_cache)
{
	// cache initialisieren
	look_up_table = g_look_up_table;
	reverse_look_up_table = g_reverse_look_up_table;
	circular_array = g_circular_array;
	data = g_data_cache;

	nr_of_cache_entries = g_nr_of_cache_entries;
	nr_of_elements = g_nr_of_elements;

	// init pointer
	ca_first = 0;
	ca_last = nr_of_cache_entries - 1;
	ca_free_pos = 0;

	for(int i=0; i<data_size[0]+data_size[1]; i++)
	{
		look_up_table[i] = -1;
	}

}


__device__ void ca_add(int id)
{
								 // clean up look up table
	int last_id = reverse_look_up_table[ circular_array[ca_last] ];
	if(circular_array[ca_last] != -1) //test, ob schon alle stellen im cache belegt sind
	{
		//pos = look_up_table[ last_id ];
		look_up_table[ last_id ] = -1;
	}
	else
	{
		circular_array[ca_last] = ca_free_pos;
		ca_free_pos++;
	}

	//circular_array[ca_last] = pos;
	ca_first = ca_last;
	ca_last = ca_last - 1;
	if(ca_last<0) ca_last = nr_of_cache_entries - 1;

	reverse_look_up_table[circular_array[ca_first]] = id;
	look_up_table[id] = circular_array[ca_first];
}


__device__ void ca_bring_forward(int pos)
{
	//    printf("bring_fordward. enter. pos = %d\n", pos);
	int current = ca_first;
	int pos_temp = circular_array[current];
	int pos_temp2 = -1;
	//    int i;
	//    printf("circular array: ");
	//    for(i=0; i< nr_of_cache_entries; i++)
	//        printf(" %d: %d - ", i, circular_array[i]);
	//    printf("\n");

	//    printf("lut: ");
	//    for(i=0; i< nr_of_elements; i++)
	//        printf(" %d: %d - ", i, look_up_table[i]);
	//    printf("\n");

	//    printf("first = %d   last = %d \n", ca_first, ca_last);

	do
	{
		//        printf("bring_fordward. cycle. \n");

		pos_temp2 = pos_temp;
		current = current + 1;
		if(current>=nr_of_cache_entries) current = 0;
		pos_temp = circular_array[current];
		//        printf("current = %d   last = %d  pt = %d, pt2 = %d\n", current, last, pos_temp, pos_temp2);
		circular_array[current] = pos_temp2;

		//    printf("circular array 2: ");
		//    for(i=0; i< nr_of_cache_entries; i++)
		//        printf(" %d: %d - ", i, circular_array[i]);
		//    printf("\n");

	} while( pos_temp != pos);

	circular_array[ca_first] = pos;

	//    printf("circular array 3: ");
	//    for(i=0; i< nr_of_cache_entries; i++)
	//        printf(" %d: %d - ", i, circular_array[i]);
	//    printf("\n");

}

__global__ void cuda_cache_update()
{
	int idset;
	if(max_p > max_q)
	{
		idset = max_p_index;
	} else
	{
		idset = max_q_index + data_size[0];
	}

	if( look_up_table[idset] == -1 )
	{
		ca_add(idset);
		ca_cachemiss = true;
		//get_data(id, set, circular_array[ca_first]);

		//printf("cache miss, id = %d, set = %d\n", id, set);
	}							 //cache hit
	else
	{
		//printf("cache hit\n");
		if(look_up_table[idset] != circular_array[ca_first])
		{
			ca_bring_forward(look_up_table[idset]);
		}
		ca_cachemiss = false;
	}
}

__global__ void cuda_kernel_computekernels_cache()
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;

								 // falls etwas mehr threads als noetig gestartet wurden
	if(tid < data_size[0] + data_size[1])
	{
		int t_set;
		int t_element;

		if(tid < data_size[0])
			t_set = 0;
		else
			t_set = 1;

		t_element = tid - (t_set) * data_size[0];

		if (max_p >= max_q)
		{
			if( ca_cachemiss == true )
				data[circular_array[ca_first] * nr_of_elements + tid] = kernel(0, max_p_index, t_set, t_element);

								 //todo: cache einbauen
			float* computed_kernels = &data[circular_array[ca_first] * nr_of_elements];

			update_xi_x(dot_xi_x, 0, 0, max_p_index, lambda, computed_kernels, tid);

			update_xi_x(dot_xi_y, 0, 1, max_p_index, lambda, computed_kernels, tid);
		}
		else
		{
			if( ca_cachemiss == true )
				data[circular_array[ca_first] * nr_of_elements + tid] = kernel(1, max_q_index, t_set, t_element);

			float* computed_kernels = &data[circular_array[ca_first] * nr_of_elements];

			update_xi_x(dot_yi_y, 1, 1, max_q_index, lambda, computed_kernels, tid);

			update_xi_x(dot_yi_x, 1, 0, max_q_index, lambda, computed_kernels, tid);
		}
	}
}
// cache ende

__global__ void cuda_kernel_init_pointer(float* g_data0, float* g_data1 , int g_maximum_index, int g_data0_size, int g_data1_size, float* g_weights0, float* g_weights1 ,
float *g_dot_xi_x, float *g_dot_yi_x, float *g_dot_xi_y, float *g_dot_yi_y,
float* g_dot_same0, float* g_dot_same1, float* g_distance, float* g_rho, struct svm_parameter g_param)
{
	dot_xi_x = g_dot_xi_x;
	dot_yi_x = g_dot_yi_x;
	dot_xi_y = g_dot_xi_y;
	dot_yi_y = g_dot_yi_y;
	dot_same[0] = g_dot_same0;
	dot_same[1] = g_dot_same1;

	//todo: gleich die richtigen arrays senden
	g_data[0] = g_data0;
	g_data[1] = g_data1;

	maximum_index = g_maximum_index;
	data_size[0] = g_data0_size;
	data_size[1] = g_data1_size;

	g_weights[0] = g_weights0;
	g_weights[1] = g_weights1;

	distance = g_distance;
	rho = g_rho;
	
	param = g_param;
}


__global__ void cuda_kernel_init_kernel()
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;

	// falls etwas mehr threads als noetig gestartet wurden
	if(tid < data_size[0] + data_size[1])
	{
		int t_set;
		int t_element;

		if(tid < data_size[0])
			t_set = 0;
		else
			t_set = 1;

		t_element = tid - (t_set) * data_size[0];

		g_weights[t_set][t_element] = 0.0;

		// initialisieren

		if(t_set == 0)
		{
			dot_xi_x[t_element]=kernel(0, 0, t_set, t_element);
			dot_yi_x[t_element]=kernel(1, 0, t_set, t_element);
		} else
		{
			dot_xi_y[t_element]=kernel(0, 0, t_set, t_element);
			dot_yi_y[t_element]=kernel(1, 0, t_set, t_element);
		}
		dot_same[t_set][t_element]=kernel(t_set, t_element, t_set, t_element);
	}

}


__global__ void cuda_kernel_init_findmax()
{
	g_weights[0][0] = 1.0;
	g_weights[1][0] = 1.0;

	dot_xi_xi = kernel(0, 0, 0, 0);
	dot_xi_yi = kernel(0, 0, 1, 0);
	dot_yi_yi = kernel(1, 0, 1, 0);

	// find max
	//max_p_index = find_max(0, dot_yi_x, dot_xi_x, dot_xi_yi, dot_xi_xi, &max_p);

	//max_q_index = find_max(1, dot_xi_y, dot_yi_y, dot_xi_yi, dot_yi_yi, &max_q);

}

__global__ void cuda_kernel_updateWeights()
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	if (max_p >= max_q)
	{
		if(tid < data_size[0])
		{
			g_weights[0][tid] *= lambda;

		}
		if(tid == max_p_index)
		{
			g_weights[0][max_p_index] += 1.0 - lambda;
		}
	} else {
		if(tid < data_size[1])
		{
			g_weights[1][tid] *= lambda;

		}
		if(tid == max_q_index)
		{
			g_weights[1][max_q_index] += 1.0 - lambda;
		}

		//if(tid == 1691)
		//	g_weights[1][1691] = 1234.56;
	}
}

__global__ void cuda_kernel_lambda()
{
	if (max_p >= max_q)
	{
		float zaehler = compute_zaehler(dot_xi_yi, dot_yi_x, dot_xi_x, 0, max_p_index);
		float nenner = compute_nenner(dot_xi_xi, dot_xi_x, 0, max_p_index);

		lambda = zaehler / nenner;

		if(zaehler == 0.0 && nenner == 0.0) lambda = 0.0;
		if(lambda < 0.0)    lambda = 0.0;
		if(lambda > 1.0)    lambda = 0.0;

		//add_to_weights(g_weights[0], lambda, max_p_index, 0);

		// update dotproducts

		dot_xi_xi = update_xi_xi(dot_xi_xi, dot_xi_x, 0, max_p_index, lambda);

		dot_xi_yi = update_xi_yi(dot_xi_yi, dot_yi_x, max_p_index, lambda);
	}
	else
	{
		double zaehler = compute_zaehler(dot_xi_yi, dot_xi_y, dot_yi_y, 1, max_q_index);
		double nenner = compute_nenner(dot_yi_yi, dot_yi_y, 1, max_q_index);

		lambda = zaehler / nenner;

		if(zaehler == 0.0 && nenner == 0.0) lambda = 0.0;
		if(lambda < 0.0)    lambda = 0.0;
		if(lambda > 1.0)    lambda = 0.0;

		//g_temp[0] = lambda;

		//add_to_weights(g_weights[1], lambda, max_q_index, 1);

		// update dotproducts

		dot_yi_yi = update_xi_xi(dot_yi_yi, dot_yi_y, 1, max_q_index, lambda);

		dot_xi_yi = update_xi_yi(dot_xi_yi, dot_xi_y, max_q_index, lambda);
	}

	*distance = dot_xi_xi + dot_yi_yi - 2 * dot_xi_yi;
	*rho = dot_xi_yi - dot_xi_xi - (dot_xi_xi + dot_yi_yi - 2 * dot_xi_yi)/2;
}

__global__ void cuda_kernel_distance()
{
	// find max
	max_p_index = find_max(0, dot_yi_x, dot_xi_x, dot_xi_yi, dot_xi_xi, &max_p);
	max_q_index = find_max(1, dot_xi_y, dot_yi_y, dot_xi_yi, dot_yi_yi, &max_q);

	//duality gap
	// absolute duality gap

	//todo: rueckgabewert an host zurueckgeben

	//float adg = max_p + max_q;

	//printf("max_p = %f  max_q = %f ", max_p, max_q);
	//printf("adg = %f ", adg);

	// relative duality gap
	// adg / ||p-q||^2 - adg
	// adg / <p-q, p-q> - adg

	//float distance = dot_xi_xi + dot_yi_yi - 2 * dot_xi_yi;

	//float rdg_nenner = distance - adg;
	//float rdg;

	//if (rdg_nenner <= 0)
	//{
	//printf("set huge value... ");
	//	rdg = 100000000000.0;	 // todo: HUGE_VAL;
	//}
	//else
	//{
	//	rdg = adg / rdg_nenner;
	//}

	//printf("<x-y,x-y> = %e " , distance);
	//printf("adg = %e " , adg);
	//printf("rdg = %e \n", rdg);
	//print_weights(x_weights, prob[0]);
	//print_weights(y_weights, prob[1]);

	//rho = - dot_xi_yi + dot_xi_xi - (dot_xi_xi + dot_yi_yi - 2 * dot_xi_yi)/2;
	//float rho = dot_xi_yi - dot_xi_xi - (dot_xi_xi + dot_yi_yi - 2 * dot_xi_yi)/2;
	//printf("xi_xi = %f   yi_yi = %f   xi_yi = %f \n", dot_xi_xi, dot_yi_yi, dot_xi_yi);

}
#endif							 // #ifndef _CUDA_KERNEL_H_
