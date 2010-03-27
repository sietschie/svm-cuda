#ifndef _CUDA_KERNEL_H_
#define _CUDA_KERNEL_H_

#include "globals.h"

__device__ float* g_data[2];
__device__ int maximum_index;
__device__ int data_size[2];
__device__ float C;
__device__ float* g_weights[2];
__device__ float lambda;
__device__ int max_p_index;
__device__ float max_p;
__device__ int max_q_index;
__device__ float max_q;
__device__ float dot_xi_yi;		 // <x_i, y_i >
__device__ float dot_xi_xi;		 // <x_i, x_i >
__device__ float dot_yi_yi;		 // <y_i, y_i >
__device__ float* dot_xi_x;
__device__ float* dot_yi_x;
__device__ float* dot_xi_y;
__device__ float* dot_yi_y;

__device__ float* get_element(int id, int set);

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


//float kernel_linear(int set1, int element1, int set2, int element2) //todo: als template implementieren
__device__ float kernel(int set1, int element1, int set2, int element2)
{
	float* px = &(g_data[set1][ element1 * maximum_index ]);
	float* py = &(g_data[set2][ element2 * maximum_index ]);

	float ret = dot(px, py );
	if(set1 == set2 && element1 == element2)
		ret += C;
	return ret;
}


/*float kernel_poly(int set1, int element1, int set2, int element2)
{
	float ret = powi(param.gamma*dot(prob[set1].x[element1], prob[set2].x[element2])+param.coef0,param.degree);
	if(set1 == set2 && element1 == element2)
		ret += param.C;
	return ret;
}

float kernel_rbf(int set1, int element1, int set2, int element2)
{
	float dots = ( dot(prob[set1].x[element1], prob[set1].x[element1])+
						dot(prob[set1].x[element1], prob[set2].x[element2])-2*
						dot(prob[set1].x[element1], prob[set2].x[element2]));
	float wgamma = -param.gamma*dots;
	float wexp = exp(wgamma);

	if(set1 == set2 && element1 == element2)
		wexp += param.C;
	return wexp;

}

float kernel_sigmoid(int set1, int element1, int set2, int element2)
{
	float ret = tanh(param.gamma*dot(prob[set1].x[element1], prob[set2].x[element2])+param.coef0);
	if(set1 == set2 && element1 == element2)
		ret += param.C;
	return ret;
}
*/

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


__device__ float compute_zaehler(float dot_xi_yi, float* dot_yi_x, float* dot_xi_x, int p, int max_p_index )
{
	 //todo: samevector, kann vorberechnet werden.
	float zaehler = dot_xi_yi - dot_yi_x[max_p_index] - dot_xi_x[max_p_index] + kernel(p,max_p_index, p, max_p_index);
	return zaehler;
}


__device__ float compute_nenner(float dot_xi_xi, float* dot_xi_x, int p, int max_p_index)
{
	float nenner = dot_xi_xi - 2* dot_xi_x[max_p_index] +  kernel(p, max_p_index, p, max_p_index);
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
		+ (1.0 - lambda)*(1.0 - lambda)*kernel(p, max_p_index, p ,max_p_index );
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

__device__ void get_data(int id, int set, int pointer)
{
	//    data[pointer] = (double) id * id;

	int i;
	for(i=0; i<data_size[0]; i++)
	{
		//printf("set1 = %d, id = %d,  set2 = %d, id = %d res = %f\n", set, id, 0, i,  kernel(set, id, 0, i));
		data[pointer * nr_of_elements + i] = kernel(set, id, 0, i);
	}

	for(i=0; i<data_size[1]; i++)
	{
		//printf("set1 = %d, id = %d,  set2 = %d, id = %d   res = %f \n", set, id, 1, i,  kernel(set, id, 1, i));
		data[pointer * nr_of_elements + i + data_size[0]] = kernel(set, id, 1, i);
	}
}


__device__ void ca_add(int id)
{
								 // clean up look up table
	int last_id = reverse_look_up_table[ circular_array[ca_last] ];
	if(circular_array[ca_last] != -1)
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


__device__ float* get_element(int id, int set, int tid)
{
	//printf(" get_element(): id = %d, set = %d \n", id, set);
	int idset = id + set* data_size[0];

	//printf("idset = %d \n", idset);

								 // cache miss
	if( look_up_table[idset] == -1 )
	{
		if(tid  == 1)
			ca_add(idset);
		get_data(id, set, circular_array[ca_first]);
		//printf("cache miss, id = %d, set = %d\n", id, set);
	}							 //cache hit
	else
	{
		//printf("cache hit\n");
		if(look_up_table[idset] != circular_array[ca_first])
		{
			ca_bring_forward(look_up_table[idset]);
		}
	}
	//printf("get_element = data[%d]  ca_first = %d \n", circular_array[ca_first], ca_first);
	return &data[circular_array[ca_first] * nr_of_elements];
}


// cache ende

__global__ void cuda_kernel_init_pointer(float* g_data0, float* g_data1 , int g_maximum_index, int g_data0_size, int g_data1_size, float* g_weights0, float* g_weights1 ,
float *g_dot_xi_x, float *g_dot_yi_x, float *g_dot_xi_y, float *g_dot_yi_y,
int g_nr_of_cache_entries, int g_nr_of_elements,
								 //todo: bessere variablennamen fuer cache zeugs finden finden
int *g_look_up_table, int* g_reverse_look_up_table, int* g_circular_array, float* g_data_cache, float* g_temp)
{
	// cache initialisieren
	look_up_table = g_look_up_table;
	reverse_look_up_table = g_reverse_look_up_table;
	circular_array = g_circular_array;
	data = g_data_cache;

	nr_of_cache_entries = g_nr_of_cache_entries;
	nr_of_elements = g_nr_of_elements;

	dot_xi_x = g_dot_xi_x;
	dot_yi_x = g_dot_yi_x;
	dot_xi_y = g_dot_xi_y;
	dot_yi_y = g_dot_yi_y;

	// init pointer
	ca_first = 0;
	ca_last = nr_of_cache_entries - 1;

	//cache init ende

	//todo: C als parameter uebergeben
	C = 0.0;
	//todo: gleich die richtigen arrays senden
	g_data[0] = g_data0;
	g_data[1] = g_data1;

	maximum_index = g_maximum_index;
	data_size[0] = g_data0_size;
	data_size[1] = g_data1_size;

	g_weights[0] = g_weights0;
	g_weights[1] = g_weights1;

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
	max_p_index = find_max(0, dot_yi_x, dot_xi_x, dot_xi_yi, dot_xi_xi, &max_p);

	max_q_index = find_max(1, dot_xi_y, dot_yi_y, dot_xi_yi, dot_yi_yi, &max_q);

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

		add_to_weights(g_weights[0], lambda, max_p_index, 0);

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

		add_to_weights(g_weights[1], lambda, max_q_index, 1);

		// update dotproducts

		dot_yi_yi = update_xi_xi(dot_yi_yi, dot_yi_y, 1, max_q_index, lambda);

		dot_xi_yi = update_xi_yi(dot_xi_yi, dot_xi_y, max_q_index, lambda);
	}

}


__global__ void cuda_kernel_computekernels()
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
			data[tid] = kernel(0, max_p_index, t_set, t_element);

								 //todo: cache einbauen
			float* computed_kernels = data;

			update_xi_x(dot_xi_x, 0, 0, max_p_index, lambda, computed_kernels, tid);

			update_xi_x(dot_xi_y, 0, 1, max_p_index, lambda, computed_kernels, tid);
		}
		else
		{
			if(tid < data_size[0])
			{
				data[tid] = kernel(1, max_q_index, 0, tid);
			}
			else
			{
				data[tid] = kernel(1, max_q_index, 1, tid - data_size[0]);
			}

			float* computed_kernels = data;

			update_xi_x(dot_yi_y, 1, 1, max_q_index, lambda, computed_kernels, tid);

			update_xi_x(dot_yi_x, 1, 0, max_q_index, lambda, computed_kernels, tid);
		}
	}
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
