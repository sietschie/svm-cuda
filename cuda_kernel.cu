#ifndef _CUDA_KERNEL_H_
#define _CUDA_KERNEL_H_

#include "globals.h"

__device__ float* g_data[2];
__device__ int max_vector;

double dot(float* px, float *py)
{
//    print_vector(px);
//    print_vector(py);
	double sum = 0;
	int i;
	for(i=0; i< max_vector; i++)
//	while(px->index != -1 && py->index != -1)
	{
		sum += px[i] * py[i];
	}
	return sum;
}

inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

    int t;
	for(t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}

//double kernel_linear(int set1, int element1, int set2, int element2) //todo: als template implementieren
__device__ double kernel(int set1, int element1, int set2, int element2)
{
//	g_data[set1][ element1 * max_vector[set1] ]
//    double ret = dot(prob[set1].x[element1], prob[set2].x[element2]);
    double ret = dot(g_data[set1][ element1 * max_vector[set1] ], g_data[set2][ element2 * max_vector[set2] ]);
    if(set1 == set2 && element1 == element2)
        ret += param.C;
    return ret;
}

double kernel_poly(int set1, int element1, int set2, int element2)
{
    double ret = powi(param.gamma*dot(prob[set1].x[element1], prob[set2].x[element2])+param.coef0,param.degree);
    if(set1 == set2 && element1 == element2)
        ret += param.C;
    return ret;
}

double kernel_rbf(int set1, int element1, int set2, int element2)
{
    double dots = ( dot(prob[set1].x[element1], prob[set1].x[element1])+
                        dot(prob[set1].x[element1], prob[set2].x[element2])-2*
                        dot(prob[set1].x[element1], prob[set2].x[element2]));
    double wgamma = -param.gamma*dots;
    double wexp = exp(wgamma);

    if(set1 == set2 && element1 == element2)
        wexp += param.C;
    return wexp;

}

double kernel_sigmoid(int set1, int element1, int set2, int element2)
{
    double ret = tanh(param.gamma*dot(prob[set1].x[element1], prob[set2].x[element2])+param.coef0);
    if(set1 == set2 && element1 == element2)
        ret += param.C;
    return ret;
}

/*double kernel_precomputed(int set1, int element1, int set2, int element2)
{
    return x[i][(int)(x[j][0].value)].value;
}*/


__global__ void
cuda_kernel( float* g_data0, float* g_data1 , int max_index, int g_max_vector, float* g_weights0, float* g_weights1 ,
float *dot_xi_x, float *dot_yi_x, float *dot_xi_y, float *dot_yi_y)
{
	//todo: gleich die richtigen arrays senden
	g_data[0] = g_data0;
	g_data[1] = g_data1;

	max_vector = g_max_vector0;

	float* g_weights[2];
	g_weights[0] = g_weights0;
	g_weights[1] = g_weights1;


    // initialize weights  -- 0 == x, 1 == y
    int i;
    for (i=0;i<max_vector[0];i++)
        g_weights[0][i] = 0.0;

    for (i=0;i<max_vector[1];i++)
        g_weights[1][i] = 0.0;

    g_weights[0][0] = 1.0;
    g_weights[1][0] = 1.0;


    // deklaration der variablen die werte zwischenspeichern
    //float *dot_xi_x; // < x_i, x> \forall x \in P
    //float *dot_yi_x;  // < y_i, x> \forall x \in P
    float dot_xi_yi; // <x_i, y_i >
    float dot_xi_xi; // <x_i, x_i >

    //float *dot_yi_y; // < y_i, y> \forall y \in Q
    //float *dot_xi_y;  // < x_i, y> \forall y \in Q
    float dot_yi_yi; // <y_i, y_i >


    // speicher anfordern 

    // initialisieren
    for (i=0;i<prob[0].l;i++) {
        dot_xi_x[i]=kernel(0, 0, 0, i);
        dot_yi_x[i]=kernel(1, 0, 0, i);
    }

    for (i=0;i<prob[1].l;i++) {
        dot_xi_y[i]=kernel(0, 0, 1, i);
        dot_yi_y[i]=kernel(1, 0, 1, i);
    }

    dot_xi_xi = kernel(0, 0, 0, 0);
    dot_xi_yi = kernel(0, 0, 1, 0);
    dot_yi_yi = kernel(1, 0, 1, 0);

//	struct svm_problem d_prob[2];

//	int *test;
//	cudaMalloc( (void**) &test, 100 * sizeof(int) );

//	int tid = threadIdx.x + blockDim.x*blockIdx.x;
//	g_data[tid] = tid; 
}

#endif // #ifndef _CUDA_KERNEL_H_
