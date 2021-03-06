#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "globals.h"
#include "kernel.h"

//static double C = 5.5;

double dot(const struct svm_node *px, const struct svm_node *py)
{
//    print_vector(px);
//    print_vector(py);
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
            //printf("  dot: %f = %f * %f, sum = %f, index = %d \n", px->value * py->value, px->value, py->value, sum, px->index);
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}
	}
    //printf("  dot = %f \n", sum);
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

double kernel_linear(int set1, int element1, int set2, int element2)
//double kernel(int set1, int element1, int set2, int element2)
{
	//printf(" dot: %d %d - %d %d \n", set1, element1, set2, element2);
    double ret = dot(prob[set1].x[element1], prob[set2].x[element2]);
    if(set1 == set2 && element1 == element2)
        ret += 1/param.C;
    return ret;
}

double kernel_poly(int set1, int element1, int set2, int element2)
{
    double ret = powi(param.gamma*dot(prob[set1].x[element1], prob[set2].x[element2])+param.coef0,param.degree);
    if(set1 == set2 && element1 == element2)
        ret += 1/param.C;
    return ret;
}

double* dot_same2[2];

void init_dotsame()
{
    dot_same2[0] = (double *) malloc(prob[0].l * sizeof(double));
    dot_same2[1] = (double *) malloc(prob[1].l * sizeof(double));
    // initialisieren
	int i;
    for (i=0;i<prob[0].l;i++) {
        dot_same2[0][i]= dot(prob[0].x[i], prob[0].x[i]); //kernel(0,i,0,i);
    }

    for (i=0;i<prob[1].l;i++) {
        dot_same2[1][i]= dot(prob[1].x[i], prob[1].x[i]);;
    }
}

double kernel_rbf(int set1, int element1, int set2, int element2)
{
	//printf("1 cached: %f   real: %f\n", dot_same2[set1][element1], dot(prob[set1].x[element1], prob[set1].x[element1]));
	//printf("2 cached: %f   real: %f\n", dot_same2[set2][element2], dot(prob[set2].x[element2], prob[set2].x[element2]));
    double dots = ( dot_same2[set1][element1] + dot_same2[set2][element2] ) -2*
                        dot(prob[set1].x[element1], prob[set2].x[element2]);
    double wgamma = -param.gamma*dots;
    double wexp = exp(wgamma);

    if(set1 == set2 && element1 == element2)
        wexp += 1/param.C;
    return wexp;

}

double kernel_sigmoid(int set1, int element1, int set2, int element2)
{
    double ret = tanh(param.gamma*dot(prob[set1].x[element1], prob[set2].x[element2])+param.coef0);
    if(set1 == set2 && element1 == element2)
        ret += 1/param.C;
    return ret;
}

/*double kernel_precomputed(int set1, int element1, int set2, int element2)
{
    return x[i][(int)(x[j][0].value)].value;
}*/

double (*kernel)(int, int, int, int);
