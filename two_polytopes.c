#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "readsvm.h"
//#define INFINITY	__builtin_inf() // todo: get rid of that


struct svm_problem prob_p;
struct svm_problem prob_q;

int compute_max_index(const struct svm_problem prob)
{
    int i;
    int max_index = 1;
    for (i=0;i<prob.l;i++)
    {
        struct svm_node* px = prob.x[i];
        while (px->index != -1)
        {
            if (px->index > max_index)
                max_index = px->index;
            px++;
        }
    }
    return max_index;
}

void print_vector( const struct svm_node* vector)
{
    while (vector->index != -1)
    {
        printf("%d: %f  ", vector->index, vector->value);
        ++vector;
    }
    printf("\n");
}

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
	return sum;
}

void add_to_weights(double* weights, double lambda, int index, struct svm_problem prob)
{
    int i;
    for (i=0;i<prob.l;i++)
    {
        if (i!=index)
            weights[i] *= lambda;
        else
            weights[i] = weights[i]*lambda + (1.0 - lambda)*1;
    }
}

void print_weights(double* weights, struct svm_problem prob)
{
	int i;
	for(i=0;i<prob.l;i++)
	{
			printf("%d:\t%f  ",i,weights[i]);
	}
	printf("\n");
}

int find_max(struct svm_problem prob_p, double *dot_yi_x, double* dot_xi_x, double dot_xi_yi, double dot_xi_xi, double *max_p) {
    // find max
    int max_p_index = -1;
    *max_p = -HUGE_VAL;
    int i;
    for (i=0;i<prob_p.l;i++) {
        double sum = dot_yi_x[i] - dot_xi_x[i] - dot_xi_yi + dot_xi_xi;
        if(sum > *max_p)
        {
            *max_p = sum;
            max_p_index = i;
        }
    }
    return max_p_index;
}

double compute_zaehler(double dot_xi_yi, double* dot_yi_x, double* dot_xi_x, struct svm_problem prob_p, int max_p_index ) {
    double zaehler = dot_xi_yi - dot_yi_x[max_p_index] - dot_xi_x[max_p_index] + dot(prob_p.x[max_p_index], prob_p.x[max_p_index]);
    return zaehler;
}

double compute_nenner(double dot_xi_xi, double* dot_xi_x, struct svm_problem prob_p, int max_p_index) {
    double nenner = dot_xi_xi - 2* dot_xi_x[max_p_index] +  dot(prob_p.x[max_p_index], prob_p.x[max_p_index]);
    return nenner;
}

double update_xi_xi(double dot_xi_xi, double* dot_xi_x, struct svm_problem prob_p, int max_p_index, double lambda) {
    dot_xi_xi = lambda * lambda * dot_xi_xi
            + 2 * lambda * (1.0 - lambda) * dot_xi_x[max_p_index]
            + (1.0 - lambda)*(1.0 - lambda)*dot(prob_p.x[max_p_index],prob_p.x[max_p_index] );
    return dot_xi_xi;
}

double update_xi_yi(double dot_xi_yi, double* dot_yi_x, int max_p_index, double lambda) {
    dot_xi_yi = lambda * dot_xi_yi + (1.0 - lambda) * dot_yi_x[max_p_index];
    return dot_xi_yi;
}

void update_xi_x(double* dot_xi_x, struct svm_problem prob_p, struct svm_problem prob_p2, int max_p_index, double lambda) {
    int i;
    for (i=0;i<prob_p2.l;i++) {
        dot_xi_x[i]= dot_xi_x[i] * lambda + (1.0 - lambda) * dot(prob_p.x[max_p_index], prob_p2.x[i]);
    }
}


int main (int argc, const char ** argv)
{
    const char* filename;

    printf("Dateien einlesen... \n");
    if (argc < 2)
        //filename = "data/heart_scale.plus";
        filename = "data/triangle.left.txt";
    else
        filename = argv[1];

    read_problem(filename);

    prob_p = prob;

    if (argc < 3)
        //filename = "data/heart_scale.minus";
        filename = "data/triangle.right.txt";
    else
        filename = argv[2];

    read_problem(filename);

    prob_q = prob;

    printf("Anzahl der eingelesenen Datensaetze: P = %d   Q = %d \n", prob_p.l, prob_q.l);

    int max_index_q = compute_max_index(prob_q); //Todo: besseren namen finden
    int max_index_p = compute_max_index(prob_p);

    printf("Groesse der Vektoren: P = %d, Q = %d \n", max_index_p, max_index_q);

    int elements = max_index_p;
    if (elements < max_index_q)	elements = max_index_q;
    elements++; //add one for the stopping index -1

    printf("Gewichtsvektoren initialisieren.. \n");
    double* x_weights = (double *) malloc(prob_p.l * sizeof(double));

    double* y_weights = (double *) malloc(prob_q.l * sizeof(double));

    // initialize weights
    int i;
    for (i=0;i<prob_p.l;i++)
        x_weights[i] = 0.0;

    for (i=0;i<prob_q.l;i++)
        y_weights[i] = 0.0;

    x_weights[0] = 1.0;
    y_weights[0] = 1.0;


    // deklaration der variablen die werte zwischenspeichern
    double *dot_xi_x; // < x_i, x> \forall x \in P
    double *dot_yi_x;  // < y_i, x> \forall x \in P
    double dot_xi_yi; // <x_i, y_i >
    double dot_xi_xi; // <x_i, x_i >

    double *dot_yi_y; // < y_i, y> \forall y \in Q
    double *dot_xi_y;  // < x_i, y> \forall y \in Q
    double dot_yi_yi; // <y_i, y_i >

    // speicher anfordern
    dot_xi_x = (double *) malloc(prob_p.l * sizeof(double));
    dot_yi_x = (double *) malloc(prob_p.l * sizeof(double));

    dot_xi_y = (double *) malloc(prob_q.l * sizeof(double));
    dot_yi_y = (double *) malloc(prob_q.l * sizeof(double));

    // initialisieren
    for (i=0;i<prob_p.l;i++) {
        dot_xi_x[i]=dot(prob_p.x[0], prob_p.x[i]);
        dot_yi_x[i]=dot(prob_q.x[0], prob_p.x[i]);
    }

    for (i=0;i<prob_q.l;i++) {
        dot_xi_y[i]=dot(prob_p.x[0], prob_q.x[i]);
        dot_yi_y[i]=dot(prob_q.x[0], prob_q.x[i]);
    }

    dot_xi_xi = dot(prob_p.x[0], prob_p.x[0]);
    dot_xi_yi = dot(prob_p.x[0], prob_q.x[0]);
    dot_yi_yi = dot(prob_q.x[0], prob_q.x[0]);


    // find max
    int max_p_index;
    double max_p;
    max_p_index = find_max(prob_p, dot_yi_x, dot_xi_x, dot_xi_yi, dot_xi_xi, &max_p);

    int max_q_index;
    double max_q;
    max_q_index = find_max(prob_q, dot_xi_y, dot_yi_y, dot_xi_yi, dot_yi_yi, &max_q);

    int j;

    for (j=0;j<1000;j++)
    {
        double lambda;
        if (max_p >= max_q)
        {
            double zaehler = compute_zaehler(dot_xi_yi, dot_yi_x, dot_xi_x, prob_p, max_p_index);
            double nenner = compute_nenner(dot_xi_xi, dot_xi_x, prob_p, max_p_index);

            lambda = zaehler / nenner;

            if(zaehler == 0.0 && nenner == 0.0) lambda = 0.0;
            if(lambda < 0.0)	lambda = 0.0;
            if(lambda > 1.0)	lambda = 0.0;

            add_to_weights(x_weights, lambda, max_p_index, prob_p);

            // update dotproducts

            dot_xi_xi = update_xi_xi(dot_xi_xi, dot_xi_x, prob_p, max_p_index, lambda);

            dot_xi_yi = update_xi_yi(dot_xi_yi, dot_yi_x, max_p_index, lambda);

            update_xi_x(dot_xi_x, prob_p, prob_p, max_p_index, lambda);

            update_xi_x(dot_xi_y, prob_p, prob_q, max_p_index, lambda);

            // find max
            max_p_index = find_max(prob_p, dot_yi_x, dot_xi_x, dot_xi_yi, dot_xi_xi, &max_p);
            max_q_index = find_max(prob_q, dot_xi_y, dot_yi_y, dot_xi_yi, dot_yi_yi, &max_q);

        }
        else
        {
            double zaehler = compute_zaehler(dot_xi_yi, dot_xi_y, dot_yi_y, prob_q, max_q_index);
            double nenner = compute_nenner(dot_yi_yi, dot_yi_y, prob_q, max_q_index);

            lambda = zaehler / nenner;

            if(zaehler == 0.0 && nenner == 0.0) lambda = 0.0;
            if(lambda < 0.0)	lambda = 0.0;
            if(lambda > 1.0)	lambda = 0.0;

            add_to_weights(y_weights, lambda, max_q_index, prob_q);

            // update dotproducts

            dot_yi_yi = update_xi_xi(dot_yi_yi, dot_yi_y, prob_q, max_q_index, lambda);

            dot_xi_yi = update_xi_yi(dot_xi_yi, dot_xi_y, max_q_index, lambda);

            update_xi_x(dot_yi_y, prob_q, prob_q, max_q_index, lambda);

            update_xi_x(dot_yi_x, prob_q, prob_p, max_q_index, lambda);

            // find max
            max_q_index = find_max(prob_q, dot_xi_y, dot_yi_y, dot_xi_yi, dot_yi_yi, &max_q);
            max_p_index = find_max(prob_p, dot_yi_x, dot_xi_x, dot_xi_yi, dot_xi_xi, &max_p);

       }

        //duality gap
        // absolute duality gap

        double adg = max_p + max_q;

        //printf("max_p = %f  max_q = %f ", max_p, max_q);
        //printf("adg = %f ", adg);

        // relative duality gap
        // adg / ||p-q||^2 - adg
        // adg / <p-q, p-q> - adg

        double distance = dot_xi_xi + dot_yi_yi - 2 * dot_xi_yi;

		printf("<x-y,x-y> = %f " , distance);

        double rdg_nenner = distance - adg;
        double rdg;

        if (rdg_nenner <= 0)
        {
            //printf("set huge value... ");
            rdg = HUGE_VAL;
        }
        else
        {
            rdg = adg / rdg_nenner;
        }

        printf("rdg = %f \n", rdg);
		//print_weights(x_weights, prob_p);
		//print_weights(y_weights, prob_q);

    }

    return 0;
}
