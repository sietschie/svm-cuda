#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "globals.h"
#include "../common/readsvm.h"

struct svm_model* model;

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
    double ret = dot(prob[set1].x[element1], model->SV[set2][element2]);
    //if(set1 == set2 && element1 == element2)
    //    ret += 1/param.C;
    return ret;
}

double kernel2_linear(int set1, int element1, int set2, int element2)
//double kernel(int set1, int element1, int set2, int element2)
{
	//printf(" dot: %d %d - %d %d \n", set1, element1, set2, element2);
    double ret = dot(model->SV[set1][element1], model->SV[set2][element2]);
    //if(set1 == set2 && element1 == element2)
    //    ret += 1/param.C;
    return ret;
}

double kernel_poly(int set1, int element1, int set2, int element2)
{
    double ret = powi(param.gamma*dot(prob[set1].x[element1], model->SV[set2][element2])+param.coef0,param.degree);
    //if(set1 == set2 && element1 == element2)
    //    ret += 1/param.C;
    return ret;
}

double kernel2_poly(int set1, int element1, int set2, int element2)
{
    double ret = powi(param.gamma*dot(model->SV[set1][element1], model->SV[set2][element2])+param.coef0,param.degree);
    //if(set1 == set2 && element1 == element2)
    //    ret += 1/param.C;
    return ret;
}

double kernel_rbf(int set1, int element1, int set2, int element2)
{
    double dots = ( dot(prob[set1].x[element1], prob[set1].x[element1])+
                        dot(model->SV[set2][element2], model->SV[set2][element2])-2*
                        dot(prob[set1].x[element1], model->SV[set2][element2]));
    double wgamma = -param.gamma*dots;
    double wexp = exp(wgamma);

    //if(set1 == set2 && element1 == element2)
    //    wexp += 1/param.C;
    return wexp;

}

double kernel2_rbf(int set1, int element1, int set2, int element2)
{
    double dots = ( dot(model->SV[set1][element1], model->SV[set1][element1])+
                        dot(model->SV[set2][element2], model->SV[set2][element2])-2*
                        dot(model->SV[set1][element1], model->SV[set2][element2]));
    double wgamma = -param.gamma*dots;
    double wexp = exp(wgamma);

    //if(set1 == set2 && element1 == element2)
    //    wexp += 1/param.C;
    return wexp;

}

double kernel_sigmoid(int set1, int element1, int set2, int element2)
{
    double ret = tanh(param.gamma*dot(prob[set1].x[element1], model->SV[set2][element2])+param.coef0);
    //if(set1 == set2 && element1 == element2)
    //    ret += 1/param.C;
    return ret;
}

double kernel2_sigmoid(int set1, int element1, int set2, int element2)
{
    double ret = tanh(param.gamma*dot(model->SV[set1][element1], model->SV[set2][element2])+param.coef0);
    //if(set1 == set2 && element1 == element2)
    //    ret += 1/param.C;
    return ret;
}

/*double kernel_precomputed(int set1, int element1, int set2, int element2)
{
    return x[i][(int)(x[j][0].value)].value;
}*/

double (*kernel)(int, int, int, int);
double (*kernel2)(int, int, int, int);

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

int max_line_len;
char *line;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

struct svm_model *svm_load_model2(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL) return NULL;
	
	// read parameters

	struct svm_model *model = malloc(sizeof(struct svm_model));
	struct svm_parameter *param;
    param = &model->param;
	//model->rho = NULL;
	//model->probA = NULL;
	//model->probB = NULL;
	//model->label = NULL;
	//model->nSV = NULL;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);

		if(strcmp(cmd,"svm_type")==0)
        {
            fscanf(fp,"%80s",cmd);
            if(strcmp("c_svc",cmd)!=0)
            {
  				fprintf(stderr,"unknown svm type.\n");
				free(model);
				return NULL;
            }
        }
		else if(strcmp(cmd,"kernel_type")==0)
		{		
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					param->kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");
				free(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			fscanf(fp,"%d",&param->degree);
		else if(strcmp(cmd,"gamma")==0)
			fscanf(fp,"%lf",&param->gamma);
		else if(strcmp(cmd,"coef0")==0)
			fscanf(fp,"%lf",&param->coef0);
		else if(strcmp(cmd,"nr_class")==0)
			fscanf(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			fscanf(fp,"%d",&model->l);
		else if(strcmp(cmd,"rho")==0)
    		fscanf(fp,"%lf",&model->rho);
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
            int i;			
            for(i=0;i<n;i++)
				fscanf(fp,"%d",&model->label[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
            int i;
			for(i=0;i<n;i++)
				fscanf(fp,"%d",&model->nSV[i]);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;	
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free(model);
			return NULL;
		}
	}

	// read sv_coef and SV

	int elements[2];
    elements[0] = 0;
    elements[1] = 0;

    int l[2];
    l[0] = 0;
    l[1] = 1;

	long pos = ftell(fp);

	max_line_len = 1024;
	line = malloc(sizeof(char) * max_line_len);
	char *p,*endptr,*idx,*val;

    int current_set = 0;

	while(readline(fp)!=NULL)
	{
		if(line[0] == '-')
			current_set = 1;
		else
			current_set = 0;

		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
				break;
			++elements[current_set];
		}
		++elements[current_set];
        ++l[current_set];
	}
	//elements += model->l;

	fseek(fp,pos,SEEK_SET);

	int m = model->nr_class - 1;
	//int l = model->l;
	//model->sv_coef = malloc(sizeof(double *) * m);
	int k;
	for(k=0;k<2;k++)
		model->weights[k] = malloc(sizeof(double) * model->nSV[k]);
	model->SV[0] = (struct svm_node**) malloc(sizeof(struct svm_node*) * l[0]);
	model->SV[1] = (struct svm_node**) malloc(sizeof(struct svm_node*) * l[1]);
	struct svm_node *x_space[2];
    x_space[0] = NULL;
    x_space[1] = NULL;

	if(l[0]>0) x_space[0] = malloc( sizeof(struct svm_node) * elements[0]);
	if(l[1]>0) x_space[1] = malloc( sizeof(struct svm_node) * elements[1]);

	int j[2];
	int i[2];

	j[0]=0;
	j[1]=0;

	i[0] = 0;
	i[1] = 0;

	while( i[0] < l[0] || i[1] < l[1] )
	//for(i=0;i<l;i++)
	{
		readline(fp);
		if(line[0] == '-')
			current_set = 1;
		else
			current_set = 0;

		model->SV[current_set][i[current_set]] = &x_space[current_set][j[current_set]];

		p = strtok(line, " \t");
		model->weights[current_set][i[current_set]] = strtod(p,&endptr);
        if( current_set == 1)
        {
            model->weights[current_set][i[current_set]] *= -1;
        }

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			x_space[current_set][j[current_set]].index = (int) strtol(idx,&endptr,10);
			x_space[current_set][j[current_set]].value = strtod(val,&endptr);

			++j[current_set];
		}
		x_space[current_set][j[current_set]++].index = -1;
        i[current_set]++;
	}
	free(line);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	model->free_sv = 1;	// XXX
	return model;
}

double compute_wxi(int p, int index) // w \cdot x_i
{
    double res=0.0;
    int i;
    for(i=0;i<model->nSV[0];i++)
    {
//        printf("xw = %f   kernel = %f sum = %f \n ", x_weights[i], kernel(0,i,p,index), kernel(0,i,p,index));
        res += model->weights[0][i] * kernel(p,index, 0,i);

    }

    //printf("res1 = %f \n", res);

    for(i=0;i<model->nSV[1];i++)
    {
//        printf("yw = %f   kernel = %f sum = %f \n ", y_weights[i], kernel(1,i,p,index), kernel(1,i,p,index));
        res -= model->weights[1][i] * kernel(p,index, 1,i);
    }

    //printf("res2 = %f \n", res);
    return res;
}

double compute2_wxi(int p, int index) // w \cdot x_i
{
    double res=0.0;
    int i;
    for(i=0;i<model->nSV[0];i++)
    {
//        printf("xw = %f   kernel = %f sum = %f \n ", x_weights[i], kernel(0,i,p,index), kernel(0,i,p,index));
        res += model->weights[0][i] * kernel2(0,i,p,index);

    }

    //printf("res1 = %f \n", res);

    for(i=0;i<model->nSV[1];i++)
    {
//        printf("yw = %f   kernel = %f sum = %f \n ", y_weights[i], kernel(1,i,p,index), kernel(1,i,p,index));
        res -= model->weights[1][i] * kernel2(1,i,p,index);
    }

    //printf("res2 = %f \n", res);
    return res;
}

double findmax(int set)
{
    printf("findmax.. \n");
    double max = compute2_wxi(set, 0);
    int i;
    for(i=0;i<model->nSV[set];i++)
    {
	double res = compute2_wxi(set, i);//dot_w(w,prob[set].x[i]);
	if( res > max )
	{
	    max = res;
	}
        //printf(" res = %f \n", res);
    }
    //printf(" max = %f \n", max);
    return max;
}

double findmin(int set)
{
    printf("findmin.. \n");
    double min = compute2_wxi(set, 0);
    int i;
    for(i=0;i < model->nSV[set];i++)
    {
	//printf("..%d..(%d)\n", i, prob[set].x[0]->index);
	double res = compute2_wxi(set, i); //dot_w(w,prob[set].x[i]);
	if( res < min )
	{
	    min = res;
	}
        //printf(" res = %f \n", res);
    }
    //printf(" min = %f \n", min);
    return min;
}


int main (int argc, char ** argv)
{

    kernel = &kernel_linear;
    kernel2 = &kernel2_linear;
    model = svm_load_model2(argv[1]);

    double t = kernel2_rbf(0,0,0,0);


	switch(model->param.kernel_type)
	{
		case LINEAR:
			kernel = &kernel_linear;
			kernel2 = &kernel2_linear;
			break;
		case POLY:
			kernel = &kernel_poly;
			kernel2 = &kernel2_poly;
			break;
		case RBF:
			kernel = &kernel_rbf;
			kernel2 = &kernel2_rbf;
			break;
		case SIGMOID:
			kernel = &kernel_sigmoid;
			kernel2 = &kernel2_sigmoid;
			break;
	}

    read_problem(argv[2], prob, &param, &max_index);

    param = model->param;
    param.C = 10000000000000000000.0;

    //double* w = compute_w( max_index, model->weights[0], model->weights[1] );
    double b = ( findmin(0) + findmax(1) ) / 2.0;

    printf("b = %f \n", b);

    int i;
    int correct0 = 0;
    for(i=0;i<prob[0].l;i++)
    {
	double ret = compute_wxi(0,i) - b;//dot_w(w, prob[0].x[i]) - b;
	//printf(" i = %d  result = %f  \n", i, ret);
	if( ret > 0 ) correct0++;
    }

    int correct1 = 0;
    for(i=0;i<prob[1].l;i++)
    {
	double ret = compute_wxi(1,i) - b; //dot_w(w, prob[1].x[i]) - b;
	//printf(" i = %d  result = %f  \n", i, ret);
	if( ret < 0 ) correct1++;
    }
    printf(" %d von %d , %d von %d \n", correct0, prob[0].l, correct1, prob[1].l);
}
