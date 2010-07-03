#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "globals.h"
#include "readsvm.h"

extern void run_cuda_kernel();
struct svm_parameter param;		// set by parse_command_line

void exit_with_help()
{
	printf(
	"Usage: svm-train [options] training_set_file [model_file]\n"
	"options:\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_set_file)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/k)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of C-SVC (default 1)\n"
	"-m cachesize : set cache size (default 10)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-v level : set verbosity level (default 1)\n"
	);
	exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;

	// default values
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.cache_size = 10;
	param.C = 1;
	param.eps = 1e-3;
	param.verbosity = 1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'v':
				param.verbosity = atoi(argv[i]);
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	// determine filenames

	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
}

int main (int argc, char ** argv)
{
    char input_filename[1024];
    char model_filename[1024];

    parse_command_line(argc, argv, input_filename, model_filename);

	read_problem(input_filename);
    if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.verbosity >= 1)
	{
		printf("vector dimension: %d \n", max_index);
		printf(" number of vectors in class 1 = %d and in class 2 = %d \n", prob[0].l, prob[1].l);
	}

	run_cuda_kernel(param);

	return 0;
}
