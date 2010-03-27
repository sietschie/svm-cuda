#include <stdio.h>

#include "globals.h"
#include "readsvm.h"

extern void run_cuda_kernel();

main()
{
	//nur ein test
	printf("svm on cuda started!\n");

	read_problem("data/a2a.t");

	printf("vector dimension: %d \n", max_index);

	run_cuda_kernel();

}
