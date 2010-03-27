#include <stdio.h>

#include "globals.h"
#include "readsvm.h"

extern void run_cuda_kernel();

main()
{
	printf("svm on cuda started!\n");

	read_problem("data/heart_scale4");

	printf("vector dimension: %d \n", max_index);

	run_cuda_kernel();

}
