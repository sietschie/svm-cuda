#include <stdio.h>

#include "globals.h"
#include "readsvm.h"

extern void run_cuda_kernel();

main()
{
	printf("hello world!\n");

	read_problem("data/a2a.t");

	printf(" %d \n", prob[0].l);

	printf("max_index = %d \n", max_index);

	run_cuda_kernel();

}
