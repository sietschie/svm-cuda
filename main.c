#include <stdio.h>

#include "globals.h"
#include "readsvm.h"

extern void run_cuda_kernel();

main()
{
	printf("hello world!\n");

	read_problem("triangle");

	printf(" %d \n", prob[0].l);

	run_cuda_kernel();

}
