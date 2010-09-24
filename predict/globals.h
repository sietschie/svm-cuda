#ifndef GLOBALS_H_INCLUDED
#define GLOBALS_H_INCLUDED

#include "../common/globals.h"

struct svm_problem prob[2];		// set by read_problem

int max_index;


struct svm_parameter param;		// set by parse_command_line


double rho;

#endif // GLOBALS_H_INCLUDED
