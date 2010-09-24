#ifndef READSVM_H_INCLUDED
#define READSVM_H_INCLUDED

#include "globals.h"


void read_problem(const char *filename, 
				struct svm_problem *prob, 
				struct svm_parameter *param, 
				int *max_index);

int svm_save_model(const char *model_file_name, 
				const struct svm_model* model,
				struct svm_problem *prob);


#endif // READSVM_H_INCLUDED
