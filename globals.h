#ifndef GLOBALS_H_INCLUDED
#define GLOBALS_H_INCLUDED

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID }; //, PRECOMPUTED }; /* kernel_type */

struct svm_node
{
	int index;
	double value;
};

struct svm_problem
{
	int l;
	double *y;
	struct svm_node **x;
};


struct svm_parameter
{
	int kernel_type;
	int degree;	/* for poly */
	float gamma;	/* for poly/rbf/sigmoid */
	float coef0;	/* for poly/sigmoid */

	/* these are for training only */
	int cache_size; /* in MB */
	float eps;	/* stopping criteria */
	float C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int verbosity;
	int maximum_iterations;

};


//
// svm_model
//
struct svm_model
{
	struct svm_parameter param;	// parameter
	int nr_class;		// number of classes, = 2 in regression/one class svm
	int l;			// total #SV
	double rho;		// constants in decision functions (rho[k*(k-1)/2])

	// for classification only

	int label[2];		// label of each class (label[k])
	int nSV[2];		// number of SVs for each class (nSV[k])
	float* weights[2]; //gewichte
	struct svm_node **SV[2]; //daten
};

struct svm_problem prob[2];		// set by read_problem

int max_index;

struct svm_node *x_space[2];
int elements[2]; //todo: globals aufteilen in datenstrukturen und daten

#endif // GLOBALS_H_INCLUDED
