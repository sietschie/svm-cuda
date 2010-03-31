#define cutilCheckMsg(msg)           __cutilCheckMsg     (msg, __FILE__, __LINE__)

inline void __cutilCheckMsg( const char *errorMessage, const char *file, const int line )
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err)
	{
		fprintf(stderr, "cutilCheckMsg() CUTIL CUDA error: %s in file <%s>, line %i : %s.\n",
			errorMessage, file, line, cudaGetErrorString( err) );
		exit(-1);
	}
	#ifdef _DEBUG
	err = cudaThreadSynchronize();
	if( cudaSuccess != err)
	{
		fprintf(stderr, "cutilCheckMsg cudaThreadSynchronize error: %s in file <%s>, line %i : %s.\n",
			errorMessage, file, line, cudaGetErrorString( err) );
		exit(-1);
	}
	#endif
}


#  define CUDA_SAFE_CALL_NO_SYNC( call) { \
		cudaError err = call; \
		if( cudaSuccess != err) \
		{ \
			fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
			__FILE__, __LINE__, cudaGetErrorString( err) ); \
			exit(EXIT_FAILURE); \
		} \
	}

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);

#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
	if( cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() Runtime API error in file <%s>, line %i : %s.\n",
			file, line, cudaGetErrorString( err) );
		exit(-1);
	}
}
