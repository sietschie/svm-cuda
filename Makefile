all:
	nvcc -g -deviceemu main.c readsvm.c cuda_main.cu
