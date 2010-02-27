all:
	nvcc -deviceemu main.c readsvm.c cuda_main.cu
