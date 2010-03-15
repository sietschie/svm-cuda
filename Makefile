all:
	nvcc -g -G main.c readsvm.c cuda_main.cu
