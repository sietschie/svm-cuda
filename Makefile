all:
	nvcc -g main.c readsvm.c cuda_main.cu
