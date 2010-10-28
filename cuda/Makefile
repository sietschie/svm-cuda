all:
	nvcc -O -arch sm_13 main.c ../common/readsvm.c cuda_main.cu -o ../svm-cuda-train
	nvcc -O -DNO_EPSILON -arch sm_13 main.c ../common/readsvm.c cuda_main.cu -o ../svm-cuda-train-noeps
