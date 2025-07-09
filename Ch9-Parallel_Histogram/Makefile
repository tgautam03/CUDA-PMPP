CC_cpu = g++
CC_gpu = nvcc -gencode arch=compute_86,code=sm_86

1_histogram_sequential: 1_histogram_sequential.cpp
	@$(CC_cpu) 1_histogram_sequential.cpp -o 1_histogram_sequential.out

2_histogram_parallel_atomic: 2_histogram_parallel_atomic.cu
	@$(CC_gpu) 2_histogram_parallel_atomic.cu -o 2_histogram_parallel_atomic.out

clean:
	@echo "Removing object files..."
	rm -r *.out