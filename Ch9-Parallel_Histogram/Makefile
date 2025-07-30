CC_cpu = g++
CC_gpu = nvcc -gencode arch=compute_86,code=sm_86

01_histogram_sequential: 01_histogram_sequential.cpp
	@$(CC_cpu) 01_histogram_sequential.cpp -o 01_histogram_sequential.out

02_histogram_parallel_atomic: 02_histogram_parallel_atomic.cu
	@$(CC_gpu) 02_histogram_parallel_atomic.cu -o 02_histogram_parallel_atomic.out

03_histogram_parallel_private: 03_histogram_parallel_private.cu
	@$(CC_gpu) 03_histogram_parallel_private.cu -o 03_histogram_parallel_private.out

04_histogram_parallel_shared: 04_histogram_parallel_shared.cu
	@$(CC_gpu) 04_histogram_parallel_shared.cu -o 04_histogram_parallel_shared.out

05_histogram_parallel_coarse: 05_histogram_parallel_coarse.cu
	@$(CC_gpu) 05_histogram_parallel_coarse.cu -o 05_histogram_parallel_coarse.out

06_histogram_parallel_aggregate: 06_histogram_parallel_aggregate.cu
	@$(CC_gpu) 06_histogram_parallel_aggregate.cu -o 06_histogram_parallel_aggregate.out

clean:
	@echo "Removing object files..."
	rm -r *.out