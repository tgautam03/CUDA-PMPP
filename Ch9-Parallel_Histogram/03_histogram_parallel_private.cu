#include <iostream>
#include <string>

__global__ void histogram_parallel_private(char *sentence_data_device, unsigned int sentence_len, unsigned int *histogram_device) 
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; // Which character this thread will work on

    if (i < sentence_len)
    {
        int alphabet_pos = sentence_data_device[i] - 'a'; // Position of the alphabet
        if (alphabet_pos >= 0 && alphabet_pos < 26) // Store if it is lowercase
            atomicAdd(&histogram_device[blockIdx.x*7 + alphabet_pos/4], 1); // Storing in private histogram
    }

    // Merge all private histograms into one
    if (blockIdx.x > 0)
    {
        __syncthreads(); // Ensuring all threads in a block are done computing private histograms
        // Assigning each thread to a bin in the private histogram
        for (unsigned int bin = threadIdx.x; bin < 7; bin+=blockDim.x)
        {
            unsigned int bin_value = histogram_device[blockIdx.x*7 + bin]; // Getting the value in the bin
            if (bin_value > 0) // Checking if the value is non-zero
            {
                atomicAdd(&histogram_device[bin], bin_value); // Updating the value in the global histogram (1st 7 bins)
            }
        }
    }
}

int main(int argc, char const *argv[])
{
    std::string sentence; // Input string
    std::cout << "Enter the sentence (all lowercase): ";
    std::getline(std::cin, sentence); // Input from terminal stored in the variable sentence

    // Book keeping
    const char *sentence_data_host = sentence.c_str();
    size_t sentence_len = sentence.length() + 1; // +1 for null terminator

    unsigned int num_threads_per_block = 256; // Num threads per block
    unsigned int num_blocks = ceil(sentence_len/(float)num_threads_per_block); // Total number of blocks

    // Computing histogram
    unsigned int *histogram_host = new unsigned int[7 * num_blocks];  // Histogram array with 7 bins (for each thread block) to store 1) a-d, e-h, i-l, and so on.
    for (int i = 0; i < 7 * num_blocks; i++)
        histogram_host[i] = 0;
    
    // Move sentence to GPU
    char *sentence_data_device;
    cudaMalloc((void**)&sentence_data_device, sentence_len*sizeof(char));
    cudaMemcpy(sentence_data_device, sentence_data_host, sentence_len*sizeof(char), cudaMemcpyHostToDevice);

    // Move histogram to GPU
    unsigned int *histogram_device;
    cudaMalloc((void**)&histogram_device, 7*num_blocks*sizeof(unsigned int));
    cudaMemcpy(histogram_device, histogram_host, 7*num_blocks*sizeof(unsigned int), cudaMemcpyHostToDevice);

    // For recording time
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    
    // Kernel execution
    cudaEventRecord(beg);
    histogram_parallel_private<<<num_blocks, num_threads_per_block>>>(sentence_data_device, sentence_len, histogram_device);
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time = elapsed_time * 1000.;

    std::cout << "Runtime : " << elapsed_time << " microseconds \n";

    // Move histogram result to CPU
    cudaMemcpy(histogram_host, histogram_device, 7*num_blocks*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Display results
    std::cout << "\n--- Character Frequency Histogram ---\n";
    std::cout << "-----------------------------------\n";

    // Define the labels for each bin
    const std::string labels[] = {"a-d", "e-h", "i-l", "m-p", "q-t", "u-x", "y-z"};

    for (int i = 0; i < 7; ++i)
    {
        // Print each label and its corresponding count on a new line
        std::cout << "Range " << labels[i] << ": " << histogram_host[i] << std::endl;
    }
    std::cout << "-----------------------------------\n";

    return 0;
}
