#include <iostream>
#include <string>
#include <chrono>

void histogram_sequential(std::string const sentence, unsigned int *histogram)
{
    for (unsigned int i = 0; i < sentence.length(); i++)
    {
        int alphabet_pos = sentence[i] - 'a'; // Position of the alphabet
        if (alphabet_pos >= 0 && alphabet_pos < 26)
        {
            histogram[alphabet_pos/4] += 1;
        }
    }
}

int main(int argc, char const *argv[])
{
    std::string sentence; // Input string
    std::cout << "Enter the sentence (all lowercase): ";
    std::getline(std::cin, sentence); // Input from terminal stored in the variable sentence

    // Computing histogram
    unsigned int histogram[7] = {0}; // Histogram array with 7 bins to store 1) a-d, e-h, i-l, and so on.
    
    auto start_time = std::chrono::high_resolution_clock::now();
    histogram_sequential(sentence, histogram);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "Runtime (should be O(N)): " << duration.count() << " microseconds";

    // Display results
    std::cout << "\n--- Character Frequency Histogram ---\n";
    std::cout << "-----------------------------------\n";

    // Define the labels for each bin
    const std::string labels[] = {"a-d", "e-h", "i-l", "m-p", "q-t", "u-x", "y-z"};

    for (int i = 0; i < 7; ++i)
    {
        // Print each label and its corresponding count on a new line
        std::cout << "Range " << labels[i] << ": " << histogram[i] << std::endl;
    }
    std::cout << "-----------------------------------\n";

    return 0;
}
