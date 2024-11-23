#include <iostream>
#include <vector>
#include <limits>
#include <omp.h>
#include <parallel/algorithm>
#include <fstream>
#include <sstream>
#include <cstdlib>

// compute the average of two integers without overflow
inline int64_t average(int64_t a, int64_t b) {
    return (a & b) + ((a ^ b) >> 1);
}

// partitioning function for quicksort
inline int64_t
partition(int64_t *arr, int64_t left, int64_t right, int64_t pivot, int64_t &smallest, int64_t &biggest) {
    int64_t *left_ptr = &arr[left];
    int64_t *right_ptr = &arr[right];
    while (left_ptr < right_ptr) {
        smallest = (*left_ptr < smallest) ? *left_ptr : smallest;
        biggest = (*left_ptr > biggest) ? *left_ptr : biggest;
        if (*left_ptr > pivot) {
            --right_ptr;
            std::swap(*left_ptr, *right_ptr);
        } else {
            ++left_ptr;
        }
    }
    return left_ptr - arr;
}

inline void insertion_sort(int64_t *arr, int64_t left, int64_t right) {
    for (int64_t i = left + 1; i <= right; i++) {
        int64_t key = arr[i];
        int64_t j = i - 1;
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

// the core recursive quicksort function
void qs_core(int64_t *arr, int64_t left, int64_t right, const int64_t pivot) {
    if (right - left < 32) {
        insertion_sort(arr, left, right);
        return;
    }

    int64_t smallest = std::numeric_limits<int64_t>::max();
    int64_t biggest = std::numeric_limits<int64_t>::min();
    int64_t bound = partition(arr, left, right + 1, pivot, smallest, biggest);

    if (smallest == biggest)
        return;

#pragma omp task final(bound - left < 10000)
    qs_core(arr, left, bound - 1, average(smallest, pivot));
    qs_core(arr, bound, right, average(pivot, biggest));
}

// wrapper for the quicksort function
void min_max_quicksort(int64_t *arr, int64_t n, int num_threads = omp_get_max_threads()) {
#pragma omp parallel num_threads(num_threads)
#pragma omp single nowait
    qs_core(arr, 0, n - 1, n > 0 ? arr[average(0, n - 1)] : 0);
}

// Class for generating pseudo-random numbers using the Xoroshiro128Plus algorithm
class Xoroshiro128Plus {
    uint64_t state[2]{};

    // Function to perform bitwise rotation to the left
    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:
    // Constructor to initialize the state with a seed
    explicit Xoroshiro128Plus(uint64_t seed = 0) {
        state[0] = (12345678901234567 + seed) | 0b1001000010000001000100101000000110010010100000011001001010000001ULL;
        state[1] = (98765432109876543 + seed) | 0b0100000011001100100000011001001010000000100100101000000110010010ULL;
        for(int i = 0; i < 10; i++) { operator()(); } // Warm-up to ensure better randomness
    }

    // Function to generate the next random number in the sequence
    uint64_t operator()() {
        const uint64_t s0 = state[0];
        uint64_t s1 = state[1];
        const uint64_t result = s0 + s1;

        s1 ^= s0;
        state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // Update state[0]
        state[1] = rotl(s1, 37);                  // Update state[1]
        return result;
    }
};

// Function to benchmark sorting algorithms
void benchmark_sorts(const std::vector<int64_t> &sizes, const std::vector<int> &num_threads_vec) {
    // Open a file to write the benchmarking results
    std::ofstream result_file("benchmark_results.csv");
    result_file << "ArraySize,NumThreads,StdSort,MinMaxQuicksort,GnuParallelSort" << std::endl;

    // Loop over different array sizes to benchmark
    for (const auto &size : sizes) {
        std::vector<int64_t> data(size);
        Xoroshiro128Plus generator(size);
        for (int64_t i = 0; i < size; ++i) {
            data[i] = generator(); // Fill the array with random numbers
        }

        // Loop over different numbers of threads for parallel benchmarking
        for (const auto &num_threads : num_threads_vec) {
            omp_set_num_threads(num_threads); // Set the number of threads to use

            // Create copies of the original data for each sorting method
            std::vector<int64_t> data_copy1 = data;
            std::vector<int64_t> data_copy2 = data;
            std::vector<int64_t> data_copy3 = data;

            // Measure the time taken by std::sort
            double start = omp_get_wtime();
            std::sort(data_copy1.begin(), data_copy1.end());
            double end = omp_get_wtime();
            double time_std_sort = end - start;

            // Measure the time taken by min_max_quicksort
            start = omp_get_wtime();
            min_max_quicksort(&data_copy2[0], size, num_threads);
            end = omp_get_wtime();
            double time_min_max = end - start;

            // Measure the time taken by __gnu_parallel::sort
            start = omp_get_wtime();
            __gnu_parallel::sort(data_copy3.begin(), data_copy3.end());
            end = omp_get_wtime();
            double time_gnu_parallel = end - start;

            // Write the benchmark results to the CSV file
            result_file << size << "," << num_threads << "," << time_std_sort << "," << time_min_max << "," << time_gnu_parallel << std::endl;
        }
    }
    result_file.close(); // Close the CSV file
}

bool verify_qs_correctness(int64_t size) {
    Xoroshiro128Plus generator(size);
    std::vector<int64_t> data(size);
    for (int64_t i = 0; i < size; ++i) {
        data[i] = generator();
    }
    std::vector<int64_t> data_copy = data;  // Duplicate for std::sort
    min_max_quicksort(&data[0], size);
    std::sort(data_copy.begin(), data_copy.end());
    return data == data_copy;  // check if arrays are equal
}

int main() {
    // Define the array sizes to benchmark
    std::vector<int64_t> array_sizes = {1000000, 10000000, 50000000, 100000000, 150000000}; // Different sizes for benchmarking
    // Define the number of threads to use for benchmarking
    std::vector<int> num_threads_vec = {1, 2, 4, 8, 12}; // Different numbers of threads to benchmark

    // Print the details of the benchmarking environment
    std::cout << "Benchmarking Environment:\n";
    std::cout << "Operating System: Windows 11 Pro (23H2 22631.4541)\n";
    std::cout << "CPU: 12th Gen Intel(R) Core(TM) i5-12500H  @ 2.50 GHz\n";
    std::cout << "RAM: 16GB\n";
    std::cout << "C++ Version: 20\n";
    std::cout << "Compiler: g++ 13.2.0\n";


    // Call the benchmark function with the defined parameters
    benchmark_sorts(array_sizes, num_threads_vec);

    std::cout << "\nResults written to benchmark_results.csv" << std::endl;


    return 0;
}
