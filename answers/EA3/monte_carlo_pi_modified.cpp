#include <iostream>
#include <iomanip>  // Include this header for setprecision
#include <omp.h>

using namespace std;

// Lightweight deterministic random number generator
double rnd(unsigned int *seed) {
    *seed = (1140671485 * (*seed) + 12820163) % (1 << 24);
    return ((double)(*seed)) / (1 << 24);
}

int main() {
    int n = 1000000000; // number of points to generate
    int counter = 0; // counter for points lying in the first quadrant of a unit circle
    auto start_time = omp_get_wtime(); // omp_get_wtime() is an OpenMP library routine

#pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num(); // Seed for each thread
        int local_counter = 0; // Local counter for each thread

#pragma omp for
        for (int i = 0; i < n; ++i) {
            double x = rnd(&seed); // Generate random number between 0.0 and 1.0
            double y = rnd(&seed); // Generate random number between 0.0 and 1.0
            if (x * x + y * y <= 1.0) { // If the point lies in the first quadrant of a unit circle
                ++local_counter;
            }
        }

#pragma omp atomic
        counter += local_counter; // Safely update the global counter
    }

    auto run_time = omp_get_wtime() - start_time;
    auto pi = 4 * (double(counter) / n);

    cout << fixed << setprecision(15); // Set precision to 15 decimal places
    cout << "pi: " << pi << endl;
    cout << "run_time: " << run_time << " s" << endl;
    cout << "n: " << n << endl;
}

