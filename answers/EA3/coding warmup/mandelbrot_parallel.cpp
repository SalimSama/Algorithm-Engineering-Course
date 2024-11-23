#include <complex>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

constexpr int width = 4000;
constexpr int height = width;

int compute_pixel(int x, int y) { // compute the gray value of a pixel
    complex<double> point(2.0 * x / width - 1.5, 2.0 * y / height - 1.0);
    complex<double> z(0, 0);
    constexpr int iterations = 100;
    int nb_iter = 0;
    while (abs(z) < 2 && nb_iter < iterations) {
        z = z * z + point;
        nb_iter++;
    }
    return (255 * nb_iter) / iterations;
}

int main() { // generate mandelbrot pgm (portable graymap)
    const string image_name = "mandelbrot.pgm";
    remove(image_name.c_str());
    const double start = omp_get_wtime();

    // Buffer to store pixel values
    vector<int> buffer(width * height);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            buffer[i * width + j] = compute_pixel(j, i);
        }
    }

    ofstream image(image_name); // Write to file after computation
    if (image.is_open()) {
        image << "P2\n" << width << " " << height << " 255\n"; // pgm header
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                image << buffer[i * width + j] << "\n";
            }
        }
        image.close();
    } else {
        cout << "Could not open the file!";
    }
    cout << omp_get_wtime() - start << " seconds" << endl;
}

