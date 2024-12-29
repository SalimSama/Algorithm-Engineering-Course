#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <string>
#include <vector>


// Funktion zur Binarisierung eines Bildes
void binarize_image(const std::string &input_path, std::string output_path, int threshold);

// Funktion zur Binarisierung eines Bildes mit OpenMP
void binarize_image_parallel(const std::string &input_path, std::string output_path, int threshold);

// Funktion: Sauvola-Binarisierung
void sauvola_binarize(const unsigned char* gray, unsigned char* out, int width, int height, int window_size, float k, float R);

// Funktion: NICK-Binarisierung
void nick_binarize(const unsigned char* gray, unsigned char* out, int width, int height, int window_size, float k);

// Funktion: Verwendung von Sauvola und NICK
void process_advanced_binarization(const std::string &input_path);


void sauvola_binarize_integral(const unsigned char* gray, unsigned char* out, int width, int height, int window_size, float k = 0.2f, float R = 128.0f);

void process_integral_binarization(const std::string &input_path);


#endif // IMAGE_UTILS_H
