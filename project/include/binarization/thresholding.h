#ifndef THRESHOLDING_H
#define THRESHOLDING_H

#include <string>

// Klassische Schwellenwert-Binarisierung (sequentiell)
void binarize_image(const std::string &input_path, std::string output_path, int threshold);

// Parallele Schwellenwert-Binarisierung mit OpenMP
void binarize_image_parallel(const std::string &input_path, std::string output_path, int threshold);

#endif // THRESHOLDING_H
