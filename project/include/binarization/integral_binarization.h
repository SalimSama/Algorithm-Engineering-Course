#ifndef INTEGRAL_BINARIZATION_H
#define INTEGRAL_BINARIZATION_H

#include <string>

// Sauvola-Binarisierung mit Integralbildern
void sauvola_binarize_integral(const unsigned char* gray, unsigned char* out, int width, int height, int window_size, float k = 0.2f, float R = 128.0f);

// Prozess zur Berechnung von Integralbildern und Durchführung der Binarisierung
void process_integral_binarization(const std::string &input_path);

#endif // INTEGRAL_BINARIZATION_H
