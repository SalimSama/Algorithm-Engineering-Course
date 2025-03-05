#ifndef ADAPTIVE_THRESHOLDING_H
#define ADAPTIVE_THRESHOLDING_H

#include <string>

// Sauvola-Binarisierung
void sauvola_binarize(const unsigned char* gray, unsigned char* out, int width, int height, int window_size, float k, float R);

// NICK-Binarisierung
void nick_binarize(const unsigned char* gray, unsigned char* out, int width, int height, int window_size, float k);

// Prozess zur Ausführung von Sauvola und NICK-Binarisierung
void process_advanced_binarization(const std::string &input_path,int window_size, float k, float R);

#endif // ADAPTIVE_THRESHOLDING_H
