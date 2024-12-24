#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <string>
#include <vector>

// Funktion zum Schreiben eines ASCII-PPM
bool write_ppm_ascii(const std::string &filename, int width, int height, int channels, const unsigned char *data);

// Funktion zum Erstellen des Ausgabe-Pfads
std::string make_output_path(const std::string &input_path);

// Funktion zum Schreiben eines binarisierten Bildes
bool write_binary_image(const std::string &filename, int width, int height, int channels, const unsigned char *data);

// Funktion zur Binarisierung eines Bildes
void binarize_image(const std::string &input_path, std::string output_path, int threshold);

#endif // IMAGE_UTILS_H
