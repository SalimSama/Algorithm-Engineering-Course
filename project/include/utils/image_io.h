#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <string>

// Bild einlesen (RGB oder Graustufen)
bool write_ppm_ascii(const std::string &filename, int width, int height, int channels, const unsigned char *data);

// Graustufen-Umwandlung
bool write_binary_image(const std::string &filename, int width, int height, int channels, const unsigned char *data);

// Hilfsfunktion zur Generierung des Ausgabepfads
std::string make_output_path(const std::string &input_path);

#endif // IMAGE_IO_H
