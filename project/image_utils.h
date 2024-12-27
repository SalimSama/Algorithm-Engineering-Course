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

// Funktion zur Binarisierung eines Bildes mit OpenMP
void binarize_image_parallel(const std::string &input_path, std::string output_path, int threshold);

// Hilfsfunktion: lokales Mittel und Standardabweichung berechnen
void local_mean_std(const unsigned char* gray, int width, int height, int x, int y, int half_win, float &mean, float &stddev);

// Funktion: Sauvola-Binarisierung
void sauvola_binarize(const unsigned char* gray, unsigned char* out, int width, int height, int window_size, float k, float R);

// Funktion: NICK-Binarisierung
void nick_binarize(const unsigned char* gray, unsigned char* out, int width, int height, int window_size, float k);

// Funktion: Verwendung von Sauvola und NICK
void process_advanced_binarization(const std::string &input_path);

void computeIntegralImages(const unsigned char* gray,
                           int width, int height,
                           std::vector<double>& integralImg,
                           std::vector<double>& integralImgSq);

inline double getSum(const std::vector<double>& integralImg,
                     int x1, int y1, int x2, int y2, int width, int height);

void local_mean_std_integral(const std::vector<double>& integralImg,
                             const std::vector<double>& integralImgSq,
                             int width, int height,
                             int x, int y, int half_win,
                             float &mean, float &stddev);

void sauvola_binarize_integral(const unsigned char* gray,
                               unsigned char* out,
                               int width, int height,
                               int window_size,
                               float k = 0.2f,
                               float R = 128.0f);

void process_integral_binarization(const std::string &input_path);


#endif // IMAGE_UTILS_H
