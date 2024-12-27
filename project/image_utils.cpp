#include <image_utils.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <chrono>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>


// Hilfsfunktion zum Schreiben eines ASCII-PPM
bool write_ppm_ascii(const std::string &filename, int width, int height, int channels, const unsigned char *data) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        return false;
    }

    // ASCII-PPM Header (P3)
    ofs << "P3\n" << width << " " << height << "\n255\n";

    // Für jeden Pixel die ersten drei Kanäle (RGB) schreiben
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            // Falls channels==4, ignorieren wir Alpha
            unsigned char r = data[idx + 0];
            unsigned char g = data[idx + 1];
            unsigned char b = data[idx + 2];
            ofs << (int)r << " " << (int)g << " " << (int)b << "\n";
        }
    }
    return true;
}

// Hilfsfunktion, um sicherzustellen, dass kein Slash vorangestellt wird,
// wenn parent_path() leer ist. So landen wir nicht im Root-Verzeichnis "/...".
std::string make_output_path(const std::string &input_path) {
    namespace fs = std::filesystem;
    fs::path p(input_path);
    std::string stem = p.stem().string();
    std::string ext = p.extension().string();
    std::string parent = p.parent_path().string();

    if (parent.empty()) {
        return stem + "_bin" + ext;
    } else {
        return parent + "/" + stem + "_bin" + ext;
    }
}

// Schreibt das binarisierte Bild als PNG/JPG oder ASCII-PPM.
bool write_binary_image(const std::string &filename, int width, int height, int channels, const unsigned char *data) {
    std::string extension = std::filesystem::path(filename).extension().string();
    for (auto &c : extension) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    if (extension == ".png") {
        return stbi_write_png(filename.c_str(), width, height, channels, data, width * channels);
    } else if (extension == ".jpg" || extension == ".jpeg") {
        return stbi_write_jpg(filename.c_str(), width, height, channels, data, 90);
    } else if (extension == ".ppm") {
        return write_ppm_ascii(filename, width, height, channels, data);
    }
    // Fallback
    return stbi_write_png(filename.c_str(), width, height, channels, data, width * channels);
}

void binarize_image(const std::string &input_path, std::string output_path, int threshold) {
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        std::cerr << "Error: Failed to load image!" << std::endl;
        return;
    }

    if (output_path.empty()) {
        output_path = make_output_path(input_path);
    }

    std::vector<unsigned char> out(width * height * channels);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < width * height; i++) {
        int idx = i * channels;
        unsigned char lum = static_cast<unsigned char>(
            0.2126f * image[idx] + 0.7152f * image[idx + 1] + 0.0722f * image[idx + 2]
        );
        unsigned char binary = (lum > threshold) ? 255 : 0;

        for (int c = 0; c < channels; c++) {
            out[idx + c] = binary;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Sequential binarization time: " << duration.count() << " seconds." << std::endl;

    if (!write_binary_image(output_path, width, height, channels, out.data())) {
        std::cerr << "Error: Failed to write image! " << output_path << std::endl;
    } else {
        std::cout << "Binarized image saved to: " << output_path << std::endl;
    }
    stbi_image_free(image);
}

void binarize_image_parallel(const std::string &input_path, std::string output_path, int threshold) {
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        std::cerr << "Error: Failed to load image!" << std::endl;
        return;
    }

    if (output_path.empty()) {
        output_path = make_output_path(input_path);
    }

    std::vector<unsigned char> out(width * height * channels);

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for simd
    for (int i = 0; i < width * height; i++) {
        int idx = i * channels;
        unsigned char lum = static_cast<unsigned char>(
            0.2126f * image[idx] + 0.7152f * image[idx + 1] + 0.0722f * image[idx + 2]
        );
        unsigned char binary = (lum > threshold) ? 255 : 0;

        for (int c = 0; c < channels; c++) {
            out[idx + c] = binary;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Parallel binarization time: " << duration.count() << " seconds." << std::endl;

    if (!write_binary_image(output_path, width, height, channels, out.data())) {
        std::cerr << "Error: Failed to write image! " << output_path << std::endl;
    } else {
        std::cout << "Parallel binarized image saved to: " << output_path << std::endl;
    }
    stbi_image_free(image);
}

//bessere Methoden der Binarisierung:

// Beispiel: Lokale Binarisierung nach Sauvola
// T(x, y) = m(x, y) * [1 + k * (s(x, y)/R - 1)]
// m(x, y): Lokaler Mittelwert
// s(x, y): Lokale Standardabweichung
// R: Dynamischer Bereich der Standardabweichung (oft 128)
// k: Empirischer Faktor (z.B. 0.2 - 0.5)

// Beispiel: Lokale Binarisierung nach Nick:
// T(x, y) = m(x, y) + k * ( sqrt( (G² - n*m²)/n ) )
// m(x, y) = lokaler Mittelwert in Fenstergroesse
// G = Summe aller Grauwerte im lokalen Fenster
// n = Anzahl Pixel im Fenster
// k z.B. um 0.1 bis 0.2

// Hilfsfunktion: lokales Mittel und Standardabweichung berechnen
// (einfach, aber ineffizient; besser Integralbilder verwenden)
void local_mean_std(const unsigned char* gray,
                    int width, int height,
                    int x, int y,
                    int half_win,
                    float &mean, float &stddev)
{
    int count = 0;
    float sum = 0.0f, sum_sq = 0.0f;
    for (int dy = -half_win; dy <= half_win; dy++) {
        for (int dx = -half_win; dx <= half_win; dx++) {
            int yy = y + dy;
            int xx = x + dx;
            if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
                unsigned char val = gray[yy * width + xx];
                sum += val;
                sum_sq += val * val;
                count++;
            }
        }
    }
    mean = sum / count;
    float var = (sum_sq / count) - (mean * mean);
    stddev = (var > 0) ? std::sqrt(var) : 0.0f;
}

// Beispiel-Funktion: Sauvola-Binarisierung
void sauvola_binarize(const unsigned char* gray,
                      unsigned char* out,
                      int width, int height,
                      int window_size,
                      float k=0.2f,   // Empirischer Faktor
                      float R=128.0f // Dynamischer Bereich
) {
    int half_win = window_size / 2;

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float mean = 0.0f, stddev = 0.0f;
            local_mean_std(gray, width, height, x, y, half_win, mean, stddev);
            float threshold = mean * (1.0f + k * ((stddev / R) - 1.0f));
            out[y * width + x] = (gray[y * width + x] > threshold) ? 255 : 0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Sauvola binarization time: " << duration.count() << " seconds." << std::endl;
}


// Beispiel-Funktion: NICK-Binarisierung
void nick_binarize(const unsigned char* gray,
                   unsigned char* out,
                   int width, int height,
                   int window_size,
                   float k=0.1f) {
    int half_win = window_size / 2;

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Summen für Nick berechnen
            float sum = 0.0f;
            int count = 0;
            for (int dy = -half_win; dy <= half_win; dy++) {
                for (int dx = -half_win; dx <= half_win; dx++) {
                    int yy = y + dy;
                    int xx = x + dx;
                    if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
                        sum += gray[yy * width + xx];
                        count++;
                    }
                }
            }
            float mean = sum / count;
            // Varianz / Standardabweichung
            float sum_sq = 0.0f;
            for (int dy = -half_win; dy <= half_win; dy++) {
                for (int dx = -half_win; dx <= half_win; dx++) {
                    int yy = y + dy;
                    int xx = x + dx;
                    if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
                        float val = gray[yy * width + xx];
                        sum_sq += (val - mean) * (val - mean);
                    }
                }
            }
            float variance = sum_sq / count;
            float stddev = std::sqrt(variance);
            // Nick-Formel: T = m + k * sqrt( (G^2 - n*m^2)/n )
            float T = mean + k * stddev;
            out[y * width + x] = (gray[y * width + x] > T) ? 255 : 0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Nick binarization time: " << duration.count() << " seconds." << std::endl;
}


void process_advanced_binarization(const std::string &input_path) {
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        std::cerr << "Error: Failed to load image!" << std::endl;
        return;
    }

    // Grauwert-Umwandlung
    std::vector<unsigned char> gray(width * height);
#pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        gray[i] = static_cast<unsigned char>(
                0.2126f * image[i * channels + 0] +
                0.7152f * image[i * channels + 1] +
                0.0722f * image[i * channels + 2]);
    }

    // Dynamische Pfade für die Ausgabedateien
    std::string output_path_sauvola = make_output_path(input_path) + "_sauvola.png";
    std::string output_path_nick = make_output_path(input_path) + "_nick.png";

    // Sauvola-Binarisierung
    std::vector<unsigned char> output_sauvola(width * height);
    sauvola_binarize(gray.data(), output_sauvola.data(), width, height, 15, 0.2f, 128.0f);

    if (!write_binary_image(output_path_sauvola, width, height, 1, output_sauvola.data())) {
        std::cerr << "Error: Failed to write Sauvola output image!" << std::endl;
    } else {
        std::cout << "Sauvola binarized image saved to: " << output_path_sauvola << std::endl;
    }

    // Nick-Binarisierung
    std::vector<unsigned char> output_nick(width * height);
    nick_binarize(gray.data(), output_nick.data(), width, height, 15, 0.1f);

    if (!write_binary_image(output_path_nick, width, height, 1, output_nick.data())) {
        std::cerr << "Error: Failed to write Nick output image!" << std::endl;
    } else {
        std::cout << "Nick binarized image saved to: " << output_path_nick << std::endl;
    }

    stbi_image_free(image);
}



