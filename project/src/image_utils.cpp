#include <../include/image_utils.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <chrono>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <../include/stb_image.h>
#include <../include/stb_image_write.h>


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

std::string make_output_path(const std::string &input_path) {
    namespace fs = std::filesystem;
    fs::path p(input_path);
    std::string stem = p.stem().string();
    std::string ext = p.extension().string();

    fs::path results_dir = "Results";
    if (!fs::exists(results_dir)) {
        fs::create_directory(results_dir);
    }

    return (results_dir / (stem + "_bin" + ext)).string();
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



void computeIntegralImages(const unsigned char* gray,
                           int width, int height,
                           std::vector<double>& integralImg,
                           std::vector<double>& integralImgSq)
{
    integralImg.resize(width * height, 0.0);
    integralImgSq.resize(width * height, 0.0);

    // 1. Zeilenweiser Durchgang (Prefix-Sums pro Zeile)
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        double sumRow = 0.0, sumRowSq = 0.0;
        for (int x = 0; x < width; x++) {
            double val = static_cast<double>(gray[y * width + x]);
            sumRow   += val;
            sumRowSq += val * val;
            integralImg[y * width + x]   = sumRow;
            integralImgSq[y * width + x] = sumRowSq;
        }
    }

    // 2. Spaltenweiser Durchgang (Prefix-Sums der bereits zeilenweise summierten Daten)
#pragma omp parallel for
    for (int x = 0; x < width; x++) {
        double colSum = 0.0, colSumSq = 0.0;
        for (int y = 0; y < height; y++) {
            colSum   += integralImg[y * width + x];
            colSumSq += integralImgSq[y * width + x];
            integralImg[y * width + x]   = colSum;
            integralImgSq[y * width + x] = colSumSq;
        }
    }
}

inline double getSum(const std::vector<double>& integralImg,
                     int x1, int y1, int x2, int y2, int width, int height)
{
    if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
    if (x2 >= width) x2 = width - 1;
    if (y2 >= height) y2 = height - 1;

    double A = (x1 > 0 && y1 > 0) ? integralImg[(y1 - 1) * width + (x1 - 1)] : 0.0;
    double B = (y1 > 0) ? integralImg[(y1 - 1) * width + x2] : 0.0;
    double C = (x1 > 0) ? integralImg[y2 * width + (x1 - 1)] : 0.0;
    double D = integralImg[y2 * width + x2];
    return D + A - B - C;
}

void local_mean_std_integral(const std::vector<double>& integralImg,
                             const std::vector<double>& integralImgSq,
                             int width, int height,
                             int x, int y, int half_win,
                             float &mean, float &stddev)
{
    int x1 = x - half_win, y1 = y - half_win;
    int x2 = x + half_win, y2 = y + half_win;
    double area = (x2 - x1 + 1) * (y2 - y1 + 1);

    double sum = getSum(integralImg, x1, y1, x2, y2, width, height);
    double sumSq = getSum(integralImgSq, x1, y1, x2, y2, width, height);

    double m = sum / area;
    double var = (sumSq / area) - (m * m);
    mean = static_cast<float>(m);
    stddev = (var > 0.0) ? static_cast<float>(std::sqrt(var)) : 0.0f;
}

void sauvola_binarize_integral(const unsigned char* gray,
                               unsigned char* out,
                               int width, int height,
                               int window_size,
                               float k,
                               float R)
{
    std::vector<double> integralImg, integralImgSq;
    computeIntegralImages(gray, width, height, integralImg, integralImgSq);

    int half_win = window_size / 2;
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float mean = 0.f, stddev = 0.f;
            local_mean_std_integral(integralImg, integralImgSq, width, height, x, y, half_win, mean, stddev);
            float threshold = mean * (1.0f + k * ((stddev / R) - 1.0f));
            out[y * width + x] = (gray[y * width + x] > threshold) ? 255 : 0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Integral Sauvola binarization time: " << duration.count() << " seconds." << std::endl;
}

void process_integral_binarization(const std::string &input_path) {
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        std::cerr << "Error: Failed to load image!" << std::endl;
        return;
    }

    std::vector<unsigned char> gray(width * height);
#pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        gray[i] = static_cast<unsigned char>(
                0.2126f * image[i * channels + 0] +
                0.7152f * image[i * channels + 1] +
                0.0722f * image[i * channels + 2]);
    }

    std::string output_path_integral = make_output_path(input_path) + "_integral_sauvola.png";
    std::vector<unsigned char> output_integral(width * height);
    sauvola_binarize_integral(gray.data(), output_integral.data(), width, height, 15, 0.2f, 128.0f);

    if (!write_binary_image(output_path_integral, width, height, 1, output_integral.data())) {
        std::cerr << "Error: Failed to write Integral Sauvola output image!" << std::endl;
    } else {
        std::cout << "Integral Sauvola binarized image saved to: " << output_path_integral << std::endl;
    }

    stbi_image_free(image);
}



