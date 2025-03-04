#include <binarization/integral_binarization.h>
#include <utils/image_io.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <spdlog/spdlog.h>


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

void adaptive_binarize_integral(const unsigned char* gray,
                       unsigned char* out,
                       int width, int height,
                       int window_size,
                       const std::vector<double>& integralImg,
                       const std::vector<double>& integralImgSq,
                       const std::function<float(float mean, float stddev)> &threshold_func) {
    int half_win = window_size / 2;

    spdlog::info("Starting adaptive integral binarization with window size {}", window_size);

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float mean = 0.0f, stddev = 0.0f;
            local_mean_std_integral(integralImg, integralImgSq, width, height, x, y, half_win, mean, stddev);
            float threshold = threshold_func(mean, stddev);
            out[y * width + x] = (gray[y * width + x] > threshold) ? 255 : 0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("Adaptive integral binarization completed in {} seconds.", duration.count());
}

void sauvola_binarize_integral(const unsigned char* gray,
                               unsigned char* out,
                               int width, int height,
                               int window_size,
                               float k,
                               float R,
                               const std::vector<double>& integralImg,
                               const std::vector<double>& integralImgSq) {
    spdlog::info("Starting Integral Sauvola binarization with window size {}, k={}, R={}.", window_size, k, R);

    auto threshold_func = [k, R](float mean, float stddev) {
        return mean * (1.0f + k * ((stddev / R) - 1.0f));
    };

    adaptive_binarize_integral(gray, out, width, height, window_size, integralImg, integralImgSq, threshold_func);

    spdlog::info("Integral Sauvola binarization completed.");
}

void process_integral_binarization(const std::string &input_path) {
    spdlog::info("Processing integral binarization for: {}", input_path);
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
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

    std::vector<double> integralImg, integralImgSq;
    computeIntegralImages(gray.data(), width, height, integralImg, integralImgSq);

    std::string output_path_integral = make_output_path(input_path) + "_integral_sauvola.png";
    std::vector<unsigned char> output_integral(width * height);

    auto start = std::chrono::high_resolution_clock::now();

    sauvola_binarize_integral(gray.data(), output_integral.data(), width, height, 15, 0.2f, 128.0f, integralImg, integralImgSq);

    if (!write_binary_image(output_path_integral, width, height, 1, output_integral.data())) {
        spdlog::error("Failed to write Integral Sauvola output image: {}", output_path_integral);
    } else {
        spdlog::info("Integral Sauvola binarized image saved to: {}", output_path_integral);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("Integral binarization process completed in {} seconds.", duration.count());

    stbi_image_free(image);
}


