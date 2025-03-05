#include <binarization/adaptive_thresholding.h>
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
void adaptive_binarize(const unsigned char* gray,
                       unsigned char* out,
                       int width, int height,
                       int window_size,
                       const std::function<float(float mean, float stddev)> &threshold_func) {
    int half_win = window_size / 2;

    spdlog::info("Starting adaptive binarization with window size {}");

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float mean = 0.0f, stddev = 0.0f;
            local_mean_std(gray, width, height, x, y, half_win, mean, stddev);
            float threshold = threshold_func(mean, stddev);
            out[y * width + x] = (gray[y * width + x] > threshold) ? 255 : 0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("Adaptive binarization completed in {} seconds.", duration.count());
}

void sauvola_binarize(const unsigned char* gray,
                      unsigned char* out,
                      int width, int height,
                      int window_size,
                      float k,
                      float R) {
    spdlog::info("Starting Sauvola binarization with window size {}, k={}, R={}.", window_size, k, R);

    auto threshold_func = [k, R](float mean, float stddev) {
        return mean * (1.0f + k * ((stddev / R) - 1.0f));
    };

    adaptive_binarize(gray, out, width, height, window_size, threshold_func);

    spdlog::info("Sauvola binarization completed.");
}

void nick_binarize(const unsigned char* gray,
                   unsigned char* out,
                   int width, int height,
                   int window_size,
                   float k) {
    spdlog::info("Starting Nick binarization with window size {}, k={}.", window_size, k);

    auto threshold_func = [k](float mean, float stddev) {
        return mean + k * stddev;
    };

    adaptive_binarize(gray, out, width, height, window_size, threshold_func);

    spdlog::info("Nick binarization completed.");
}

void process_advanced_binarization(const std::string &input_path, int window_size, float k, float R) {
    spdlog::info("Processing advanced binarization for: {} with window size {}, k={}, R={}", input_path, window_size, k, R);
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

    std::string output_path_sauvola = make_output_path(input_path) + "_sauvola.png";
    std::string output_path_nick = make_output_path(input_path) + "_nick.png";

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> output_sauvola(width * height);
    sauvola_binarize(gray.data(), output_sauvola.data(), width, height, window_size, k, R);

    if (!write_binary_image(output_path_sauvola, width, height, 1, output_sauvola.data())) {
        spdlog::error("Failed to write Sauvola output image: {}", output_path_sauvola);
    } else {
        spdlog::info("Sauvola binarized image saved to: {}", output_path_sauvola);
    }

    std::vector<unsigned char> output_nick(width * height);
    nick_binarize(gray.data(), output_nick.data(), width, height, window_size, k);

    if (!write_binary_image(output_path_nick, width, height, 1, output_nick.data())) {
        spdlog::error("Failed to write Nick output image: {}", output_path_nick);
    } else {
        spdlog::info("Nick binarized image saved to: {}", output_path_nick);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("Advanced binarization process completed in {} seconds.", duration.count());

    stbi_image_free(image);
}
