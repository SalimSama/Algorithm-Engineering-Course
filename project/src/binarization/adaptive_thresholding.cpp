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

/**
 * Computes the local mean and standard deviation of a grayscale image within a given window.
 *
 * @param gray Pointer to the grayscale image data.
 * @param width Image width.
 * @param height Image height.
 * @param x X-coordinate of the pixel being processed.
 * @param y Y-coordinate of the pixel being processed.
 * @param half_win Half of the window size for computing local statistics.
 * @param mean Reference variable to store the computed mean.
 * @param stddev Reference variable to store the computed standard deviation.
 */
void local_mean_std(const unsigned char* gray,
                    int width, int height,
                    int x, int y,
                    int half_win,
                    float &mean, float &stddev)
{
    int count = 0;
    float sum = 0.0f, sum_sq = 0.0f;

    // Iterate over the window centered at (x, y)
    for (int dy = -half_win; dy <= half_win; dy++) {
        for (int dx = -half_win; dx <= half_win; dx++) {
            int yy = y + dy;
            int xx = x + dx;

            // Ensure the indices are within image boundaries
            if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
                unsigned char val = gray[yy * width + xx];
                sum += val;
                sum_sq += val * val;
                count++;
            }
        }
    }

    // Compute the mean
    mean = sum / count;

    // Compute variance and standard deviation
    float var = (sum_sq / count) - (mean * mean);
    stddev = (var > 0) ? std::sqrt(var) : 0.0f;
}

/**
 * Applies adaptive thresholding to a grayscale image using a user-defined threshold function.
 *
 * @param gray Input grayscale image data.
 * @param out Output binarized image data.
 * @param width Image width.
 * @param height Image height.
 * @param window_size Size of the local window for threshold calculation.
 * @param threshold_func Lambda function to calculate threshold based on mean and standard deviation.
 */
void adaptive_binarize(const unsigned char* gray,
                       unsigned char* out,
                       int width, int height,
                       int window_size,
                       const std::function<float(float mean, float stddev)> &threshold_func) {
    int half_win = window_size / 2;

    spdlog::info("Starting adaptive binarization with window size {}");

    auto start = std::chrono::high_resolution_clock::now();

    // Parallelized loop to process each pixel in the image
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float mean = 0.0f, stddev = 0.0f;

            // Compute local mean and standard deviation
            local_mean_std(gray, width, height, x, y, half_win, mean, stddev);

            // Compute the adaptive threshold
            float threshold = threshold_func(mean, stddev);

            // Apply thresholding
            out[y * width + x] = (gray[y * width + x] > threshold) ? 255 : 0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    spdlog::info("Adaptive binarization completed in {} seconds.", duration.count());
}

/**
 * Implements Sauvola's binarization method.
 *
 * @param gray Input grayscale image data.
 * @param out Output binarized image data.
 * @param width Image width.
 * @param height Image height.
 * @param window_size Size of the local window for threshold calculation.
 * @param k Parameter that adjusts the thresholding sensitivity.
 * @param R Dynamic range of standard deviation (typically 128 for 8-bit images).
 */
void sauvola_binarize(const unsigned char* gray,
                      unsigned char* out,
                      int width, int height,
                      int window_size,
                      float k,
                      float R) {
    spdlog::info("Starting Sauvola binarization with window size {}, k={}, R={}.", window_size, k, R);

    // Define the threshold function for Sauvola
    auto threshold_func = [k, R](float mean, float stddev) {
        return mean * (1.0f + k * ((stddev / R) - 1.0f));
    };

    adaptive_binarize(gray, out, width, height, window_size, threshold_func);

    spdlog::info("Sauvola binarization completed.");
}

/**
 * Implements NICK binarization method.
 *
 * @param gray Input grayscale image data.
 * @param out Output binarized image data.
 * @param width Image width.
 * @param height Image height.
 * @param window_size Size of the local window for threshold calculation.
 * @param k Parameter that adjusts the thresholding sensitivity.
 */
void nick_binarize(const unsigned char* gray,
                   unsigned char* out,
                   int width, int height,
                   int window_size,
                   float k) {
    spdlog::info("Starting Nick binarization with window size {}, k={}.", window_size, k);

    // Define the threshold function for Nick's method
    auto threshold_func = [k](float mean, float stddev) {
        return mean - k * stddev;
    };

    adaptive_binarize(gray, out, width, height, window_size, threshold_func);

    spdlog::info("Nick binarization completed.");
}

/**
 * Loads an image, converts it to grayscale, applies Sauvola and Nick binarization,
 * and saves the results.
 *
 * @param input_path Path to the input image file.
 * @param window_size Size of the local window for threshold calculation.
 * @param k Parameter for thresholding.
 * @param R Dynamic range parameter for Sauvola's method.
 */

void process_advanced_binarization(const std::string &input_path, int window_size, float k, float R) {
    spdlog::info("Processing advanced binarization for: {} with window size {}, k={}, R={}", input_path, window_size, k, R);

    int width, height, channels;

    // Load the image
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
        return;
    }

    // Convert to grayscale
    std::vector<unsigned char> gray(width * height);
    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        gray[i] = static_cast<unsigned char>(
                0.2126f * image[i * channels + 0] +
                0.7152f * image[i * channels + 1] +
                0.0722f * image[i * channels + 2]);
    }

    // Generate output file paths
    std::string output_path_sauvola = make_output_path(input_path, "sauvola");
    std::string output_path_nick = make_output_path(input_path, "nick");

    auto start = std::chrono::high_resolution_clock::now();

    // Apply Sauvola binarization
    std::vector<unsigned char> output_sauvola(width * height);
    sauvola_binarize(gray.data(), output_sauvola.data(), width, height, window_size, k, R);

    if (!write_binary_image(output_path_sauvola, width, height, 1, output_sauvola.data())) {
        spdlog::error("Failed to write Sauvola output image: {}", output_path_sauvola);
    } else {
        spdlog::info("Sauvola binarized image saved to: {}", output_path_sauvola);
    }

    // Apply Nick binarization
    std::vector<unsigned char> output_nick(width * height);
    nick_binarize(gray.data(), output_nick.data(), width, height, window_size, k);

    if (!write_binary_image(output_path_nick, width, height, 1, output_nick.data())) {
        spdlog::error("Failed to write Nick output image: {}", output_path_nick);
    } else {
        spdlog::info("Nick binarized image saved to: {}", output_path_nick);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    spdlog::info("Advanced binarization process completed in {} seconds.", duration.count());

    stbi_image_free(image);
}
