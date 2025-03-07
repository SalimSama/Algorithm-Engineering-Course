#include <binarization/thresholding.h>
#include <utils/image_io.h>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <spdlog/spdlog.h>
/**
 * @brief Performs sequential image binarization using a threshold.
 *
 * This function reads an image from the specified input path, applies
 * a simple thresholding operation to convert it into a binary image,
 * and saves the result to the output path.
 *
 * @param input_path Path to the input image file.
 * @param output_path Path where the output image will be saved. If empty, a default path is generated.
 * @param threshold The threshold value for binarization (0-255).
 */
void binarize_image(const std::string &input_path, std::string output_path, int threshold) {
    spdlog::info("Starting sequential binarization with threshold {} for: {}", threshold, input_path);

    // Variables to store image properties
    int width, height, channels;

    // Load the image from the input file
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
        return;
    }

    // If no output path is specified, generate one automatically
    if (output_path.empty()) {
        output_path = make_output_path(input_path);
    }

    // Create an output buffer for the binarized image
    std::vector<unsigned char> out(width * height * channels);

    // Start measuring the execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Iterate through each pixel and apply thresholding
    for (int i = 0; i < width * height; i++) {
        int idx = i * channels;

        // Compute grayscale luminance using standard weights
        unsigned char lum = static_cast<unsigned char>(
            0.2126f * image[idx] + 0.7152f * image[idx + 1] + 0.0722f * image[idx + 2]
        );

        // Convert to binary: 255 for above threshold, 0 for below
        unsigned char binary = (lum > threshold) ? 255 : 0;

        // Assign the binary value to all channels (grayscale effect)
        for (int c = 0; c < channels; c++) {
            out[idx + c] = binary;
        }
    }

    // Stop measuring execution time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("Sequential binarization completed in {} seconds.", duration.count());

    // Write the output image
    if (!write_binary_image(output_path, width, height, channels, out.data())) {
        spdlog::error("Failed to write binarized image: {}", output_path);
    } else {
        spdlog::info("Binarized image saved to: {}", output_path);
    }

    // Free memory allocated for the input image
    stbi_image_free(image);
}

/**
 * @brief Performs parallel image binarization using OpenMP.
 *
 * This function reads an image from the specified input path, applies
 * a thresholding operation in parallel using OpenMP, and saves the
 * binarized image to the output path.
 *
 * @param input_path Path to the input image file.
 * @param output_path Path where the output image will be saved. If empty, a default path is generated.
 * @param threshold The threshold value for binarization (0-255).
 */
void binarize_image_parallel(const std::string &input_path, std::string output_path, int threshold) {
    spdlog::info("Starting parallel binarization with threshold {} for: {}", threshold, input_path);

    // Variables to store image properties
    int width, height, channels;

    // Load the image from the input file
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
        return;
    }

    // If no output path is specified, generate one automatically
    if (output_path.empty()) {
        output_path = make_output_path(input_path);
    }

    // Create an output buffer for the binarized image
    std::vector<unsigned char> out(width * height * channels);

    // Start measuring the execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Parallel loop using OpenMP to speed up computation
#pragma omp parallel for simd
    for (int i = 0; i < width * height; i++) {
        int idx = i * channels;

        // Compute grayscale luminance using standard weights
        unsigned char lum = static_cast<unsigned char>(
            0.2126f * image[idx] + 0.7152f * image[idx + 1] + 0.0722f * image[idx + 2]
        );

        // Convert to binary: 255 for above threshold, 0 for below
        unsigned char binary = (lum > threshold) ? 255 : 0;

        // Assign the binary value to all channels (grayscale effect)
        for (int c = 0; c < channels; c++) {
            out[idx + c] = binary;
        }
    }

    // Stop measuring execution time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("Parallel binarization completed in {} seconds.", duration.count());

    // Write the output image
    if (!write_binary_image(output_path, width, height, channels, out.data())) {
        spdlog::error("Failed to write parallel binarized image: {}", output_path);
    } else {
        spdlog::info("Parallel binarized image saved to: {}", output_path);
    }

    // Free memory allocated for the input image
    stbi_image_free(image);
}