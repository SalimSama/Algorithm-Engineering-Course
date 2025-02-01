#include <binarization/thresholding.h>
#include <utils/image_io.h>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <spdlog/spdlog.h>


void binarize_image(const std::string &input_path, std::string output_path, int threshold) {
    spdlog::info("Starting sequential binarization with threshold {} for: {}", threshold, input_path);
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
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
    spdlog::info("Sequential binarization completed in {} seconds.", duration.count());

    if (!write_binary_image(output_path, width, height, channels, out.data())) {
        spdlog::error("Failed to write binarized image: {}", output_path);
    } else {
        spdlog::info("Binarized image saved to: {}", output_path);
    }
    stbi_image_free(image);
}

void binarize_image_parallel(const std::string &input_path, std::string output_path, int threshold) {
    spdlog::info("Starting parallel binarization with threshold {} for: {}", threshold, input_path);
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
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
    spdlog::info("Parallel binarization completed in {} seconds.", duration.count());

    if (!write_binary_image(output_path, width, height, channels, out.data())) {
        spdlog::error("Failed to write parallel binarized image: {}", output_path);
    } else {
        spdlog::info("Parallel binarized image saved to: {}", output_path);
    }
    stbi_image_free(image);
}
