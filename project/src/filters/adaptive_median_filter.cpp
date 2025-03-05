#include <filters/adaptive_median_filter.h>
#include <utils/image_io.h>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <spdlog/spdlog.h>


void majority_vote(const std::string &input_path, std::string output_path) {
    // compute n images (one for each method that we choose to apply)
    spdlog::info("Starting majority vote for: {}", input_path);
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

    std::vector<unsigned char> img_binarize_image_parallel(width * height * channels);

    // add all intensity values to a new image with uncapped intensities
    // divide intensity values by n (this should scale the image back to the prior scale)
    // binarize image again (using half the maximum intensity as the threshold)
}

struct WindowParams {
    int min_size;
    int max_size;
};

WindowParams estimate_optimal_window_sizes(const std::vector<unsigned char>& gray, int width, int height) {
    // 1. Estimate noise level using homogeneous region variance
    std::vector<float> block_variances;
    const int block_size = 8;

    for (int y = 0; y < height - block_size; y += block_size) {
        for (int x = 0; x < width - block_size; x += block_size) {
            float mean = 0.0f, var = 0.0f;

            // Calculate mean
            for (int by = 0; by < block_size; by++) {
                for (int bx = 0; bx < block_size; bx++) {
                    mean += gray[(y + by) * width + (x + bx)];
                }
            }
            mean /= (block_size * block_size);

            // Calculate variance
            for (int by = 0; by < block_size; by++) {
                for (int bx = 0; bx < block_size; bx++) {
                    float diff = gray[(y + by) * width + (x + bx)] - mean;
                    var += diff * diff;
                }
            }
            var /= (block_size * block_size);

            // Only consider homogeneous regions (low gradient blocks)
            if (var < 200.0f) { // Threshold for homogeneous regions
                block_variances.push_back(var);
            }
        }
    }

    // Sort variances and take median as noise estimate
    if (block_variances.empty()) {
        return {3, 9}; // Default if estimation fails
    }

    std::nth_element(block_variances.begin(),
                    block_variances.begin() + block_variances.size()/2,
                    block_variances.end());
    float noise_level = block_variances[block_variances.size()/2];

    // 2. Calculate edge density (approximation of image complexity)
    std::vector<unsigned char> edges(width * height, 0);
    float edge_density = 0.0f;

    // Simple Sobel edge detection
    #pragma omp parallel for reduction(+:edge_density)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float gx = -gray[(y-1)*width + (x-1)] - 2*gray[y*width + (x-1)] - gray[(y+1)*width + (x-1)] +
                       gray[(y-1)*width + (x+1)] + 2*gray[y*width + (x+1)] + gray[(y+1)*width + (x+1)];
            float gy = -gray[(y-1)*width + (x-1)] - 2*gray[(y-1)*width + x] - gray[(y-1)*width + (x+1)] +
                       gray[(y+1)*width + (x-1)] + 2*gray[(y+1)*width + x] + gray[(y+1)*width + (x+1)];

            float magnitude = std::sqrt(gx*gx + gy*gy);
            if (magnitude > 30.0f) { // Edge threshold
                edge_density += 1.0f;
            }
        }
    }
    edge_density /= (width * height);

    // 3. Determine window sizes based on noise level and edge density
    int min_size = 3; // Base min size
    int max_size = 7; // Base max size

    // Adjust based on noise level
    if (noise_level < 10.0f) {
        max_size = 7;
    } else if (noise_level < 30.0f) {
        max_size = 11;
    } else {
        max_size = 15;
    }

    // Fine-tune based on edge density
    if (edge_density > 0.1f) { // Image has many details
        max_size = std::max(7, max_size - 4);
    }

    // Make sure min_size is odd
    if (min_size % 2 == 0) min_size++;

    // Make sure max_size is odd
    if (max_size % 2 == 0) max_size++;

    spdlog::info("Estimated noise level: {:.2f}, Edge density: {:.4f}", noise_level, edge_density);
    spdlog::info("Selected window sizes - min: {}, max: {}", min_size, max_size);

    return {min_size, max_size};
}

void increase_window_size(const std::vector<unsigned char> &input, std::vector<unsigned char> &temp_window,
                          int old_window_size, int new_window_size,
                          const int width, const int height, const int xpos, const int ypos) {

    const int old_half_win = (old_window_size - 1) / 2;
    const int new_half_win = (new_window_size - 1) / 2;

    // Top and bottom rows
    for (int dx = -new_half_win; dx <= new_half_win; ++dx) {
        const int xx = xpos + dx;

        // Top new row
        const int yy_top = ypos - new_half_win;
        if (yy_top >= 0 && xx >= 0 && xx < width) {
            temp_window.push_back(input[yy_top * width + xx]);
        }

        // Bottom new row
        const int yy_bottom = ypos + new_half_win;
        if (yy_bottom < height && xx >= 0 && xx < width) {
            temp_window.push_back(input[yy_bottom * width + xx]);
        }
    }

    // Left and right columns (excluding corners)
    for (int dy = -new_half_win + 1; dy < new_half_win; ++dy) {
        const int yy = ypos + dy;
        if (dy >= -old_half_win && dy <= old_half_win) {
            // Left new column
            const int xx_left = xpos - new_half_win;
            if (xx_left >= 0 && yy >= 0 && yy < height) {
                temp_window.push_back(input[yy * width + xx_left]);
            }

            // Right new column
            const int xx_right = xpos + new_half_win;
            if (xx_right < width && yy >= 0 && yy < height) {
                temp_window.push_back(input[yy * width + xx_right]);
            }
        }
    }
}

void get_window(const std::vector<unsigned char> &input,
                int width, int height,
                int x, int y,
                int window_size,
                std::vector<unsigned char> &output_window) {

    output_window.clear(); // Wiederverwendung des Puffers
    const int half_win = window_size / 2;

    for (int dy = -half_win; dy <= half_win; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= height) continue;

        const int row_offset = yy * width;
        for (int dx = -half_win; dx <= half_win; ++dx) {
            const int xx = x + dx;
            if (xx >= 0 && xx < width) {
                output_window.push_back(input[row_offset + xx]);
            }
        }
    }
}

void adaptive_median_filter_process(const std::vector<unsigned char> &input, std::vector<unsigned char> *output,
                                   const int width, const int height, const int channels,
                                   int min_win_size, int max_window_size) {

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        // Vor-Allokation des temp_window pro Thread
        std::vector<unsigned char> temp_window;
        temp_window.reserve(max_window_size * max_window_size);

        for (int x = 0; x < width; ++x) {
            const int pos = y * width + x;
            const unsigned char pxl = input[pos];

            int current_win_size = min_win_size;
            temp_window.clear();

            // Initialen Window auffüllen
            get_window(input, width, height, x, y, current_win_size, temp_window);

            bool flag = false;

            while (!flag && current_win_size < max_window_size) {
                // Partielle Sortierung mit nth_element - effizienter als vollständige Sortierung
                const size_t mid_idx = temp_window.size() / 2;
                std::nth_element(temp_window.begin(), temp_window.begin() + mid_idx, temp_window.end());

                const unsigned char local_median = temp_window[mid_idx];
                const unsigned char local_min = *std::min_element(temp_window.begin(), temp_window.end());
                const unsigned char local_max = *std::max_element(temp_window.begin(), temp_window.end());

                if (local_median > local_min && local_median < local_max) {
                    if (pxl > local_min && pxl < local_max) {
                        (*output)[pos] = pxl;
                    } else {
                        (*output)[pos] = local_median;
                    }
                    flag = true;
                } else {
                    const int old_win_size = current_win_size;
                    current_win_size += 2;

                    // Nur neue Elemente des größeren Fensters hinzufügen
                    increase_window_size(input, temp_window, old_win_size, current_win_size, width, height, x, y);
                }
            }

            if (!flag) {
                const size_t mid_idx = temp_window.size() / 2;
                std::nth_element(temp_window.begin(), temp_window.begin() + mid_idx, temp_window.end());
                (*output)[pos] = temp_window[mid_idx];
            }
        }
    }
}

void adaptive_median_filter(const std::string &input_path, std::string output_path) {
    auto start = std::chrono::high_resolution_clock::now();
    spdlog::info("adaptive_median_filter Starting processing on: {}", input_path);

    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);

    if (!image) {
        spdlog::error("[adaptive_median_filter] Failed to load image: {}", input_path);
        return;
    }

    if (output_path.empty()) {
        output_path = make_output_path(input_path);
    }

    // Grayscale conversion
    std::vector<unsigned char> gray(width * height);
#pragma omp parallel for simd
    for (int i = 0; i < width * height; ++i) {
        gray[i] = static_cast<unsigned char>(
            0.2126f * image[i * channels + 0] +
            0.7152f * image[i * channels + 1] +
            0.0722f * image[i * channels + 2]);
    }

    // Estimate optimal window sizes
    WindowParams params = estimate_optimal_window_sizes(gray, width, height);

    std::vector<unsigned char> output(width * height);
    adaptive_median_filter_process(gray, &output, width, height, 1, params.min_size, params.max_size);

    if (!write_binary_image(output_path, width, height, 1, output.data())) {
        spdlog::error("[adaptive_median_filter] Failed to write filtered image: {}", output_path);
    } else {
        spdlog::info("[adaptive_median_filter] Filtered image saved to: {}", output_path);
    }

    stbi_image_free(image);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("adaptive_median_filter Total runtime: {} seconds", duration.count());
}