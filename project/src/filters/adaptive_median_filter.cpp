#include <filters/adaptive_median_filter.h>
#include <utils/image_io.h>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <spdlog/spdlog.h>


// Structure to store the minimum and maximum window sizes for filtering
struct WindowParams {
    int min_size;
    int max_size;
};

// Function to estimate optimal window sizes based on image characteristics
WindowParams estimate_optimal_window_sizes(const std::vector<unsigned char>& gray, int width, int height) {
    // 1. Calculate image-wide median and MAD for adaptive thresholding
    std::vector<unsigned char> values(gray);
    const size_t mid_idx = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid_idx, values.end());
    const unsigned char median_value = values[mid_idx];

    // Calculate absolute deviations from median
    std::vector<float> abs_deviations;
    abs_deviations.reserve(values.size());
    for (auto val : values) {
        abs_deviations.push_back(std::abs(static_cast<float>(val) - median_value));
    }

    // Calculate MAD
    std::nth_element(abs_deviations.begin(),
                    abs_deviations.begin() + abs_deviations.size()/2,
                    abs_deviations.end());
    const float mad = abs_deviations[abs_deviations.size()/2];

    // 2. Estimate noise using MAD-based homogeneous region detection
    std::vector<float> block_variances;
    const int block_size = 8;
    const float var_threshold = 1.4826f * mad * 2.0f; // Scale factor converts MAD to standard deviation equivalent

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

            // Use MAD-based threshold for homogeneous regions
            if (var < var_threshold) {
                block_variances.push_back(var);
            }
        }
    }

    // Robust noise estimation using median of variances
    float noise_level = 0.0f;
    if (!block_variances.empty()) {
        std::nth_element(block_variances.begin(),
                        block_variances.begin() + block_variances.size()/2,
                        block_variances.end());
        noise_level = block_variances[block_variances.size()/2];
    } else {
        noise_level = 1.4826f * mad; // Fallback using MAD directly
    }

    // 3. Compute edge density using gradient magnitude
    float edge_density = 0.0f;
    const float edge_threshold = 1.4826f * mad * 1.5f; // Adaptive threshold based on MAD

    #pragma omp parallel for reduction(+:edge_density)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            // Compute Sobel gradient magnitude
            float gx = -gray[(y-1)*width + (x-1)] - 2*gray[y*width + (x-1)] - gray[(y+1)*width + (x-1)] +
                       gray[(y-1)*width + (x+1)] + 2*gray[y*width + (x+1)] + gray[(y+1)*width + (x+1)];
            float gy = -gray[(y-1)*width + (x-1)] - 2*gray[(y-1)*width + x] - gray[(y-1)*width + (x+1)] +
                       gray[(y+1)*width + (x-1)] + 2*gray[(y+1)*width + x] + gray[(y+1)*width + (x+1)];

            float magnitude = std::sqrt(gx*gx + gy*gy);
            if (magnitude > edge_threshold) {
                edge_density += 1.0f;
            }
        }
    }
    edge_density /= (width * height);

    // 4. Determine window sizes based on noise level and edge density
    int min_size = 3;
    int max_size;

    // Adaptive determination based on noise level relative to MAD
    float relative_noise = noise_level / (1.4826f * mad);
    if (relative_noise < 0.5f) {
        max_size = 7;  // Low noise
    } else if (relative_noise < 1.5f) {
        max_size = 11; // Medium noise
    } else {
        max_size = 15; // High noise
    }

    // Adjust max window size based on edge density
    if (edge_density > 0.08f) {
        // Many details - reduce window to preserve them
        max_size = std::max(7, max_size - 4);
    }

    // Ensure odd-sized windows
    if (min_size % 2 == 0) min_size++;
    if (max_size % 2 == 0) max_size++;

    spdlog::info("Image statistics - Median: {}, MAD: {:.2f}", median_value, mad);
    spdlog::info("Estimated relative noise: {:.2f}, Edge density: {:.4f}", relative_noise, edge_density);
    spdlog::info("Selected window sizes - min: {}, max: {}", min_size, max_size);

    return {min_size, max_size};
}

void increase_window_size(const std::vector<unsigned char> &input,
                          std::vector<unsigned char> &temp_window,
                          int old_window_size, int new_window_size,
                          const int width, const int height, const int xpos, const int ypos) {

    const int old_half_win = (old_window_size - 1) / 2;
    const int new_half_win = (new_window_size - 1) / 2;

    // Calculate maximum number of new pixels to add
    int max_new_elements = 4 * new_window_size - 4;

    // Ensure capacity is sufficient
    if (temp_window.capacity() < temp_window.size() + max_new_elements) {
        temp_window.reserve(temp_window.size() + max_new_elements);
    }
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

// Function to extract a window from the image at a given location
void get_window(const std::vector<unsigned char> &input,
                int width, int height,
                int x, int y,
                int window_size,
                std::vector<unsigned char> &output_window) {

    output_window.clear();
    const int half_win = window_size / 2;

    // Estimate required capacity and reserve once
    int estimated_size = (window_size * window_size * 3) / 4; // ~75% of window might be valid
    output_window.reserve(estimated_size);

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

// Adaptive median filtering process
void adaptive_median_filter_process(const std::vector<unsigned char> &input, std::vector<unsigned char> *output,
                                   const int width, const int height, const int channels,
                                   int min_win_size, int max_window_size) {

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        std::vector<unsigned char> temp_window;
        temp_window.reserve(max_window_size * max_window_size);

        for (int x = 0; x < width; ++x) {
            const int pos = y * width + x;
            const unsigned char pxl = input[pos];

            int current_win_size = min_win_size;
            temp_window.clear();

            // Extract initial window
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

    spdlog::info("adaptive_median_filter Starting processing on: {}", input_path);

    int width, height, channels;

    // Load the input image from the file path
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);

    if (!image) {
        spdlog::error("[adaptive_median_filter] Failed to load image: {}", input_path);
        return;
    }

    // If no output path is specified, generate one based on the input path
    if (output_path.empty()) {
        output_path = make_output_path(input_path)+ "_amf.png";
    }

    // Convert the image to grayscale using standard luminance weights
    std::vector<unsigned char> gray(width * height);

    // Parallel loop to speed up grayscale conversion
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

    auto start = std::chrono::high_resolution_clock::now();

    // Apply the adaptive median filter to the grayscale image
    adaptive_median_filter_process(gray, &output, width, height, 1, params.min_size, params.max_size);

    if (!write_binary_image(output_path, width, height, 1, output.data())) {
        spdlog::error("[adaptive_median_filter] Failed to write filtered image: {}", output_path);
    } else {
        spdlog::info("[adaptive_median_filter] Filtered image saved to: {}", output_path);
    }

    stbi_image_free(image);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    spdlog::info("adaptive_median_filter Total runtime: {} seconds", duration.count());
}