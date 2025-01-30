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
#include <spdlog/spdlog.h>

// Hilfsfunktion zum Schreiben eines ASCII-PPM
bool write_ppm_ascii(const std::string &filename, int width, int height, int channels, const unsigned char *data) {
    spdlog::info("Writing ASCII PPM file: {}", filename);
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        spdlog::error("Failed to open file: {}", filename);
        return false;
    }

    ofs << "P3\n" << width << " " << height << "\n255\n";

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            unsigned char r = data[idx];
            unsigned char g = data[idx + 1];
            unsigned char b = data[idx + 2];
            ofs << (int)r << " " << (int)g << " " << (int)b << "\n";
        }
    }
    spdlog::info("Successfully wrote ASCII PPM file: {}", filename);
    return true;
}

std::string make_output_path(const std::string &input_path) {
    spdlog::info("Creating output path for input: {}", input_path);
    namespace fs = std::filesystem;
    fs::path p(input_path);
    std::string stem = p.stem().string();
    std::string ext = p.extension().string();

    fs::path results_dir = "Results";
    if (!fs::exists(results_dir)) {
        fs::create_directory(results_dir);
        spdlog::info("Created results directory: {}", results_dir.string());
    }

    std::string output_path = (results_dir / (stem + "_bin" + ext)).string();
    spdlog::info("Output path created: {}", output_path);
    return output_path;
}

bool write_binary_image(const std::string &filename, int width, int height, int channels, const unsigned char *data) {
    spdlog::info("Writing binary image to: {}", filename);
    std::string extension = std::filesystem::path(filename).extension().string();
    for (auto &c : extension) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    bool success = false;
    if (extension == ".png") {
        success = stbi_write_png(filename.c_str(), width, height, channels, data, width * channels);
    } else if (extension == ".jpg" || extension == ".jpeg") {
        success = stbi_write_jpg(filename.c_str(), width, height, channels, data, 90);
    } else if (extension == ".ppm") {
        success = write_ppm_ascii(filename, width, height, channels, data);
    } else {
        success = stbi_write_png(filename.c_str(), width, height, channels, data, width * channels);
    }

    if (success) {
        spdlog::info("Successfully wrote binary image: {}", filename);
    } else {
        spdlog::error("Failed to write binary image: {}", filename);
    }
    return success;
}

bool comp(unsigned char a, unsigned char b) {
    return a < b;
}

void increase_window_size(const std::vector<unsigned char> &input, std::vector<unsigned char> *temp_window,
                          int new_window_size, const int width, const int height, const int xpos, const int ypos) {

    const int half_win = (new_window_size - 1) / 2;

    // Top and bottom rows
    for (int dx = -half_win; dx <= half_win; ++dx) {
        const int xx = xpos + dx;
        const int yy_top = ypos - half_win;
        if (yy_top >= 0 && xx >= 0 && xx < width) {
            temp_window->push_back(input[yy_top * width + xx]);
        }
        const int yy_bottom = ypos + half_win;
        if (yy_bottom < height && xx >= 0 && xx < width) {
            temp_window->push_back(input[yy_bottom * width + xx]);
        }
    }

    // Left and right columns (excluding corners)
    for (int dy = -half_win + 1; dy <= half_win - 1; ++dy) {
        const int yy = ypos + dy;
        const int xx_left = xpos - half_win;
        if (xx_left >= 0 && yy >= 0 && yy < height) {
            temp_window->push_back(input[yy * width + xx_left]);
        }
        const int xx_right = xpos + half_win;
        if (xx_right < width && yy >= 0 && yy < height) {
            temp_window->push_back(input[yy * width + xx_right]);
        }
    }
}

std::vector<unsigned char> get_window(const std::vector<unsigned char> &input,
                                      int width, int height,
                                      int x, int y,
                                      int window_size) {
    std::vector<unsigned char> output_window;
    const int half_win = window_size / 2;
    output_window.reserve(window_size * window_size);

    for (int dy = -half_win; dy <= half_win; ++dy) {
        for (int dx = -half_win; dx <= half_win; ++dx) {
            const int yy = y + dy;
            const int xx = x + dx;
            if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
                output_window.push_back(input[yy * width + xx]);
            }
        }
    }
    return output_window;
}

void adaptive_median_filter_process(const std::vector<unsigned char> &input, std::vector<unsigned char> *output,
                                    const int width, const int height, const int channels,
                                    int min_win_size, int max_window_size) {

    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<unsigned char> temp_window;
            temp_window.reserve(max_window_size * max_window_size);

            int current_win_size = min_win_size;
            temp_window = get_window(input, width, height, x, y, current_win_size);
            std::sort(temp_window.begin(), temp_window.end(), comp);

            const int pos = y * width + x;
            const unsigned char pxl = input[pos];

            unsigned char local_median = temp_window[temp_window.size() / 2];
            unsigned char local_max = temp_window.back();
            unsigned char local_min = temp_window.front();
            bool flag = false;

            while (!flag && current_win_size < max_window_size) {
                if (local_median > local_min && local_median < local_max) {
                    if (pxl > local_min && pxl < local_max) {
                        (*output)[pos] = pxl;
                    } else {
                        (*output)[pos] = local_median;
                    }
                    flag = true;
                } else {
                    current_win_size += 2;
                    increase_window_size(input, &temp_window, current_win_size, width, height, x, y);
                    std::sort(temp_window.begin(), temp_window.end(), comp);
                    local_median = temp_window[temp_window.size() / 2];
                    local_max = temp_window.back();
                    local_min = temp_window.front();
                }
            }

            if (!flag) {
                (*output)[pos] = local_median;
            }
        }
    }
}

void adaptive_median_filter(const std::string &input_path, std::string output_path) {
    auto total_start = std::chrono::high_resolution_clock::now();
    spdlog::info("Starting adaptive median filter");

    auto load_start = std::chrono::high_resolution_clock::now();
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    auto load_end = std::chrono::high_resolution_clock::now();
    spdlog::info("Image loading took {} ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count());

    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
        return;
    }

    if (output_path.empty()) {
        output_path = make_output_path(input_path);
    }

    auto convert_start = std::chrono::high_resolution_clock::now();
    std::vector<unsigned char> gray(width * height);
    #pragma omp parallel for
    for (int i = 0; i < width * height; ++i) {
        gray[i] = static_cast<unsigned char>(
            0.2126f * image[i * channels + 0] +
            0.7152f * image[i * channels + 1] +
            0.0722f * image[i * channels + 2]);
    }
    auto convert_end = std::chrono::high_resolution_clock::now();
    spdlog::info("Grayscale conversion took {} ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(convert_end - convert_start).count());

    auto filter_start = std::chrono::high_resolution_clock::now();
    std::vector<unsigned char> output(width * height);
    adaptive_median_filter_process(gray, &output, width, height, 1, 3, 10);
    auto filter_end = std::chrono::high_resolution_clock::now();
    spdlog::info("Filtering took {} ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(filter_end - filter_start).count());

    auto write_start = std::chrono::high_resolution_clock::now();
    if (!write_binary_image(output_path, width, height, 1, output.data())) {
        spdlog::error("Failed to write filtered image: {}", output_path);
    } else {
        spdlog::info("Filtered image saved to: {}", output_path);
    }
    auto write_end = std::chrono::high_resolution_clock::now();
    spdlog::info("Writing image took {} ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(write_end - write_start).count());

    stbi_image_free(image);

    auto total_end = std::chrono::high_resolution_clock::now();
    spdlog::info("Total runtime: {} ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count());
}
