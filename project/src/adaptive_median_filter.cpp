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


void increase_window_size(const std::vector<unsigned char> &image, std::vector<unsigned char> *temp_window, int old_winsize,
                          const int width, const int height, const int xpos, const int ypos) {

    // go one row up, down, left and right and push everything onto temp_window

    int half_win = old_winsize/2;

    // row over the window needs to be inserted, if it is still inside the image
    for (int x = xpos - half_win -1; x <= xpos + half_win + 1; x++) {
        if (ypos - half_win - 1 >= 0 && x >= 0 && x <= width - 1) {         // 1 2 pos 4 5 wenn aktuelles x im Bild ist kommt es rein
            temp_window->push_back(image[(ypos - half_win - 1) * width + x]);
        }

        // the same goes for the row below the window
        if (ypos + half_win + 1 <= height - 1 && x >= 0 && x <= width - 1) {
            temp_window->push_back(image[(ypos + half_win + 1) * width + x]);
        }
    }

    for (int y = ypos - half_win; y <= ypos + half_win; y++) {
        if (xpos - half_win - 1 >= 0 && y >= 0 && y <= height -1) {
            temp_window->push_back(image[y * width + xpos - half_win -1]);
        }
        if (xpos + half_win + 1 <= width - 1 && y >= 0 && y <= height - 1) {
            temp_window->push_back(image[y * width + xpos + half_win + 1]);
        }
    }
}



std::vector<unsigned char> get_window(std::vector<unsigned char> *image,
                                      int width, int height,
                                      int x, int y,
                                      int window_size) {

    std::vector<unsigned char> output_window;
    const int half_win = window_size / 2;
    unsigned int array_pos = 0;
    for (int dy = -half_win; dy <= half_win; dy++) {
        for (int dx = -half_win; dx <= half_win; dx++) {

            int yy = y + dy;
            int xx = x + dx;
            if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
                output_window.push_back((*image)[yy * width + xx]);
            }
            array_pos += 1;
        }
    }
    return output_window;
}



void median_filter(std::vector<unsigned char> *image, const int width, const int height, const int channels,
                            int min_win_size, int max_window_size) {
    // first lets try a simple median filter

    std::vector<unsigned char> temp_window;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            temp_window.clear();
            temp_window = get_window(image, width, height, x, y, min_win_size);
            std::sort(temp_window.begin(), temp_window.end(), comp);
            (*image)[y * width + x] = temp_window[temp_window.size() / 2];
        }
    }
}

void adaptive_median_filter_process(std::vector<unsigned char> *image, const int width, const int height, const int channels,
                            int min_win_size, int max_window_size) {

    std::vector<unsigned char> temp_window;
    int current_win_size;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            temp_window.clear();
            current_win_size = 1;
            temp_window = get_window(image, width, height, x, y, min_win_size);

            // while (current_win_size < min_win_size) { // increase the window size until we reach the minimum size requirements
            //     increase_window_size(*image, &temp_window, current_win_size, width, height, x, y);
            //     current_win_size++;
            // }

            int pos = y * width + x;
            unsigned char pxl = (*image)[pos];

            std::sort(temp_window.begin(), temp_window.end(), comp);
            unsigned char local_median = temp_window[temp_window.size() / 2];
            unsigned char local_max = temp_window[temp_window.size()-1];
            unsigned char local_min = temp_window[0];
            bool flag = false;

            while (!flag && current_win_size < max_window_size) {   // while the window size is smaller than max and the pixel hasn't been calculated
                if (local_median > local_min && local_median < local_max) {
                    if (pxl > local_min && pxl < local_max) {
                        flag = true;
                    }
                    else {
                        (*image)[pos] = local_median;
                        flag = true;
                    }

                } else {
                    current_win_size++;
                    temp_window = get_window(image, width, height, x, y, current_win_size);
                    //increase_window_size(*image, &temp_window, current_win_size, width, height, x, y);
                    std::sort(temp_window.begin(), temp_window.end(), comp);
                    local_median = temp_window[temp_window.size() / 2];
                    local_max = temp_window[temp_window.size()-1];
                    local_min = temp_window[0];
                }
            }
            // if (!flag) {
            //     (*image)[pos] = local_median;
            // }
        }
    }
}



void adaptive_median_filter(const std::string &input_path, std::string output_path) {
    spdlog::info("Starting adaptive median filter ");
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);

    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
        return;
    }

    if (output_path.empty()) {
        output_path = make_output_path(input_path);
    }

    std::vector<unsigned char> gray(width * height);

#pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        gray[i] = static_cast<unsigned char>(
                0.2126f * image[i * channels + 0] +
                0.7152f * image[i * channels + 1] +
                0.0722f * image[i * channels + 2]);
    }

    std::vector<unsigned char> gray_copy = gray;

    adaptive_median_filter_process(&gray, width, height, 1, 3, 10);
    // median_filter(&gray, width, height, 1, 5, 10);

    if (!write_binary_image(output_path, width, height, 1, gray.data())) {
        spdlog::error("Failed to write filtered image: {}", output_path);
    } else {
        spdlog::info("Filtered image saved to: {}", output_path);
    }

    stbi_image_free(image);

}