#include "image_utils.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>


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

// Hilfsfunktion, um sicherzustellen, dass kein Slash vorangestellt wird,
// wenn parent_path() leer ist. So landen wir nicht im Root-Verzeichnis "/...".
std::string make_output_path(const std::string &input_path) {
    namespace fs = std::filesystem;
    fs::path p(input_path);
    std::string stem = p.stem().string();
    std::string ext = p.extension().string();
    std::string parent = p.parent_path().string();

    if (parent.empty()) {
        return stem + "_bin" + ext;
    } else {
        return parent + "/" + stem + "_bin" + ext;
    }
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

    if (!write_binary_image(output_path, width, height, channels, out.data())) {
        std::cerr << "Error: Failed to write image! " << output_path << std::endl;
    } else {
        std::cout << "Binarized image saved to: " << output_path << std::endl;
    }
    stbi_image_free(image);
}