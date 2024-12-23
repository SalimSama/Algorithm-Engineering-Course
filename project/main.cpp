#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Hilfsfunktion, um sicherzustellen,
// dass kein Slash vorangestellt wird, wenn parent_path() leer ist.
// So landen wir nicht im Root-Verzeichnis "/...".
std::string make_output_path(const std::string &input_path) {
    namespace fs = std::filesystem;
    fs::path p(input_path);
    std::string stem = p.stem().string();
    std::string ext = p.extension().string();
    std::string parent = p.parent_path().string();

    if (parent.empty()) {
        // Kein Verzeichnis, Datei liegt also im aktuellen Pfad
        return stem + "_bin" + ext;
    } else {
        // Dateiname + "_bin" im gleichen Ordner
        return parent + "/" + stem + "_bin" + ext;
    }
}

// Schreibt das binarisierte Bild als PNG/JPG.
// (Für .ppm gibt es kein stbi_write_ppm, deshalb Fallback auf PNG/JPG.)
bool write_binary_image(const std::string &filename, int width, int height, int channels, const unsigned char *data) {
    std::string extension = std::filesystem::path(filename).extension().string();
    for (auto &c : extension) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    if (extension == ".png") {
        return stbi_write_png(filename.c_str(), width, height, channels, data, width * channels);
    } else if (extension == ".jpg" || extension == ".jpeg") {
        // Qualität 90 (Range: 1–100)
        return stbi_write_jpg(filename.c_str(), width, height, channels, data, 90);
    }
    // Fallback (z.B. .ppm etc.): Schreibe als PNG
    return stbi_write_png(filename.c_str(), width, height, channels, data, width * channels);
}

void binarize_image(const std::string &input_path, std::string output_path, int threshold) {
    int width, height, channels;

    // Bild laden
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        std::cerr << "Error: Failed to load image!" << std::endl;
        return;
    }

    // Wenn kein Output-Pfad angegeben, automatisch erzeugen
    if (output_path.empty()) {
        output_path = make_output_path(input_path);
    }

    // Binarisiertes Bild vorbereiten
    std::vector<unsigned char> out(width * height * channels);
    for (int i = 0; i < width * height; i++) {
        int idx = i * channels;
        // Luminanzformel
        unsigned char lum = static_cast<unsigned char>(
                0.2126f * image[idx] + 0.7152f * image[idx + 1] + 0.0722f * image[idx + 2]
        );
        unsigned char binary = (lum > threshold) ? 255 : 0;

        // Alle Kanäle (auch Alpha) gleichsetzen
        for (int c = 0; c < channels; c++) {
            out[idx + c] = binary;
        }
    }

    // Schreiben
    if (!write_binary_image(output_path, width, height, channels, out.data())) {
        std::cerr << "Error: Failed to write image! " << output_path << std::endl;
    } else {
        std::cout << "Binarized image saved to: " << output_path << std::endl;
    }
    stbi_image_free(image);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr
                << "Usage: " << argv[0]
                << " <input.(png|jpg|ppm...)> <threshold> [<output.(png|jpg|...)>]"
                << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    int threshold = std::stoi(argv[2]);

    std::string output_path;
    if (argc > 3) {
        output_path = argv[3];
    }

    binarize_image(input_path, output_path, threshold);
    return 0;
}
