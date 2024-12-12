#include <iostream>  // Für Ein- und Ausgabe
#include <vector>    // Für dynamische Arrays (std::vector)
#include <filesystem>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <fstream>

void write_ppm(const std::string &filename, int width, int height, int channels, const unsigned char *data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }

    // PPM Header schreiben
    file << "P6\n" << width << " " << height << "\n255\n";

    // Bilddaten schreiben
    if (channels >= 3) {
        for (int i = 0; i < width * height * channels; i += channels) {
            file.put(data[i]);     // Rot
            file.put(data[i + 1]); // Grün
            file.put(data[i + 2]); // Blau
        }
    } else {
        std::cerr << "Error: Unsupported channel count for PPM. Only RGB supported." << std::endl;
    }

    file.close();
    std::cout << "PPM image saved to " << filename << std::endl;
}


// Funktion: Bild binarisieren (Schwarz-Weiß-Umwandlung basierend auf einem Schwellenwert)
void binarize_image(const std::string &input_path, const std::string &output_path, int threshold) {
    int width, height, channels;  // Variablen für Bildbreite, -höhe und Farbkanäle

    // Lade das Bild (PPM-Format) in den Speicher
    // stbi_load gibt einen Zeiger auf die Bilddaten zurück
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);

    // Überprüfe, ob das Bild erfolgreich geladen wurde
    if (!image) {
        std::cerr << "Error: Failed to load image!" << std::endl;
        return;
    }



    // Erstelle einen Vektor für die Ausgabe (binarisiertes Bild)
    std::vector<unsigned char> output_image(width * height * channels);

    // Iteriere über jeden Pixel im Bild
    for (int i = 0; i < width * height * channels; i += channels) {
        // Berechne den Grauwert basierend auf den RGB-Werten (Luminanzformel)
        unsigned char intensity = 0.2126 * image[i] + 0.7152 * image[i + 1] + 0.0722 * image[i + 2];

        // Schwellenwertprüfung: Wenn der Grauwert größer als der Threshold ist, setze auf Weiß (255), sonst auf Schwarz (0)
        unsigned char binary = (intensity > threshold) ? 255 : 0;

        // Setze alle Farbkanäle des Pixels auf den binären Wert (Schwarz/Weiß)
        for (int c = 0; c < channels; ++c) {
            output_image[i + c] = binary;
        }
    }

    // Schreibe das binarisierte Bild in eine neue Datei
    if (output_image.empty()) {
        std::cerr << "Error: Output image is empty!" << std::endl;
        return;
    }

// Verwenden Sie write_ppm anstelle von stbi_write_ppm
    write_ppm(output_path, width, height, channels, output_image.data());


    // Gib den Speicher für das ursprüngliche Bild frei
    stbi_image_free(image);
}



void reduce_noise(const std::string &input_path, const std::string &output_path, int kernel_size) {
    int width, height, channels;

    // Lade das Bild (PPM-Format) in den Speicher
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        std::cerr << "Error: Failed to load image!" << std::endl;
        return;
    }

    // Erstelle einen Vektor für die Ausgabe (geglättetes Bild)
    std::vector<unsigned char> output_image(width * height * channels);

    // Berechne den Radius basierend auf der Kernelgröße
    int radius = kernel_size / 2;

    // Iteriere über alle Pixel im Bild
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Für jeden Kanal (z. B. R, G, B)
            for (int c = 0; c < channels; ++c) {
                int sum = 0;           // Summe der Nachbarwerte
                int count = 0;         // Anzahl der berücksichtigten Nachbarn

                // Iteriere über den Kernel-Bereich
                for (int ky = -radius; ky <= radius; ++ky) {
                    for (int kx = -radius; kx <= radius; ++kx) {
                        int nx = x + kx; // Nachbarpixel x
                        int ny = y + ky; // Nachbarpixel y

                        // Stelle sicher, dass die Nachbarpixel im gültigen Bereich liegen
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            sum += image[(ny * width + nx) * channels + c];
                            count++;
                        }
                    }
                }
                // Berechne den Durchschnittswert und speichere ihn im Ausgabebild
                output_image[(y * width + x) * channels + c] = sum / count;
            }
        }
    }

    // Schreibe das geglättete Bild in eine neue Datei
    /*if (!stbi_write_ppm(output_path.c_str(), width, height, channels, output_image.data())) {
        std::cerr << "Error: Failed to write image!" << std::endl;
    }*/

    // Gib den Speicher für das ursprüngliche Bild frei
    stbi_image_free(image);
}


int main(int argc, char *argv[]) {
    // Überprüfe, ob der Benutzer die richtigen Argumente übergeben hat
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.ppm> <output.ppm> <threshold>" << std::endl;
        return 1;
    }

    // Lese die Eingabewerte aus den Argumenten
    std::string input_path = argv[1];   // Pfad zur Eingabedatei (PPM)
    std::string output_path = argv[2];  // Pfad zur Ausgabedatei (PPM)
    int threshold = std::stoi(argv[3]); // Schwellenwert für die Binarisierung (z. B. 128)

    // Rufe die Binarisierungsfunktion auf
    binarize_image(input_path, output_path, threshold);

    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    // Informiere den Benutzer über den Abschluss
    std::cout << "Image processed and saved to " << output_path << std::endl;
    return 0;
}
