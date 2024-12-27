#include <iostream>
#include "image_utils.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.(png|jpg|ppm...)> <threshold> [<output.(png|jpg|ppm...)>]"
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
    binarize_image_parallel(input_path, output_path, threshold);
    process_advanced_binarization(input_path);
    process_integral_binarization(input_path);
    return 0;
}
