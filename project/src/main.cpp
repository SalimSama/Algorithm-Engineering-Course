#include <iostream>
#include <string>
#include "binarization/thresholding.h"
#include "binarization/adaptive_thresholding.h"
#include "binarization/integral_binarization.h"
#include "filters/adaptive_median_filter.h"
#include "../external/spdlog/include/spdlog/spdlog.h"
#include "../external/spdlog/include/spdlog/sinks/basic_file_sink.h"

void displayMenu() {
    std::cout << "\nChoose an operation:\n";
    std::cout << "1. Sequential Binarization\n";
    std::cout << "2. Parallel Binarization\n";
    std::cout << "3. Advanced Binarization (Sauvola & Nick)\n";
    std::cout << "4. Integral Image Binarization\n";
    std::cout << "5. All Methods\n";
    std::cout << "6. Apply adaptive median filter\n";
    std::cout << "Enter your choice (1-5): ";
}

int main(int argc, char *argv[]) {
    std::cout << "Program started, writing to output.log!" << std::endl;

    try {
        auto logger = spdlog::basic_logger_mt("file_logger", "logs/output.log");
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::info);
        spdlog::info("***** Program started *****\n\n");

        if (argc < 3) {
            spdlog::error("Usage: {} <input.(png|jpg|ppm...)> <threshold> [<output.(png|jpg|ppm...)>]", argv[0]);
            return 1;
        }

        std::string input_path = argv[1];
        int threshold = std::stoi(argv[2]);
        std::string output_path;
        if (argc > 3) {
            output_path = argv[3];
        }

        int choice = 0;
        while (choice < 1 || choice > 6) {
            displayMenu();
            std::cin >> choice;
            if (std::cin.fail() || choice < 1 || choice > 5) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid input. Please enter a number between 1 and 5." << std::endl;
                choice = 0;
            }
        }

        switch (choice) {
            case 1:
                binarize_image(input_path, output_path, threshold);
                break;
            case 2:
                binarize_image_parallel(input_path, output_path, threshold);
                break;
            case 3:
                process_advanced_binarization(input_path);
                break;
            case 4:
                process_integral_binarization(input_path);
                break;
            case 5:
                binarize_image(input_path, output_path, threshold);
                binarize_image_parallel(input_path, output_path, threshold);
                process_advanced_binarization(input_path);
                process_integral_binarization(input_path);
                adaptive_median_filter(input_path, output_path);
                break;
            case 6:
                adaptive_median_filter(input_path, output_path);
                break;
        }

        spdlog::info("***** Program finished successfully *****\n\n");
    } catch (const std::exception &e) {
        spdlog::critical("Unhandled exception: {}", e.what());
        return 1;
    }

    return 0;
}
