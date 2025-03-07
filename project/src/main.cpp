/*
* Image Processing Tool
 *
 * This program processes images using various binarization and filtering techniques.
 * It supports multiple methods, including:
 *  - Sequential thresholding
 *  - Parallel thresholding
 *  - Advanced binarization (Sauvola, Nick)
 *  - Integral binarization
 *  - Adaptive median filtering
 *  - Running all available methods
 *
 * The program accepts command-line arguments to specify input/output paths,
 * processing methods, and parameters such as threshold values, window sizes,
 * and algorithm-specific parameters.
 *
 * Logging is handled using the spdlog library, which records execution details
 * in "logs/output.log".
 *
 * Author: Salim Alkhaddoor & Alexander Keller
 * Date: 07.03.2025
 */

#include <iostream>  // Standard library for input and output operations
#include <string>    // Standard string library for handling strings

// Including custom header files for different binarization and filtering methods
#include "binarization/thresholding.h"
#include "binarization/adaptive_thresholding.h"
#include "binarization/integral_binarization.h"
#include "filters/adaptive_median_filter.h"

// Including external logging library (spdlog) for logging messages
#include "../external/spdlog/include/spdlog/spdlog.h"
#include "../external/spdlog/include/spdlog/sinks/basic_file_sink.h"

// Function to display help information on how to use the program
void printHelp() {
    std::cout << "\nImage Processing Tool\n\n";
    std::cout << "Usage:\n";
    std::cout << "  ./image_processor --input <input> --method <method> [options]\n\n";

    std::cout << "Required arguments:\n";
    std::cout << "  -i, --input <path>    Input image file path\n";
    std::cout << "  -m, --method <name>   Processing method to use:\n";
    std::cout << "                        (sequential, parallel, advanced, integral, adaptive_median, all)\n\n";

    std::cout << "Options:\n";
    std::cout << "  -o, --output <path>   Output file path (required for some methods)\n";
    std::cout << "  -t, --threshold <num> Threshold value (default: 128)\n";
    std::cout << "  -h, --help            Show this help message\n\n";
    std::cout << "  -w, --window_size <num>  Kernel size for adaptive methods (default: 15)\n";
    std::cout << "  --k <num>               Parameter k for Sauvola/Nick (default: 0.2)\n";
    std::cout << "  --R <num>               Dynamic range R for Sauvola (default: 128.0)\n";

    // Examples of command-line usage
    std::cout << "Examples:\n";
    std::cout << "  Basic thresholding:     ./image_processor -i input.jpg -o out.jpg -m sequential -t 150\n";
    std::cout << "  Sauvola and Nick binarization:   ./image_processor --input in.png --method advanced\n";
    std::cout << "  Run all methods:        ./image_processor -i image.ppm -o results/ -m all\n";
    std::cout << "  Show help:              ./image_processor --help\n";
}

// Main function, entry point of the program
int main(int argc, char *argv[]) {
    std::cout << "Program started, writing to output.log!" << std::endl;

    try {
        // Initializing the logger to write logs to "logs/output.log"
        auto logger = spdlog::basic_logger_mt("file_logger", "logs/output.log");
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::info);
        spdlog::info("\n\n***** Program started *****\n\n");

        // Variables to store input parameters
        std::string input_path;
        std::string output_path;
        std::string method;
        int threshold = 128;   // Default threshold value
        int window_size = 15;  // Default window size for adaptive methods
        float k = 0.2f;        // Default parameter k for Sauvola/Nick
        float R = 128.0f;      // Default dynamic range R for Sauvola

        // Parsing command line arguments
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            // Show help message and exit
            if (arg == "--help" || arg == "-h") {
                printHelp();
                return 0;
            }
            // Input file path
            if (arg == "--input" || arg == "-i") {
                if (i + 1 < argc) {
                    input_path = argv[++i];
                } else {
                    spdlog::error("Missing value for --input");
                    std::cout << "Missing input!" << std::endl;
                    return 1;
                }
            }
            // Output file path
            else if (arg == "--output" || arg == "-o") {
                if (i + 1 < argc) {
                    output_path = argv[++i];
                } else {
                    spdlog::error("Missing value for --output");
                    std::cout << "Missing output!" << std::endl;
                    return 1;
                }
            }
            // Processing method
            else if (arg == "--method" || arg == "-m") {
                if (i + 1 < argc) {
                    method = argv[++i];
                } else {
                    spdlog::error("Missing value for --method");
                    std::cout << "Missing method!" << std::endl;
                    return 1;
                }
            }
            // Threshold value
            else if (arg == "--threshold" || arg == "-t") {
                if (i + 1 < argc) {
                    try {
                        threshold = std::stoi(argv[++i]);
                    } catch (const std::exception& e) {
                        spdlog::error("Invalid threshold value: {}", e.what());
                        return 1;
                    }
                } else {
                    spdlog::error("Missing value for --threshold");
                    std::cout << "Missing threshold!" << std::endl;
                    return 1;
                }
            }
            // Window size for adaptive methods
            else if (arg == "--window_size" || arg == "-w") {
                if (i + 1 < argc) {
                    try {
                        window_size = std::stoi(argv[++i]);
                    } catch (const std::exception& e) {
                        spdlog::error("Invalid window_size value: {}", e.what());
                        return 1;
                    }
                } else {
                    spdlog::error("Missing value for --window_size");
                    return 1;
                }
            }
            // Parameter k for Sauvola/Nick
            else if (arg == "--k") {
                if (i + 1 < argc) {
                    try {
                        k = std::stof(argv[++i]);
                    } catch (const std::exception& e) {
                        spdlog::error("Invalid k value: {}", e.what());
                        return 1;
                    }
                } else {
                    spdlog::error("Missing value for --k");
                    return 1;
                }
            }
            // Dynamic range R for Sauvola
            else if (arg == "--R") {
                if (i + 1 < argc) {
                    try {
                        R = std::stof(argv[++i]);
                    } catch (const std::exception& e) {
                        spdlog::error("Invalid R value: {}", e.what());
                        return 1;
                    }
                } else {
                    spdlog::error("Missing value for --R");
                    return 1;
                }
            }
            // Unknown argument handling
            else {
                spdlog::error("Unknown argument: {}", arg);
                std::cout << "Unknown arguments!" << std::endl;
                return 1;
            }
        }

        // Ensure required arguments are provided
        if (input_path.empty()) {
            spdlog::error("Input path is required (use --input)");
            printHelp();
            return 1;
        }
        if (method.empty()) {
            spdlog::error("Method is required (use --method)");
            printHelp();
            return 1;
        }

        // Validate the method name
        const std::string valid_methods[] = {"sequential", "parallel", "advanced", "integral", "adaptive_median", "all"};
        bool valid = false;
        for (const auto& m : valid_methods) {
            if (method == m) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            spdlog::error("Invalid method: {}", method);
            return 1;
        }

        // Execute the selected processing method
        if (method == "sequential") {
            binarize_image(input_path, output_path, threshold);
        }
        else if (method == "parallel") {
            binarize_image_parallel(input_path, output_path, threshold);
        }
        else if (method == "advanced") {
            process_advanced_binarization(input_path, window_size, k, R);
        }
        else if (method == "integral") {
            process_integral_binarization(input_path, window_size, k, R);
        }
        else if (method == "adaptive_median") {
            adaptive_median_filter(input_path, output_path);
        }
        else if (method == "all") {
            binarize_image_parallel(input_path, output_path, threshold);
            process_advanced_binarization(input_path, window_size, k, R);
            process_integral_binarization(input_path, window_size, k, R);
            adaptive_median_filter(input_path, output_path);
        }

        spdlog::info("***** Program finished successfully *****\n\n");
    } catch (const std::exception &e) {
        spdlog::critical("Unhandled exception: {}", e.what());
        printHelp();
        return 1;
    }

    return 0;
}
