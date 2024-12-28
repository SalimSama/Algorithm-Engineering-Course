// main.cpp
#include "image_utils.h"
#include "../external/spdlog/include/spdlog/spdlog.h"
#include "../external/spdlog/include/spdlog/sinks/basic_file_sink.h"

int main(int argc, char *argv[]) {
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

        binarize_image(input_path, output_path, threshold);
        binarize_image_parallel(input_path, output_path, threshold);
        process_advanced_binarization(input_path);
        process_integral_binarization(input_path);

        spdlog::info("***** Program finished successfully *****\n\n");
    } catch (const std::exception &e) {
        spdlog::critical("Unhandled exception: {}", e.what());
        return 1;
    }
    return 0;
}
