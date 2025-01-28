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

void binarize_image(const std::string &input_path, std::string output_path, int threshold) {
    spdlog::info("Starting sequential binarization with threshold {} for: {}", threshold, input_path);
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
        return;
    }

    std::cout << "Number of channels " << channels << std::endl;

    if (output_path.empty()) {
        output_path = make_output_path(input_path);
    }

    std::vector<unsigned char> out(width * height * channels);
    auto start = std::chrono::high_resolution_clock::now();

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

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("Sequential binarization completed in {} seconds.", duration.count());

    if (!write_binary_image(output_path, width, height, channels, out.data())) {
        spdlog::error("Failed to write binarized image: {}", output_path);
    } else {
        spdlog::info("Binarized image saved to: {}", output_path);
    }
    stbi_image_free(image);
}

void binarize_image_parallel(const std::string &input_path, std::string output_path, int threshold) {
    spdlog::info("Starting parallel binarization with threshold {} for: {}", threshold, input_path);
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
        return;
    }

    if (output_path.empty()) {
        output_path = make_output_path(input_path);
    }

    std::vector<unsigned char> out(width * height * channels);
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for simd
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

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("Parallel binarization completed in {} seconds.", duration.count());

    if (!write_binary_image(output_path, width, height, channels, out.data())) {
        spdlog::error("Failed to write parallel binarized image: {}", output_path);
    } else {
        spdlog::info("Parallel binarized image saved to: {}", output_path);
    }
    stbi_image_free(image);
}

void local_mean_std(const unsigned char* gray,
                    int width, int height,
                    int x, int y,
                    int half_win,
                    float &mean, float &stddev)
{
    int count = 0;
    float sum = 0.0f, sum_sq = 0.0f;
    for (int dy = -half_win; dy <= half_win; dy++) {
        for (int dx = -half_win; dx <= half_win; dx++) {
            int yy = y + dy;
            int xx = x + dx;
            if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
                unsigned char val = gray[yy * width + xx];
                sum += val;
                sum_sq += val * val;
                count++;
            }
        }
    }
    mean = sum / count;
    float var = (sum_sq / count) - (mean * mean);
    stddev = (var > 0) ? std::sqrt(var) : 0.0f;
}

// Beispiel-Funktion: Sauvola-Binarisierung
void adaptive_binarize(const unsigned char* gray,
                       unsigned char* out,
                       int width, int height,
                       int window_size,
                       const std::function<float(float mean, float stddev)> &threshold_func) {
    int half_win = window_size / 2;

    spdlog::info("Starting adaptive binarization with window size {}");

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float mean = 0.0f, stddev = 0.0f;
            local_mean_std(gray, width, height, x, y, half_win, mean, stddev);
            float threshold = threshold_func(mean, stddev);
            out[y * width + x] = (gray[y * width + x] > threshold) ? 255 : 0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("Adaptive binarization completed in {} seconds.", duration.count());
}

void sauvola_binarize(const unsigned char* gray,
                      unsigned char* out,
                      int width, int height,
                      int window_size,
                      float k,
                      float R) {
    spdlog::info("Starting Sauvola binarization with window size {}, k={}, R={}.", window_size, k, R);

    auto threshold_func = [k, R](float mean, float stddev) {
        return mean * (1.0f + k * ((stddev / R) - 1.0f));
    };

    adaptive_binarize(gray, out, width, height, window_size, threshold_func);

    spdlog::info("Sauvola binarization completed.");
}

void nick_binarize(const unsigned char* gray,
                   unsigned char* out,
                   int width, int height,
                   int window_size,
                   float k) {
    spdlog::info("Starting Nick binarization with window size {}, k={}.", window_size, k);

    auto threshold_func = [k](float mean, float stddev) {
        return mean + k * stddev;
    };

    adaptive_binarize(gray, out, width, height, window_size, threshold_func);

    spdlog::info("Nick binarization completed.");
}

void process_advanced_binarization(const std::string &input_path) {
    spdlog::info("Processing advanced binarization for: {}", input_path);
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
        return;
    }

    std::vector<unsigned char> gray(width * height);
    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        gray[i] = static_cast<unsigned char>(
                0.2126f * image[i * channels + 0] +
                0.7152f * image[i * channels + 1] +
                0.0722f * image[i * channels + 2]);
    }

    std::string output_path_sauvola = make_output_path(input_path) + "_sauvola.png";
    std::string output_path_nick = make_output_path(input_path) + "_nick.png";

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> output_sauvola(width * height);
    sauvola_binarize(gray.data(), output_sauvola.data(), width, height, 15, 0.2f, 128.0f);

    if (!write_binary_image(output_path_sauvola, width, height, 1, output_sauvola.data())) {
        spdlog::error("Failed to write Sauvola output image: {}", output_path_sauvola);
    } else {
        spdlog::info("Sauvola binarized image saved to: {}", output_path_sauvola);
    }

    std::vector<unsigned char> output_nick(width * height);
    nick_binarize(gray.data(), output_nick.data(), width, height, 15, 0.1f);

    if (!write_binary_image(output_path_nick, width, height, 1, output_nick.data())) {
        spdlog::error("Failed to write Nick output image: {}", output_path_nick);
    } else {
        spdlog::info("Nick binarized image saved to: {}", output_path_nick);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("Advanced binarization process completed in {} seconds.", duration.count());

    stbi_image_free(image);
}




void computeIntegralImages(const unsigned char* gray,
                           int width, int height,
                           std::vector<double>& integralImg,
                           std::vector<double>& integralImgSq)
{
    integralImg.resize(width * height, 0.0);
    integralImgSq.resize(width * height, 0.0);

    // 1. Zeilenweiser Durchgang (Prefix-Sums pro Zeile)
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        double sumRow = 0.0, sumRowSq = 0.0;
        for (int x = 0; x < width; x++) {
            double val = static_cast<double>(gray[y * width + x]);
            sumRow   += val;
            sumRowSq += val * val;
            integralImg[y * width + x]   = sumRow;
            integralImgSq[y * width + x] = sumRowSq;
        }
    }

    // 2. Spaltenweiser Durchgang (Prefix-Sums der bereits zeilenweise summierten Daten)
#pragma omp parallel for
    for (int x = 0; x < width; x++) {
        double colSum = 0.0, colSumSq = 0.0;
        for (int y = 0; y < height; y++) {
            colSum   += integralImg[y * width + x];
            colSumSq += integralImgSq[y * width + x];
            integralImg[y * width + x]   = colSum;
            integralImgSq[y * width + x] = colSumSq;
        }
    }
}

inline double getSum(const std::vector<double>& integralImg,
                     int x1, int y1, int x2, int y2, int width, int height)
{
    if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
    if (x2 >= width) x2 = width - 1;
    if (y2 >= height) y2 = height - 1;

    double A = (x1 > 0 && y1 > 0) ? integralImg[(y1 - 1) * width + (x1 - 1)] : 0.0;
    double B = (y1 > 0) ? integralImg[(y1 - 1) * width + x2] : 0.0;
    double C = (x1 > 0) ? integralImg[y2 * width + (x1 - 1)] : 0.0;
    double D = integralImg[y2 * width + x2];
    return D + A - B - C;
}

void local_mean_std_integral(const std::vector<double>& integralImg,
                             const std::vector<double>& integralImgSq,
                             int width, int height,
                             int x, int y, int half_win,
                             float &mean, float &stddev)
{
    int x1 = x - half_win, y1 = y - half_win;
    int x2 = x + half_win, y2 = y + half_win;
    double area = (x2 - x1 + 1) * (y2 - y1 + 1);

    double sum = getSum(integralImg, x1, y1, x2, y2, width, height);
    double sumSq = getSum(integralImgSq, x1, y1, x2, y2, width, height);

    double m = sum / area;
    double var = (sumSq / area) - (m * m);
    mean = static_cast<float>(m);
    stddev = (var > 0.0) ? static_cast<float>(std::sqrt(var)) : 0.0f;
}

void adaptive_binarize_integral(const unsigned char* gray,
                       unsigned char* out,
                       int width, int height,
                       int window_size,
                       const std::vector<double>& integralImg,
                       const std::vector<double>& integralImgSq,
                       const std::function<float(float mean, float stddev)> &threshold_func) {
    int half_win = window_size / 2;

    spdlog::info("Starting adaptive integral binarization with window size {}");

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float mean = 0.0f, stddev = 0.0f;
            local_mean_std_integral(integralImg, integralImgSq, width, height, x, y, half_win, mean, stddev);
            float threshold = threshold_func(mean, stddev);
            out[y * width + x] = (gray[y * width + x] > threshold) ? 255 : 0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("Adaptive integral binarization completed in {} seconds.", duration.count());
}

void sauvola_binarize_integral(const unsigned char* gray,
                               unsigned char* out,
                               int width, int height,
                               int window_size,
                               float k,
                               float R,
                               const std::vector<double>& integralImg,
                               const std::vector<double>& integralImgSq) {
    spdlog::info("Starting Integral Sauvola binarization with window size {}, k={}, R={}.", window_size, k, R);

    auto threshold_func = [k, R](float mean, float stddev) {
        return mean * (1.0f + k * ((stddev / R) - 1.0f));
    };

    adaptive_binarize_integral(gray, out, width, height, window_size, integralImg, integralImgSq, threshold_func);

    spdlog::info("Integral Sauvola binarization completed.");
}

void process_integral_binarization(const std::string &input_path) {
    spdlog::info("Processing integral binarization for: {}", input_path);
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
        return;
    }

    std::vector<unsigned char> gray(width * height);
    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        gray[i] = static_cast<unsigned char>(
                0.2126f * image[i * channels + 0] +
                0.7152f * image[i * channels + 1] +
                0.0722f * image[i * channels + 2]);
    }

    std::vector<double> integralImg, integralImgSq;
    computeIntegralImages(gray.data(), width, height, integralImg, integralImgSq);

    std::string output_path_integral = make_output_path(input_path) + "_integral_sauvola.png";
    std::vector<unsigned char> output_integral(width * height);

    auto start = std::chrono::high_resolution_clock::now();

    sauvola_binarize_integral(gray.data(), output_integral.data(), width, height, 15, 0.2f, 128.0f, integralImg, integralImgSq);

    if (!write_binary_image(output_path_integral, width, height, 1, output_integral.data())) {
        spdlog::error("Failed to write Integral Sauvola output image: {}", output_path_integral);
    } else {
        spdlog::info("Integral Sauvola binarized image saved to: {}", output_path_integral);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    spdlog::info("Integral binarization process completed in {} seconds.", duration.count());

    stbi_image_free(image);
}

void majority_vote(const std::string &input_path, std::string output_path) {
    // compute n images (one for each method that we choose to apply)
    spdlog::info("Starting majority vote for: {}", input_path);
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
        return;
    }

    if (output_path.empty()) {
        output_path = make_output_path(input_path);
    }

    std::vector<unsigned char> out(width * height * channels);
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> img_binarize_image_parallel(width * height * channels);
    std::vector<unsigned char> (width * height * channels);

    // add all intensity values to a new image with uncapped intensities
    // divide intensity values by n (this should scale the image back to the prior scale)
    // binarize image again (using half the maximum intensity as the threshold)
}



std::vector<unsigned char> get_window(unsigned char *image,
                                      int width, int height,
                                      int x, int y,
                                      int window_size)
{
    std::vector<unsigned char> output_window(window_size * window_size);
    const int half_win = window_size / 2;
    unsigned int array_pos = 0;
    for (int dy = -half_win; dy <= half_win; dy++) {
        for (int dx = -half_win; dx <= half_win; dx++) {
            array_pos += 1;
            int yy = y + dy;
            int xx = x + dx;
            if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
                output_window[array_pos] = image[xx * width + yy];
            }
        }
    }

    return output_window;

}

void increase_window_size(unsigned char *image, std::vector<unsigned char> *temp_window, int old_winsize,
                            const int width, const int height, const int xpos, const int ypos) {

    // go one row up, down, left and right and push everything onto temp_window

    //std::cout << "WINDOW SIZE INCREASED with width image " << width << " and height " << height << std::endl;
    // std::cout << "The initial positions are xpos = " << xpos << ", ypos = " << ypos << "." << std::endl;

    int half_win = old_winsize/2;
    // std::cout << half_win << " is half the window size \n";


    // row over the window needs to be inserted, if it is still inside the image
    for (int x = xpos - half_win -1; x <= xpos + half_win + 1; x++) {
        //std::cout << x << "\n";
        if (ypos - half_win - 1 >= 0 && x >= 0 && x <= width - 1) {         // 1 2 pos 4 5 wenn aktuelles x im Bild ist kommt es rein
            //std::cout << "Condition: " << ypos-half_win - 1 << std::endl;
            // std::cout << "Inserted in window! Case 1 \n";
            // std::cout << "Position " << x << ", " << ypos << std::endl;
            temp_window->push_back(image[(ypos - half_win - 1) * width + x]);
        }

        // the same goes for the row below the window
        if (ypos + half_win + 1 <= height - 1 && x >= 0 && x <= width - 1) {
            // std::cout << "Condition: " << ypos-half_win + 1 << std::endl;
            // std::cout << "Inserted in window! Case 2 \n";
            // std::cout << "Position " << x << ", " << ypos << std::endl;
            temp_window->push_back(image[(ypos + half_win + 1) * width + x]);
        }
    }

    for (int y = ypos - half_win; y <= ypos + half_win; y++) {
        if (xpos - half_win - 1 >= 0 && y >= 0 && y <= height -1) {
            // std::cout << "Inserted in window! Case 3 \n";
            // std::cout << "Position " << xpos << ", " << y << std::endl;
            temp_window->push_back(image[y * width + xpos - half_win -1]);
        }
        if (xpos + half_win + 1 <= width - 1 && y >= 0 && y <= height - 1) {
            // std::cout << "Inserted in window! Case 4 \n";
            // std::cout << "Position " << xpos << ", " << y << std::endl;
            temp_window->push_back(image[y * width + xpos + half_win + 1]);
        }
    }

    // std::cout << std::endl << std::endl;

}

bool comp(unsigned char a, unsigned char b) {
    return a > b;
}

void print_vec(const std::vector<unsigned char>& vec) {
    // for (unsigned char i: vec) {
    //     std::cout << i << " ";
    // }
}

void adaptive_median_filter(unsigned char *image, const int width, const int height, const int channels,
                            int min_win_size, int max_window_size) {
    if (!image) {
        std::cout << "Null pointer in adaptive_median_filter \n" ;
        return;
    }

    std::vector<unsigned char> temp_window;
    unsigned char local_median, local_max, local_min = 0;
    int current_win_size;
    std::cout << "Maße " << width << " " << height << std::endl;
// #pragma omp parallel for
    for (int y = 0; y < height - 1; y++) {
        for (int x = 0; x < width - 1; x++) {
            // std::cout << "Process position "<< x << " " << y << std::endl ;
            bool flag = false; // have we calculated the pixel or not flag

            // RACE CONDITION, new window for each thread
            temp_window.clear(); // clear the window
            temp_window.push_back(image[y * width + x]); // only insert the pixel at the current position
            current_win_size = 1; // keep track of the window size

            while (current_win_size != min_win_size) { // increase the window size until we reach the minimum size requirements
                print_vec(temp_window);
                increase_window_size(image, &temp_window, current_win_size, width, height, x, y);
                current_win_size++;

            }
            // std::cout << "ESCAPED WHILE 1 " << std::endl;

            int pos = y * width + x;
            unsigned char pxl = image[pos];

            // find min, find max, find median
            sort(temp_window.begin(), temp_window.end(), comp);
            local_median = temp_window[temp_window.size() / 2];
            local_max = temp_window[temp_window.size()-1];
            local_min = temp_window[0];

            while (flag == false && current_win_size < max_window_size) {       // while we haven't found the pixel or reached the max size
                if (local_median > local_min && local_median < local_max) {
                    if (pxl > local_min && pxl < local_max) {
                        flag = true; // simply keep the pixel's vale
                    } else {
                        image[pos] = local_median; // set the value as the local median
                    }
                } else {
                    increase_window_size(image, &temp_window, current_win_size, width, height, x, y); // increase the window size
                    current_win_size++;

                    // find min, find max, find median
                    sort(temp_window.begin(), temp_window.end(), comp);
                    local_median = temp_window[temp_window.size() / 2];
                    local_max = temp_window[temp_window.size()-1];
                    local_min = temp_window[0];
                }

            }

            // again just keep the pixel's value

        }
    }

};


void adaptive_median_filter_test(const std::string &input_path, std::string output_path) {
    spdlog::info("Starting adaptive median filter ");
    int width, height, channels;
    unsigned char *image = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    std::cout << "Image size: " << width * height << std::endl; // PROBLEM: RAN OUT OF STACK MEMORY; HAVE TO REWRITE IT AS A HEAP INSTEAD APPARENTLY

    if (!image) {
        spdlog::error("Failed to load image: {}", input_path);
        return;
    }

    if (output_path.empty()) {
        output_path = make_output_path(input_path);
    }

    unsigned char gray[width * height];

#pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        gray[i] = static_cast<unsigned char>(
                0.2126f * image[i * channels + 0] +
                0.7152f * image[i * channels + 1] +
                0.0722f * image[i * channels + 2]);
    }

    // TURN THE IMAGE INTO INTENSITY IMAGE
    int n = sizeof(gray) / sizeof(gray[0]);

    std::vector<unsigned char> out(gray, gray + n); //converts image array to vector
    adaptive_median_filter(gray, width, height, channels, 3, 10);

    if (!write_binary_image(output_path, width, height, channels, out.data())) {
        spdlog::error("Failed to write filtered image: {}", output_path);
    } else {
        spdlog::info("Filtered image saved to: {}", output_path);
    }

    stbi_image_free(image);

}

