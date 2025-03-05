# Algorithm-Engineering-Course

## Image Processing Tool

A C++ program for image binarization and filtering with multiple algorithms and parallel processing support.

## Features

- **Binarization Methods:**
  - Sequential Thresholding
  - Parallel Thresholding
  - Sauvola & Nick Adaptive Thresholding
  - Integral Image Binarization
- **Filters:**
  - Adaptive Median Filter
- **Batch Processing:** Run multiple methods sequentially
- **Logging:** Detailed operation logging to `logs/output.log`

## Requirements

- C++17 compiler
- stb_image
- OpenMP
- CMake 3.12+
- spdlog (included in `external/`)

## Installation

1. Clone repository:
   ```bash
   git clone https://github.com/SalimSama/Algorithm-Engineering-Course
   cd project
   ```

2. Build with CMake:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```


## Usage

```bash
./image_processor [OPTIONS]
```

### Options

| Flag                | Description                                  | Required               |
|---------------------|--------------------------------------------|------------------------|
| `-i, --input <PATH>`  | Input image path                           | Yes                    |
| `-m, --method <NAME>` | Processing method (see below)              | Yes                    |
| `-o, --output <PATH>` | Output path (required for some methods)     | Depends on method      |
| `-t, --threshold <NUM>` | Threshold value (default: 128)            | No                     |
| `-h, --help`         | Show help message                          | No                     |

### Available Methods

| Method Name       | Description                                  |
|------------------|--------------------------------------------|
| `sequential`     | Basic threshold binarization               |
| `parallel`       | Multi-threaded threshold binarization      |
| `advanced`        | Sauvola & Nick adaptive thresholding       |
| `integral`       | Integral image binarization                |
| `adaptive_median`| Adaptive median filter                     |
| `all`            | Run parallel + integral + adaptive_median  |

## Examples

1. Basic thresholding with custom value:
   ```bash
   ./image_processor -i input.jpg -o output.jpg -m sequential -t 150
   ```

2. Sauvola adaptive thresholding:
   ```bash
   ./image_processor --input image.png --method advanced
   ```

3. Process all methods with default threshold:
   ```bash
   ./image_processor -i input.ppm -o results/ -m all
   ```

4. Get help:
   ```bash
   ./image_processor --help
   ```

## Output

- Processed images saved to specified output path
- Detailed logs in `logs/output.log`:
  ```log
  [2023-08-20 14:30:45] [info] ***** Program started *****
  [2023-08-20 14:30:45] [info] Loading image from: input.jpg
  [2023-08-20 14:30:46] [info] Applied parallel binarization (128ms)
  [2023-08-20 14:30:46] [info] ***** Program finished successfully *****
  ```

## Troubleshooting

**Common Issues:**
- *Undefined reference to OpenCV functions:* Verify OpenCV installation
- *Method requires output path:* Add `-o` parameter for methods needing output
- *Unsupported image format:* Use common formats (`.jpg`, `.png`, `.ppm`)
- *Permission denied:* Make executable with `chmod +x image-processor`

**Logging:**
- Check `logs/output.log` for detailed error messages
- Enable debug logging by changing `spdlog::level::info` to `spdlog::level::debug`

## License

GNU General Public License v3.0
