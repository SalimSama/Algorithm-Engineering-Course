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

| Flag                      | Description                                    | Required |
|---------------------------|------------------------------------------------|----------|
| `-i, --input <PATH>`      | Input image path                               | Yes      |
| `-m, --method <NAME>`     | Processing method (see below)                  | Yes      |
| `-o, --output <PATH>`     | Output path (required for some methods)        | No       |
| `-t, --threshold <NUM>`   | Threshold value (default: 128)                 | No       |
| `-w, --window_size <NUM>` | Kernel size for adaptive methods (default: 15) | No       |
| `--k <NUM>`               | for Sauvola/Nick (default: 0.2)                | No       |
| `--R <NUM>`               | for Sauvola (default: 128.0)                   | No       |
| `-h, --help`              | Show help message                              | No       |

### Available Methods

| Method Name      | Description                                 |
|------------------|---------------------------------------------|
| `sequential`     | Basic threshold binarization                |
| `parallel`       | Multi-threaded threshold binarization       |
| `advanced`       | Sauvola & Nick adaptive thresholding        |
| `integral`       | Integral image binarization                 |
| `adaptive_median`| Adaptive median filter                      |
| `all`            | Run parallel + integral + adaptive_median   |

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

## Parameter Tuning Tutorial

Optimize binarization and filtering results by understanding these key parameters:

### Parameter Art & Effects

```text
[Window Size]            [R Value]               [k Value]
  ┌───────────┐          ▲ Dynamic Range          ▲ Sensitivity
  │  ●  ●  ●  │          │                        │
  │  ●  ●  ●  │        Higher ◄─┐  ┌─► Lower      Higher ◄─┐  ┌─► Lower  
  │  ●  ●  ●  │        Less     │  │   More      Preserve  │  │   Clean
  └───────────┘       Sensitive ▼  ▼ Sensitive   Text      ▼  ▼ Background
  Larger: Handle        (Default: 128)            (Range: 0.05-0.5)
  illumination          Ideal for:                Typical:
  variations             - Clean backgrounds       - 0.34 default
  Smaller: Keep         - Noisy documents         - 0.2-0.25 light text
  fine details                                    
```

### Key Parameters Guide

1. **Window Size (Neighborhood)**
   - **What it does**: Controls local area analysis
   - **Sweet Spot**: 
     - Documents: 15-35px 
     - Natural Images: 25-45px
   - **Tradeoff**: 
     - Larger = Better illumination handling
     - Smaller = Sharper text edges

2. **R (Dynamic Range)**
   - **Formula**: `Threshold = mean * (1 + k*(std_dev/R - 1))`
   - **Pro Tips**:
     - Increase R (128→160) for:
       - Consistent backgrounds
       - Low-contrast documents
     - Decrease R (128→96) for:
       - Vintage/aged documents
       - Noisy images

3. **k (Sensitivity Factor)**
   - **Visual Impact**:
     ```
     k=0.15          k=0.34          k=0.5
     ┌────────┐      ┌────────┐      ┌────────┐
     │████  ██│      │███   ██│      │██    ██│
     │  ██ █  │      │ ██ ██  │      │  ███   │
     └────────┘      └────────┘      └────────┘
     Clean Background Balanced Approach    Faint Text
     ```
   - **Rule of Thumb**:
     - 0.15-0.25: Modern documents
     - 0.3-0.4: Historical archives
     - 0.45-0.5: Pencil sketches

### Optimization Workflow

1. **Baseline Test**
   ```bash
   ./image_processor -i doc.jpg -m advanced -o test1.jpg
   ```
   Defaults: Window=15, R=128, k=0.2

2. **Diagnose Issues**
   - Too much noise? Try:
     ```bash
     ./image_processor ... -w=35 --k=0.25 --R=140
     ```
   - Missing faint text? Try:
     ```bash
     ./image_processor ... -w=15 --k=0.45 --R=112
     ```

3. **Adaptive Median Filter Combo**
   For salt-and-pepper noise:
   ```bash
   # First denoise then binarize
   ./image_processor -i noisy.jpg -m adaptive_median -o denoised.jpg
   ./image_processor -i denoised.jpg -m advanced -o final.jpg
   ```

### Parameter Matrix Cheatsheet

| Scenario                | Window |  R  |  k  | Median Window |
|-------------------------|--------|-----|-----|---------------|
| Modern document scan    |   25   | 128 | 0.2 |       7       |
| Historical manuscript   |   15   | 96  | 0.4 |       11      |
| Camera-captured text    |   35   | 160 | 0.3 |       15      |
| Pencil sketch           |   20   | 64  | 0.5 |       -       |

### Visual Examples

1. **Underexposed Photo**
   - **Before**: Text merges with background
   - **Fix**: `window=40, R=110, k=0.4`
   - **After**: Clear character separation

2. **Ink Bleed Through**
   - **Before**: Reverse side text visible
   - **Fix**: `window=15, R=140, k=0.25`
   - **After**: Background noise suppressed

3. **Low-Contrast Fax**
   - **Before**: Gray text on gray background
   - **Fix**: `window=20, R=128, k=0.45`
   - **After**: Text becomes crisp black

### Pro Tip

- **Batch Testing**:
  ```bash
  for w in 15 25 35; do
    for k in 0.2 0.3 0.4; do
      ./image_processor -i doc.jpg -o output_${w}_${k}.jpg -m advanced -w=$w --k=$k
    done
  done
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

