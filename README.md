[![CC BY 4.0][cc-by-shield]][cc-by]

# ParalellCryptoMining
This project focuses on optimizing cryptocurrency mining through parallelism, comparing the efficiency of OpenMP (CPU) and RAPIDS (GPU) against a sequential C baseline.

## Dependencies

### General C Development
To compile and run the C code, you'll need a C compiler and associated development tools. Here's how you can install them:

- **Windows (using MinGW):**
  1. Download and install [MinGW](http://mingw-w64.org/doku.php/download/mingw-builds).
  2. Select the desired architecture and follow the installation instructions.
  3. Add MinGW's `bin` directory to your system's PATH environment variable.

- **Linux:**
  ```bash
  sudo apt-get install build-essential
  ```

- **macOS:**
  ```bash
  xcode-select --install
  ```

### CUDA Development (For GPU Parallelism)
If you intend to run the code using GPU parallelism, you'll need to install NVIDIA's CUDA toolkit:

1. Download the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your operating system.
2. Follow the installation instructions provided for your specific platform.

### OpenMP (For CPU Parallelism)
OpenMP support is generally included with the compiler. For MinGW on Windows, ensure you have the proper OpenMP-enabled version.

### GNU Make (For Using Makefile)
To compile the code using the provided Makefile, you'll need to install GNU Make:

- **Windows (using MinGW):**
  ```bash
  mingw32-make
  ```

- **Linux/macOS:**
  ```bash
  sudo apt-get install make
  ```

## Sequential

To compile and run the sequential baseline code, follow these steps:

1. Navigate to the `sequential` directory:
   ```bash
   cd path/to/sequential
   ```

2. Compile the code using the provided Makefile:
   ```bash
   mingw32-make
   ```

3. Run the compiled binary:
   ```bash
   ./sequential_miner
   ```

4. Clean up the compiled objects and executables:
    ```bash
    mingw32-make clean
    ```

Replace `path/to/sequential` with the actual path to the `sequential` directory in your project.


# License
This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
