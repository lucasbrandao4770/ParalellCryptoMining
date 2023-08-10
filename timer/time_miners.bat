:: Navigate to the top-level directory
cd ..

:: Compile and run the Sequential miner
echo Running Sequential Miner:
cd .\sequential
mingw32-make
powershell -command "Measure-Command { .\sequential_miner.exe }"
mingw32-make clean
cd ..

:: Compile and run the OpenMP miner
echo Running OpenMP Miner:
cd .\cpu_parallel
mingw32-make
powershell -command "Measure-Command { .\openmp_miner.exe }"
mingw32-make clean
cd ..

:: Compile and run the CUDA miner
echo Running CUDA Miner:
cd .\gpu_parallel
mingw32-make
powershell -command "Measure-Command { .\cuda_miner.exe }"
mingw32-make clean
cd ..
