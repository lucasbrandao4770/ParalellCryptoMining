:: Navigate to the top-level directory
cd ..

:: Save the top-level directory path
set TOP_DIR=%CD%

:: Create the results directory if it doesn't exist
mkdir results

:: Get the hour
set HOUR=%time:~0,2%

:: Check if the hour is a single digit and prepend "0" if necessary
if "%HOUR:~0,1%"==" " set HOUR=0%HOUR:~1,1%

:: Construct filename with date and time
set FILENAME=%date:/=-%_%HOUR%-%time:~3,2%-%time:~6,2%
set FILENAME=%FILENAME: =_%
set FILENAME=%FILENAME:,=%
set FILENAME=%TOP_DIR%/results/%FILENAME%

:: Append command-line arguments to the filename
set ARGS=%*
if not "%ARGS%" == "" (
    set FILENAME=%FILENAME%_%ARGS%
)

:: Add .txt extension
set FILENAME=%FILENAME%.txt

:: Compile and run the Sequential miner
echo Running Sequential Miner:
cd .\sequential
mingw32-make
powershell -command "echo 'Time spent for sequential code:' | Out-File -FilePath '%FILENAME%' -Append; Measure-Command { .\sequential_miner.exe %ARGS% } | Out-File -FilePath '%FILENAME%' -Append"
mingw32-make clean
cd ..

:: Compile and run the OpenMP miner
echo Running OpenMP Miner:
cd .\cpu_parallel
mingw32-make
powershell -command "echo 'Time spent for OpenMP code:' | Out-File -FilePath '%FILENAME%' -Append; Measure-Command { .\openmp_miner.exe %ARGS% } | Out-File -FilePath '%FILENAME%' -Append"
mingw32-make clean
cd ..

:: Compile and run the CUDA miner
echo Running CUDA Miner:
cd .\gpu_parallel
mingw32-make
powershell -command "echo 'Time spent for CUDA code:' | Out-File -FilePath '%FILENAME%' -Append; Measure-Command { .\cuda_miner.exe %ARGS% } | Out-File -FilePath '%FILENAME%' -Append"
mingw32-make clean
cd ..
