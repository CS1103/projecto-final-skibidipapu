@echo off
REM Build script for Neural Network Project (Windows)

echo === Building Neural Network Project ===

REM Check if build directory exists, if not create it
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

REM Navigate to build directory
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake ..

REM Check if CMake configuration was successful
if %ERRORLEVEL% neq 0 (
    echo Error: CMake configuration failed!
    exit /b 1
)

REM Build the project
echo Building project...
cmake --build . --config Release

REM Check if build was successful
if %ERRORLEVEL% neq 0 (
    echo Error: Build failed!
    exit /b 1
)

echo === Build completed successfully! ===
echo.
echo To run the demo:
echo   .\neural_net_demo.exe
echo.
echo To run tests:
echo   .\neural_net_tests.exe
echo.
echo To clean build:
echo   cmake --build . --target clean 