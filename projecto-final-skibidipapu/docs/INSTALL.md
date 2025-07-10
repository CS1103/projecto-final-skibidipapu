# Installation Guide

## Prerequisites

### Required Software
- **Compiler**: GCC 11 or higher, or Clang 12 or higher
- **CMake**: Version 3.18 or higher
- **Eigen**: Version 3.4 or higher
- **Git**: For cloning the repository

### Installing Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install build-essential cmake git
sudo apt install libeigen3-dev
```

#### macOS
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install cmake eigen
```

#### Windows
1. Install Visual Studio 2019 or later with C++ support
2. Install CMake from https://cmake.org/download/
3. Install Eigen from https://eigen.tuxfamily.org/

## Building the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/projecto-final-skibidipapu.git
cd projecto-final-skibidipapu
```

### Step 2: Create Build Directory
```bash
mkdir build
cd build
```

### Step 3: Configure with CMake
```bash
cmake ..
```

### Step 4: Build the Project
```bash
make -j4
```

### Step 5: Run Tests (Optional)
```bash
make test
```

### Step 6: Run the Demo
```bash
./neural_net_demo
```

## Troubleshooting

### Common Issues

1. **Eigen not found**
   ```
   CMake Error: Could not find Eigen3
   ```
   **Solution**: Install Eigen3 development package
   ```bash
   sudo apt install libeigen3-dev  
   brew install eigen             
   ```

2. **Compiler version too old**
   ```
   error: 'auto' keyword not supported
   ```
   **Solution**: Update your compiler to GCC 11+ or Clang 12+

3. **CMake version too old**
   ```
   CMake Error: CMake 3.18 or higher is required
   ```
   **Solution**: Update CMake to version 3.18 or higher

### Platform-Specific Notes

#### Windows
- Use Visual Studio Developer Command Prompt
- Ensure Eigen is properly installed and CMake can find it
- Consider using vcpkg for easier dependency management

#### macOS
- If using Xcode, ensure command line tools are installed
- Eigen is available via Homebrew

#### Linux
- Most distributions have Eigen3 available in their package managers
- For custom Eigen installation, set `EIGEN3_INCLUDE_DIR` environment variable

## Verification

After successful installation, you should be able to:

1. Build the project without errors
2. Run the demo program: `./neural_net_demo`
3. See output showing neural network training progress
4. Run tests: `./neural_net_tests`

The demo will show:
- Data generation and preprocessing
- Neural network architecture creation
- Training progress with loss values
- Final accuracy on test set 