
echo "=== Building Neural Network Project ==="

if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

echo "Configuring with CMake..."
cmake ..

if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed!"
    exit 1
fi

echo "Building project..."
make -j4

if [ $? -ne 0 ]; then
    echo "Error: Build failed!"
    exit 1
fi

echo "=== Build completed successfully! ==="
echo ""
echo "To run the demo:"
echo "  ./build/neural_net_demo"
echo ""
echo "To run tests:"
echo "  ./build/neural_net_tests"
echo ""
echo "To clean build:"
echo "  make clean" 