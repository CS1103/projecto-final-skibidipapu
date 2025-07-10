#include <iostream>
#include <random>
#include <memory>
#include <chrono>
#include "../include/tensor.h"
#include "../include/nn_interfaces.h"
#include "../include/nn_activation.h"
#include "../include/nn_dense.h"
#include "../include/nn_loss.h"
#include "../include/nn_optimizer.h"
#include "../include/neural_network.h"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace utec::algebra;
using namespace utec::neural_network;
using namespace std;

template<typename T>
class RandomInitializer {
    mt19937 gen_;
    normal_distribution<T> dist_;
    
public:
    RandomInitializer(T mean = 0.0, T stddev = 0.1) 
        : gen_(random_device{}()), dist_(mean, stddev) {}
    
    void operator()(Tensor<T, 2>& tensor) {
        for (size_t i = 0; i < tensor.size(); ++i) {
            tensor[i] = dist_(gen_);
        }
    }
};

template<typename T>
class ZeroInitializer {
public:
    void operator()(Tensor<T, 2>& tensor) {
        tensor.fill(T(0));
    }
};

template<typename T>
pair<Tensor<T, 2>, Tensor<T, 2>> generate_xor_data(size_t num_samples = 10000) {
    mt19937 gen(random_device{}());
    uniform_real_distribution<T> dist(-0.1, 0.1);
    
    Tensor<T, 2> X(num_samples, 2);
    Tensor<T, 2> Y(num_samples, 1);
    
    for (size_t i = 0; i < num_samples; ++i) {
        T x1 = (i % 4 < 2) ? T(0) : T(1);
        T x2 = (i % 2 == 0) ? T(0) : T(1);
        
        X(i, 0) = x1 + dist(gen);
        X(i, 1) = x2 + dist(gen);
        
        Y(i, 0) = (x1 != x2) ? T(1) : T(0);
    }
    
    return {X, Y};
}

template<typename T>
class PerformanceMonitor {
    chrono::high_resolution_clock::time_point start_;
    
public:
    void start() {
        start_ = chrono::high_resolution_clock::now();
    }
    
    T elapsed_seconds() {
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start_);
        return T(duration.count()) / T(1000.0);
    }
};

int main() {
#ifdef _OPENMP
    cout << "OpenMP ACTIVO. Numero de hilos disponibles: " << omp_get_max_threads() << '\n';
#else
    cout << "OpenMP NO ACTIVO." << '\n';
#endif
    cout << "=== UTEC Neural Network Project ===" << '\n';
    cout << "Implementacion de Red Neuronal Multicapa con Sistema de Tensores UTEC" << '\n';
    cout << "=====================================================================" << '\n';
    
    using T = double;
    
    try {
        cout << "\n1. Generando datos de entrenamiento (problema XOR)..." << '\n';
        // Cambios para optimizar velocidad de entrenamiento
        const size_t epochs = 500;
        const size_t batch_size = 128;
        const T learning_rate = 0.01;
        
        auto [X_train, y_train] = generate_xor_data<T>(25000);
        cout << "   Datos generados: " << X_train.shape()[0] << " muestras, " 
                  << X_train.shape()[1] << " caracteristicas" << '\n';
        cout << "   X_train shape: [" << X_train.shape()[0] << ", " << X_train.shape()[1] << "]" << '\n';
        cout << "   y_train shape: [" << y_train.shape()[0] << ", " << y_train.shape()[1] << "]" << '\n';
        
        auto [X_test, y_test] = generate_xor_data<T>(1000);
        
        cout << "\n2. Configurando arquitectura de red neuronal..." << '\n';
        NeuralNetwork<T> network;
        
        cout << "   Creando inicializadores..." << '\n';
        RandomInitializer<T> weight_init(0.0, 0.1);
        ZeroInitializer<T> bias_init;
        cout << "   Inicializadores creados exitosamente" << '\n';
        
        cout << "   Agregando capa Dense(2, 64)..." << '\n';
        network.add_layer(std::make_unique<Dense<T>>(2, 64, weight_init, bias_init));
        cout << "   Agregando capa ReLU..." << '\n';
        network.add_layer(std::make_unique<ReLU<T>>());
        cout << "   Agregando capa Dense(64, 64)..." << '\n';
        network.add_layer(std::make_unique<Dense<T>>(64, 64, weight_init, bias_init));
        cout << "   Agregando otra capa ReLU..." << '\n';
        network.add_layer(std::make_unique<ReLU<T>>());
        cout << "   Agregando capa Dense(64, 1)..." << '\n';
        network.add_layer(std::make_unique<Dense<T>>(64, 1, weight_init, bias_init));
        cout << "   Agregando capa Sigmoid..." << '\n';
        network.add_layer(std::make_unique<Sigmoid<T>>());
        
        cout << "   Arquitectura: 2 -> 64 -> 64 -> 1(Sigmoid)" << '\n';
        
        cout << "\n3. Iniciando entrenamiento..." << '\n';
        cout << "   Epocas: " << epochs << '\n';
        cout << "   Tamano de batch: " << batch_size << '\n';
        cout << "   Learning rate: " << learning_rate << '\n';
        
        PerformanceMonitor<T> monitor;
        monitor.start();
        
        network.train<BCELoss, SGD>(X_train, y_train, epochs, batch_size, learning_rate);
        
        T training_time = monitor.elapsed_seconds();
        cout << "   Tiempo de entrenamiento: " << training_time << " segundos" << '\n';
        
        cout << "\n4. Evaluando rendimiento..." << '\n';
        
        auto predictions = network.predict(X_test);
        
        size_t correct = 0;
        for (size_t i = 0; i < y_test.shape()[0]; ++i) {
            T pred = predictions(i, 0) > 0.5 ? T(1) : T(0);
            T true_val = y_test(i, 0);
            if (pred == true_val) {
                correct++;
            }
        }
        
        T accuracy = T(correct) / T(y_test.shape()[0]) * T(100.0);
        
        cout << "   Precision en test: " << accuracy << "%" << '\n';
        cout << "   Muestras correctas: " << correct << "/" << y_test.shape()[0] << '\n';
        
        cout << "\n5. Demostracion XOR:" << '\n';
        cout << "   Entrada\tSalida Real\tSalida Predicha" << '\n';
        cout << "   -----------------------------------------" << '\n';
        
        Tensor<T, 2> xor_inputs(4, 2);
        xor_inputs(0, 0) = 0.0; xor_inputs(0, 1) = 0.0;  
        xor_inputs(1, 0) = 0.0; xor_inputs(1, 1) = 1.0; 
        xor_inputs(2, 0) = 1.0; xor_inputs(2, 1) = 0.0;  
        xor_inputs(3, 0) = 1.0; xor_inputs(3, 1) = 1.0;  
        
        auto xor_predictions = network.predict(xor_inputs);
        
        for (size_t i = 0; i < 4; ++i) {
            T expected = (xor_inputs(i, 0) != xor_inputs(i, 1)) ? T(1) : T(0);
            T predicted = xor_predictions(i, 0);
            cout << "   [" << xor_inputs(i, 0) << ", " << xor_inputs(i, 1) << "]\t"
                      << expected << "\t\t" << predicted << '\n';
        }
        // Imprimir 20 ejemplos de prediccion del test
        cout << "\nEjemplos de prediccion en el test:" << '\n';
        for (size_t i = 0; i < 20; ++i) {
            cout << "   Entrada: [" << X_test(i, 0) << ", " << X_test(i, 1) << "]\t"
                      << "Real: " << y_test(i, 0) << "\tPredicho: " << predictions(i, 0) << '\n';
        }
        
        cout << "\n6. Metricas de rendimiento:" << '\n';
        cout << "   Tiempo total: " << training_time << " segundos" << '\n';
        cout << "   Epocas por segundo: " << T(epochs) / training_time << '\n';
        cout << "   Precision final: " << accuracy << "%" << '\n';
        
        cout << "\n7. Informacion del sistema:" << '\n';
        cout << "   Sistema de tensores: UTEC Tensor System" << '\n';
        cout << "   Optimizador: SGD" << '\n';
        cout << "   Funcion de perdida: Binary Cross-Entropy" << '\n';
        cout << "   Funciones de activacion: ReLU, Sigmoid" << '\n';
        
        cout << "\n=== Entrenamiento completado exitosamente ===" << '\n';
        
    } catch (const exception& e) {
        cerr << "Error durante la ejecucion: " << e.what() << '\n';
        return 1;
    }
    
    return 0;
} 