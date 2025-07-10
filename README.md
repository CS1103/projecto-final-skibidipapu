[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## CS2013 Programacion III · Informe Final

### Descripcion

Implementacion completa de una red neuronal multicapa en C++ para clasificacion, usando un sistema de tensores propio (UTEC Tensor System). El objetivo es entender y construir desde cero los fundamentos de las redes neuronales, sin depender de librerías externas, y resolver el problema XOR y otros problemas de clasificación.

---

## Contenidos
1. [Datos generales](#datos-generales)
2. [Requisitos e instalacion](#requisitos-e-instalacion)
3. [Investigacion teorica](#1-investigacion-teorica)
4. [Diseño y arquitectura](#2-diseño-y-arquitectura)
5. [Ejecucion y resultados](#3-ejecucion-y-resultados)
6. [Analisis del rendimiento](#4-analisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones y aprendizajes](#6-conclusiones-y-aprendizajes)
9. [Bibliografia](#7-bibliografia)
10. [Licencia](#licencia)

---

## Datos generales

**Tema:** Redes Neuronales en AI - Implementacion de Multi-Layer Perceptron con Sistema de Tensores UTEC

**Grupo:** `projecto-final-skibidipapu`

**Integrantes**

| Nombre                        | Codigo     |
|-------------------------------|------------|
| Joaquin Ocaña                 | 202410542  |
| Lucia Castro                  | 202410615  | 
| Benjamin Suarez               | 202410391  | 
| Valeria Rios Gomez            | 202410492  |

---

## Requisitos e instalacion

1. **Compilador:** GCC 11+ o compatible con C++17
2. **Dependencias:**
   - CMake 3.18+
   - Sistema de tensores UTEC (incluido)
   - OpenMP (opcional, para paralelizacion)
3. **Instalacion:**
   ```bash
   git clone https://github.com/your-username/projecto-final-skibidipapu.git
   cd projecto-final-skibidipapu
   mkdir build && cd build
   cmake ..
   make -j4
   ```

---

## 1. Investigacion teorica

- **Fundamentos:** Explicacion de redes neuronales, perceptron, multicapa, funciones de activacion, backpropagation.
- **Sistema de Tensores UTEC:** Implementacion propia de tensores N-dimensionales, operaciones matematicas, producto matricial, broadcasting.
- **Arquitectura MLP:** Capas densas, funciones de activacion (ReLU, Sigmoid), funciones de perdida (MSE, Binary Cross-Entropy), optimizadores (SGD, Adam).

---

## 2. Diseño y arquitectura

- **Estructura del proyecto:**
  - `src/`: Codigo principal (`main.cpp`)
  - `include/`: Headers (tensores, capas, activaciones, optimizadores, red)
  - `tests/`: Pruebas unitarias
- **Sistema de tensores:** Permite manipular datos de cualquier dimension, operaciones matematicas y algebra lineal.
- **Red neuronal:** Modular, permite agregar capas, cambiar arquitectura y funciones de activacion facilmente.
- **Paralelizacion:** Uso de OpenMP para acelerar la multiplicacion de matrices en CPUs multinucleo.
- **Patrones de diseño:** Strategy (optimizadores), Factory (inicializacion de pesos), Template (activacion y perdida).

---

## 3. Ejecucion y resultados

- **Demo principal:** Resolucion del problema XOR con datos generados artificialmente (ruido incluido).
- **Configuracion:**
  - 25,000 muestras de entrenamiento, 1,000 de test
  - 500 epocas, batch size 128, learning rate 0.01
  - Arquitectura: 2 -> 64 -> 64 -> 1 (ReLU, Sigmoid)
- **Entrenamiento:**
  - Por batches reales, monitoreo de loss y tiempo cada 50 epocas
  - Precision final: 100% en XOR
- **Salida esperada:**
  ```
  === UTEC Neural Network Project ===
  1. Generando datos de entrenamiento (problema XOR)...
  2. Configurando arquitectura de red neuronal...
  3. Iniciando entrenamiento...
  4. Evaluando rendimiento...
  5. Demostracion XOR...
  6. Metricas de rendimiento...
  ```
- **Ejemplo de prediccion:**
  - Entrada: [0, 1] → Salida esperada: 1 → Salida predicha: ~1
  - Entrada: [1, 1] → Salida esperada: 0 → Salida predicha: ~0

---

## 4. Analisis del rendimiento

- **Metricas:**
  - Tiempo total de entrenamiento (con y sin OpenMP)
  - Precision final y loss promedio
  - Escalabilidad con diferentes tamaños de batch y muestras
- **Ventajas:**
  - Codigo ligero, sin dependencias externas
  - Modularidad y facilidad de extension
  - Paralelizacion efectiva con OpenMP
- **Limitaciones:**
  - Sin soporte GPU
  - Sin datasets reales grandes
  - Sin interfaz grafica
- **Mejoras futuras:**
  - BLAS para multiplicaciones
  - Soporte GPU (CUDA/OpenCL)
  - Nuevos optimizadores y funciones de activacion
  - Soporte para datasets reales (MNIST, CIFAR)

---

## 5. Conclusiones y aprendizajes

- Logramos implementar una red neuronal funcional y eficiente desde cero.
- El sistema de tensores propio nos permitió entender a fondo la base matemática de la IA.
- Aprendimos sobre optimización, paralelización y trabajo colaborativo en C++.
- El proyecto es base para futuras extensiones y experimentos en IA.

---

## 6. Bibliografia

1. I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.
2. M. Nielsen, *Neural Networks and Deep Learning*. Determination Press, 2015.
3. S. Haykin, *Neural Networks and Learning Machines*, 3rd ed. Pearson, 2009.
4. G. Hinton, "Learning representations by back-propagating errors," *Nature*, vol. 323, no. 6088, pp. 533–536, 1986.
5. D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning representations by back-propagating errors," *Cognitive Modeling*, vol. 1, pp. 1–10, 1988.

---
## 7. Video del proyecto
Link del video almacenado en indrive:
https://drive.google.com/file/d/10Z8ZWf4krCCsGhhK1j_vDYpwRHJxta2C/view?usp=sharing

## Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

## Fundamentos de IA y entrenamiento

### ¿Qué es una época?
Una **época** es una pasada completa por todo el conjunto de datos de entrenamiento. En cada época, la red neuronal ve todos los ejemplos y ajusta sus pesos para minimizar el error.

### ¿Qué es un batch?
Un **batch** es un pequeño subconjunto de los datos de entrenamiento. En vez de actualizar los pesos después de ver todos los datos, se actualizan después de cada batch, lo que acelera y estabiliza el aprendizaje.

### ¿Por qué usamos el problema XOR?
El **problema XOR** es un clásico en IA porque no puede resolverse con un perceptrón simple (una sola capa). Requiere una red multicapa, lo que lo convierte en un excelente ejemplo para probar la capacidad de aprendizaje de una red neuronal.

### ¿Cómo se generan los datos?
Los datos se generan artificialmente, creando combinaciones de 0 y 1 (con algo de ruido) para simular entradas y salidas del problema XOR. Esto permite entrenar y evaluar la red de manera controlada.

### ¿Qué es el loss y la precisión?
- **Loss (función de pérdida):** Mide qué tan lejos están las predicciones de la red respecto a los valores reales. Un loss bajo indica buen aprendizaje.
- **Precisión:** Es el porcentaje de predicciones correctas sobre el total de ejemplos de test. Una precisión del 100% significa que la red clasificó correctamente todos los ejemplos.

### ¿Cómo se interpreta el avance del entrenamiento?
Durante el entrenamiento, se muestra el loss promedio y el tiempo transcurrido cada cierto número de épocas. Si el loss baja y la precisión sube, la red está aprendiendo correctamente.
