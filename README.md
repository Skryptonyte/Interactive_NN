# Interactive Neural Net

This is a small commandline driven C program that demonstrates the working of a simple neural network as a proof of concept. This neural network uses sigmoid activation functions and a simple mean square loss function.

The program uses matrix operations powered by GNU Scientific Library and its built in BLAS routines making it very performant.


## Requirements

* Latest version of GNU Scientific Library
* Recent version of GCC/Clang/MSVC

## Compiling

``` gcc neuralnet.c -lgsl -lcblas -lm -o neuralnet.out ```
