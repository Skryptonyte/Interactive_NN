# Interactive Neural Net

This is a small C program that demonstrates the working of a simple network as a proof of concept. The program is powered by GNU Scientific Library and its built in BLAS routines making it very performant.


## Requirements

* Latest version of GNU Scientific Library
* Recent version of GCC/Clang/MSVC

## Compiling

``` gcc neuralnet.c -lgsl -lblas -lm -o neuralnet.out ```
