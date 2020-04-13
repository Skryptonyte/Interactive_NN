#include <math.h>
#include <gsl/gsl_vector.h>


#define SIGMOID 0
#define RELU 1
#define TANH 2

#define ACTIVATION_TABLE_SIZE 3

// Derivatives of activation functions

double sigmoid_derivative(double x){
	double t = exp(x);
	double y = t/(1+t);

	return y*(1-y);
} 

double RELU_derivative(double x){
	if (x <= 0.0){
		return 0.0;
	}
	else{
		return 1.0;
	}
}

double tanh_derivative(double x){
	double t= 1/cosh(x);
	return t*t;
}

// Activation functions ( for vectors )


void sigmoid_V(gsl_vector* a){
        for (int i = 0; i < a->size; i++){
                double x = exp(gsl_vector_get(a,i));
                gsl_vector_set(a,i,x/(1.0+x));
        }
}


void RELU_V(gsl_vector* a){
        for (int i = 0; i < a->size; i++){
                double x = fmax(x,0.0);
        }
}

void tanh_V(gsl_vector* a){
	for (int i =0; i < a->size; i++){
		gsl_vector_set(a,i,tanh(gsl_vector_get(a,i)));
	}	
}


// Table of functions

void (*activation_V[ACTIVATION_TABLE_SIZE])(gsl_vector* ) = {sigmoid_V, RELU_V, tanh_V};
double (*activation_derivatives[ACTIVATION_TABLE_SIZE])(double) = {sigmoid_derivative, RELU_derivative, tanh_derivative};
