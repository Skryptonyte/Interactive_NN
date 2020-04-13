#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "neuralnet_activations.h"

int activation;

// Helper functions


void clear_buffer(FILE* f){
	 char ch = getc(f);
                while (!ch){
                        ch = getc(f);
                }

}




gsl_vector* forwardPropagate(gsl_vector* inputVector, gsl_matrix** weightPtrs, gsl_vector** a, gsl_vector** z,int* nodeCountArray, int layers);
double backPropagate(gsl_vector* inputVector, gsl_vector* outputVector, gsl_matrix** weightPtrs, gsl_vector** a,gsl_vector** z,int* nodeCountArray, double learningRate,int regularizationMask,double alpha,double beta,int layer);

int main(){

srand(time(0));

int nodeCountArray[100];
gsl_matrix* weightPtrs[100];
gsl_vector* a[100];
gsl_vector* z[100];
int layerCount = 0;
puts("AI Menu");

puts("-------------------------------");
puts("1. Construct/Reconstruct Neural Network");
puts("2. Test Data");
puts("3. Train Data");
puts("4. Save Data");
puts("5. Load Data");
puts("-------------------------------\n");

int option = -1;
int created = 0;
while (1){
	printf("Enter option: ");
	scanf("%d",&option);
	if (option == 1){   //Create neural net
		if (created){
			puts("Cleaning up old network..");
			for (int i = 1; i <= layerCount; i++){
				gsl_matrix_free(weightPtrs[i]);
				gsl_vector_free(a[i]);
				gsl_vector_free(z[i]);
			}
		}

		int flag = 0;
		memset(nodeCountArray, 0, sizeof(int) * 100);
		memset(weightPtrs, 0, sizeof(gsl_matrix*) * 100);
		memset(a, 0, sizeof(gsl_vector*) * 100);
		memset(z, 0, sizeof(gsl_vector*) * 100);
		
		puts("Activation functions: ");
		puts("1. Sigmoid");
		puts("2. RELU");
		puts("3. tanh");
		
		while (1){
		printf("Enter activation function: ");		
		scanf("%d", &activation);
		activation--;
			if (activation < 0 || activation >= ACTIVATION_TABLE_SIZE){
				puts("Invalid activation function!");
			}
			else{
				break;
			}
		}
		while (!flag){
		printf("Input number of nodes: ");
		scanf("%d",nodeCountArray);
			if (nodeCountArray[0] <= 0){
				puts("Value must be non-zero and positive!");
			}
			else{
				flag ^= 1;
			}
		}
		layerCount = 0;	
		while (1){
			int input = 0;
			printf("Do you want to add a layer? (1/0): ");
			scanf("%d",&input);
			if (input == 1){
				layerCount++;
				int nodeCount = 0;
				int flag = 0;
	
				while (!flag){
					printf("Enter number of nodes: ");
					scanf("%d",&nodeCount);
					if (nodeCount <= 0){
						puts("Value must be non-zero and positive!");
					}
					else{
						flag ^= 1;
					}
				}
				weightPtrs[layerCount] = gsl_matrix_calloc(nodeCount, nodeCountArray[layerCount - 1]);
				if (!weightPtrs[layerCount]){
					puts("MALLOC ERROR");
					exit(0);
				}
				for (int i = 0; i < nodeCount; i++){
					for (int j = 0; j <nodeCountArray[layerCount-1];j++){
						//printf("%d %d\n",i,j);
						double t = (double) rand()/RAND_MAX;
						gsl_matrix_set(weightPtrs[layerCount], i, j,2.0*(t-0.5));
						//printf("%lf\n",gsl_matrix_get(weightPtrs[layerCount],i,j));
					}
				}
				nodeCountArray[layerCount] = nodeCount;
		}
			else{
				break;
			}
		}
		created = 1;
		
	}
	else if (option == 2){
		if (layerCount == 0){
			puts("Neural Network Uninitialized");
			continue;
		}
		gsl_vector* inputVector = gsl_vector_calloc(nodeCountArray[0]);
		if (!inputVector){
			puts("MALLOC ERROR!");
			exit(1);
		}
		int count = 0;
		puts("Enter Test Data: ");
		while (count < nodeCountArray[0]){
			double input;
			scanf("%lf", &input);
			gsl_vector_set(inputVector, count,input);
			count++;
		}
		gsl_vector* v = forwardPropagate(inputVector, weightPtrs,a, z, nodeCountArray, layerCount);
		gsl_vector_free(v);
		for (int i = 1; i <= layerCount; i++){
			gsl_vector_free(a[i]);
			gsl_vector_free(z[i]);
		}
	}
	else if (option == 3){
		double learningRate;
		int epoch;
		int batchCount;
		int regularization;
		
		double alpha, beta;
		if(layerCount == 0){
			puts("Neural Network Uninitialized");
			continue;
		}

		
		gsl_vector* inputVector = gsl_vector_calloc(nodeCountArray[0]);
		gsl_vector* outputTrainVector = gsl_vector_calloc(nodeCountArray[layerCount]);
		
		gsl_vector** batchInput;
		gsl_vector** batchExpected;

		printf("Enter number of batches: ");
		scanf("%d",&batchCount);
		
		batchInput = malloc(sizeof(gsl_vector*) * batchCount);
		batchExpected = malloc(sizeof(gsl_vector*)* batchCount);
		
		// Recieve Batch Data
		for (int i = 0; i < batchCount; i++){
		printf("Batch %d: \n",i+1);
		puts("Input Training Data: ");
		int count = 0;
		batchInput[i] = gsl_vector_calloc(nodeCountArray[0]);
		while (count < nodeCountArray[0]){
                        double input;
                        scanf("%lf", &input);
                        gsl_vector_set(batchInput[i], count,input);
                        count++;
                }
		count = 0;
		batchExpected[i] = gsl_vector_calloc(nodeCountArray[layerCount]);
		puts("Input Expected Data: ");
		while (count < nodeCountArray[layerCount]){
                        double input;
                        scanf("%lf", &input);
                        gsl_vector_set(batchExpected[i], count,input);
                        count++;
                }
		}

		printf("Enter learning rate: ");
		scanf("%lf",&learningRate);
		
		printf("Enter epoch count: ");
		scanf("%d",&epoch);

		printf("Apply regularization? (0: None, 1: L1, 2: L2, 3: Both): ");
		scanf("%d",&regularization);
		
		if (regularization & 1){
			printf("Set L1 regularization constant: ");
			scanf("%lf", &alpha);
		}else{
			alpha = 0.0;
		}
		
		if (regularization & 2){
			printf("Set L2 regularization constant: ");
			scanf("%lf", &beta);
		}else{
			beta = 0.0;
		}

		int i = 1;
		while (i <= epoch){
			for (int j = 0; j < batchCount; j++){
				gsl_vector* outputVector = forwardPropagate(batchInput[j], weightPtrs, a, z, nodeCountArray, layerCount);
				double MSE = backPropagate(outputVector, batchExpected[j],weightPtrs, a, z,nodeCountArray, learningRate,regularization,alpha,beta,layerCount);
				printf("EPOCH: %d,BATCH: %d, Mean Squared Error: %lf\n",i,j+1,MSE);
				gsl_vector_free(outputVector);
			}
		i++;
		}
	}
	else if (option == 4){
		if (layerCount == 0){
			puts("Uninitialized neural network");
			continue;
		}

		char fileName[256];
		clear_buffer(stdin);

		printf("Enter file name: ");
		fgets(fileName,256,stdin);
	
		char* finalName = strtok(fileName,"\n");
		FILE* FH = fopen(finalName,"wb");

		// Clear STDIN for proper function of scanf 
		// Prepend magic number to file
		
		fputc('I',FH);
		fputc('N', FH);
		fputc('N', FH);
		fputc('_',FH);
			
		puts("Writing to file..");
		fwrite(&layerCount, sizeof(int), 1, FH);
		fwrite(&activation, sizeof(int), 1, FH);
		fwrite(nodeCountArray, sizeof(int), 100, FH);
		
		int layers = 0;
		for (int i = 1; i <= layerCount; i++){
			gsl_matrix_fwrite(FH, weightPtrs[i]);
		}
		fclose(FH);
	}
	else if (option == 5){
		
		char fileName[256];
                clear_buffer(stdin);

                printf("Enter file name: ");
                fgets(fileName,256,stdin);
  
                char* finalName = strtok(fileName,"\n");
                FILE* FH = fopen(finalName,"rb");
	
		if (!FH){
			puts("File does not exist");
			continue;
		}
		
		char magic[4] = {fgetc(FH), fgetc(FH), fgetc(FH), fgetc(FH)};
		
		if (magic[0] != 'I' || magic[1] != 'N' || magic[2] != 'N' || magic[3] != '_'){
			puts("Incorrect magic number");
			continue;
		}
		
		int prevLayerCount = layerCount;
		fread(&layerCount, sizeof(int), 1, FH);
		fread(&activation, sizeof(int), 1, FH);
		fread(nodeCountArray,sizeof(int),100,FH);
		created = 1;	

		for (int i = 1; i <= prevLayerCount; i++){
			if (weightPtrs[i]){
				gsl_matrix_free(weightPtrs[i]);
				weightPtrs[i] = NULL;
			}
		}
		
		for (int i = 1; i <= layerCount; i++){
			weightPtrs[i] = gsl_matrix_calloc(nodeCountArray[i], nodeCountArray[i-1]);
		}
	
		for (int i = 1; i <= layerCount; i++){
			gsl_matrix_fread(FH, weightPtrs[i]);
		}		
		fclose(FH); 
	}
}
return 0;
}

gsl_vector* forwardPropagate(gsl_vector* inputVector, gsl_matrix** weightPtrs, gsl_vector** a, gsl_vector** z, int* nodeCountArray, int layers){
	gsl_vector* v1 = NULL;
	gsl_vector* v2 = gsl_vector_calloc(nodeCountArray[0]);
	gsl_vector_memcpy(v2, inputVector);

	z[0] = gsl_vector_calloc(nodeCountArray[0]);
	gsl_vector_memcpy(z[0], inputVector);
	// Foward Propagate via Matrix Vector Multiplication using GSL BLAS
	for (int i = 1; nodeCountArray[i] != 0; i++){

		v1 = gsl_vector_calloc(nodeCountArray[i]);	
		gsl_blas_dgemv(CblasNoTrans,  1.0, weightPtrs[i], v2, 0.0, v1);
			
		a[i] = gsl_vector_calloc(nodeCountArray[i]);
                gsl_vector_memcpy(a[i], v1);
		
		activation_V[activation](v1);

		z[i] = gsl_vector_calloc(nodeCountArray[i]);
		gsl_vector_memcpy(z[i], v1);

		gsl_vector_free(v2);
		v2 = gsl_vector_calloc(nodeCountArray[i]);

		gsl_vector_memcpy(v2, v1);
		gsl_vector_free(v1);
	}
			
	printf("Output: [");
	
	// Output value of last layer of neurons
	for (int i = 0; i < nodeCountArray[layers]; i++){
		printf("%lf ",gsl_vector_get(v2,i));
	}
	
	
	printf("]\n");
	return v2;
}

double backPropagate(gsl_vector* inputVector, gsl_vector* outputVector ,gsl_matrix** weightPtrs, gsl_vector** a,gsl_vector** z,int* nodeCountArray,double learningRate, int regularizationMask, double alpha ,double beta,int layer){
// Backpropagate first layer

	gsl_vector *derivLayer = gsl_vector_calloc(nodeCountArray[layer-1]);
	gsl_vector* prevDerivLayer = gsl_vector_calloc(nodeCountArray[layer]);
	
	gsl_matrix* offsetMatrix[layer+1];
	 
	gsl_vector* differenceLayer = gsl_vector_calloc(nodeCountArray[layer]);
	gsl_vector_memcpy(differenceLayer, inputVector);
	gsl_vector_sub(differenceLayer, outputVector);
	
	double MSE = 0.0;
	for (int i = 0; i < nodeCountArray[layer]; i++){
		double t = gsl_vector_get(differenceLayer,i);
		MSE += t*t; 
	}	
	MSE /= nodeCountArray[layer];
	for (int i = 0; i < nodeCountArray[layer]; i++){
		double t = activation_derivatives[activation](gsl_vector_get(a[layer], i)) * (gsl_vector_get(differenceLayer,i)); 
		gsl_vector_set(prevDerivLayer, i,t);
	}

	offsetMatrix[layer] = gsl_matrix_calloc(nodeCountArray[layer],  nodeCountArray[layer-1]);
	gsl_blas_dger(learningRate, prevDerivLayer,z[layer-1], offsetMatrix[layer]);
	for (int l = layer-1; l >= 1; l--){	
		for (int i = 0; i < nodeCountArray[l]; i++){
			double k = 0.0;
			for (int j = 0; j < nodeCountArray[l+1]; j++){
				k += gsl_vector_get(prevDerivLayer, j) * activation_derivatives[activation](gsl_vector_get(a[l],i)) * gsl_matrix_get(weightPtrs[l+1],j,i);
			}
			gsl_vector_set(derivLayer,i,k);
		}
	
		// Create offset Matrix
		offsetMatrix[l] = gsl_matrix_calloc(nodeCountArray[l],  nodeCountArray[l-1]);
		gsl_blas_dger(learningRate, derivLayer, z[l-1], offsetMatrix[l]);

		gsl_vector_free(prevDerivLayer);
		prevDerivLayer = gsl_vector_calloc(nodeCountArray[l]);

		gsl_vector_memcpy(prevDerivLayer, derivLayer);

		gsl_vector_free(derivLayer);
		derivLayer = gsl_vector_calloc(nodeCountArray[l-1]);
	}

	gsl_vector_free(prevDerivLayer);
	gsl_vector_free(derivLayer);
	gsl_vector_free(differenceLayer);

	gsl_vector_free(z[0]);
		
	// Compute new weights	
	
	for (int l = 1; l <= layer; l++){
		if (regularizationMask & 1){
			puts("L1");
			gsl_matrix* L1 = gsl_matrix_calloc(weightPtrs[l]->size1, weightPtrs[l]->size2);
			gsl_matrix_set_all(L1,alpha);
		
			gsl_matrix_sub(weightPtrs[l], L1);
			gsl_matrix_free(L1);
		}
		if (regularizationMask & 2){
			puts("L2");
			gsl_matrix* L2 = gsl_matrix_calloc(weightPtrs[l]->size1, weightPtrs[l]->size2);
			gsl_matrix_memcpy(L2, weightPtrs[l]);
			gsl_matrix_scale(L2, beta);
	
			gsl_matrix_sub(weightPtrs[l], L2);
			gsl_matrix_free(L2);
		}
		gsl_matrix_sub(weightPtrs[l],offsetMatrix[l]);
		gsl_matrix_free(offsetMatrix[l]);
		gsl_vector_free(a[l]);
		gsl_vector_free(z[l]);
	}

	return MSE;
}	
