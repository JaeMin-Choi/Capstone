#include <iostream>
#include <fstream>
#include "Layer.h"
#include "NeuralNetwork.h"
using namespace std;

#define MAX_EPOCH 1000000
#define MAX_ERROR 0.001f

#define INPUT_DATA_DIMENSION 2
#define TRAINING_DATA_SIZE 4
#define MLP_LAYER_NUM 2

#define LEARNING_RATE 0.05f
void call_NN();


int main(int argc, char* argv[]) {
	call_NN();

	return 0;
}

void call_NN(){
	FILE* myfile = fopen("result.txt", "w");

	int aNet_OutputDim[] = { 4,1 };
	float train_data[4][2] = {
		{ 0.f , 0.f },
		{ 0.f , 1.f },
		{ 1.f , 1.f },
		{ 1.f , 0.f }
	};
	float desired_output[] = { 0.f, 1.f, 1.f, 0.f };


	NeuralNetwork myNetwork;
	myNetwork.Init(INPUT_DATA_DIMENSION, MLP_LAYER_NUM, aNet_OutputDim);

	printf("Start Neural-Netwrok Training \n");
	fprintf(myfile,"Start Neural-Netwrok Training \n");

	for (int epoch = 0; epoch < MAX_EPOCH; epoch++) {
		float error = 0.f;

		for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
			myNetwork.Back_Propagate(train_data[i], &desired_output[i]);
			error += myNetwork.Get_Error(&desired_output[i]);
		}
		error /= TRAINING_DATA_SIZE;

		myNetwork.Weight_Update(LEARNING_RATE);
		if ((epoch + 1) % 100 == 0) {
			printf("epoch = %d, error = %f \n", epoch + 1, error);
			fprintf(myfile,"epoch = %d, error = %f \n", epoch + 1, error);
		}

		if (error < MAX_ERROR)
			break;
	}

	printf("Finish Neural-Network Training \n\n");
	fprintf(myfile,"Finish Neural-Network Training \n\n");

	printf("Test Neural-Network \n");
	fprintf(myfile,"Test Neural-Network \n");

	for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
		myNetwork.Propagate(train_data[i]);
		float* pOutput = myNetwork.Get_Output();
		float* pHidden = myNetwork[0].Get_Output();

		printf("test%d: (%f %f) --> (%f %f %f %f) --> %f\n", i, train_data[i][0], train_data[i][1], pHidden[0], pHidden[1], pHidden[2], pHidden[3], pOutput[0]);
		fprintf(myfile, "test%d: (%f %f) --> (%f %f %f %f) --> %f\n", i, train_data[i][0], train_data[i][1], pHidden[0], pHidden[1], pHidden[2], pHidden[3], pOutput[0]);
	}

	fclose(myfile);
}
