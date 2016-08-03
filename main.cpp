#include <iostream>
#include <fstream>
#include "Layer.h"
#include "NeuralNetwork.h"
#include "SimpleAutoEncoder.h"
#include "StackedAutoEncoder.h"

using namespace std;

#define MAX_EPOCH 1000000
#define MAX_ERROR 0.001f

#define INPUT_DATA_DIMENSION 2
#define TRAINING_DATA_SIZE 4
#define MLP_LAYER_NUM 2

#define LEARNING_RATE 0.05f

void call_NN();
void call_SimpleAE();
void call_StackedAE();



int main(int argc, char* argv[]) {
//	call_NN();
//	call_SimpleAE();

	call_StackedAE();
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

	printf("\n\nStart Training Neural-Netwrok \n");
	fprintf(myfile,"\n\nStart Training Neural-Netwrok \n");

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


void call_SimpleAE() {
	FILE* myfile = fopen("result.txt", "a");
	float train_data[4][2] = {
		{ 0.f , 0.f },
		{ 0.f , 1.f },
		{ 1.f , 1.f },
		{ 1.f , 0.f }
	};


	int hidden_dim = 5;
	SimpleAutoEncoder myAE(2, hidden_dim);

	printf("\n\nStart Training Simple-Autoencoder \n");
	fprintf(myfile, "\n\nStart Training Simple-Autoencoder \n");

	for (int epoch = 0; epoch < MAX_EPOCH; epoch++) {
		float error = 0.f;

		for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
			myAE.Back_Propagate(train_data[i]);
			error += myAE.Get_Decoding_Error();
		}

		error /= TRAINING_DATA_SIZE;

		myAE.Weight_Update(LEARNING_RATE);

		if ((epoch + 1) % 100 == 0) {
			printf("epoch = %d, error = %f \n", epoch + 1, error);
			fprintf(myfile, "epoch = %d, error = %f \n", epoch + 1, error);

			for (int j = 0; j < TRAINING_DATA_SIZE; j++) {
				myAE.Decoding(train_data[j]);
				float *decode_result = myAE.Get_Decoding_Result();
				printf("test%d: (%f %f) --> (%f %f)\n", j, train_data[j][0], train_data[j][1], decode_result[0], decode_result[1]);
				fprintf(myfile, "test%d: (%f %f) --> (%f %f)\n", j, train_data[j][0], train_data[j][1], decode_result[0], decode_result[1]);
			}
		}

		if (error < MAX_ERROR)
			break;
	}

	printf("Finish Training AutoEncoder  \n\n");
	fprintf(myfile, "Finish Training AutoEncoder  \n\n");
	
	printf("Test AutoEncoder\n");
	fprintf(myfile, "Test AutoEncoder\n");

	for (int j = 0; j < TRAINING_DATA_SIZE; j++) {
		myAE.Decoding(train_data[j]);
		float *encode_result = myAE.Get_Encoding_Result();
		float *decode_result = myAE.Get_Decoding_Result();
		printf("test%d: (%f %f) --> (%f %f %f %f %f --> (%f %f)\n", j, train_data[j][0], train_data[j][1], encode_result[0], encode_result[1], 
													encode_result[2], encode_result[3], encode_result[4], decode_result[0], decode_result[1]);

		fprintf(myfile, "test%d: (%f %f) --> (%f %f %f %f %f --> (%f %f)\n", j, train_data[j][0], train_data[j][1], encode_result[0], encode_result[1],
												encode_result[2], encode_result[3], encode_result[4], decode_result[0], decode_result[1]);
	}

	fclose(myfile);
}



void call_StackedAE(){
	int inputDim = 100; //mnistfeature = 784
	float training_data[][100] = {
	   {1,1,0,0,0,0,1,1,0,0,
		0,1,0,0,0,0,0,0,1,0,
		0,0,1,0,0,1,0,0,0,0,
		0,1,0,0,1,1,1,0,1,0,
		0,1,0,0,1,0,0,1,0,0,
		0,1,0,0,0,0,0,0,0,0,
		0,1,0,0,0,1,0,1,0,1,
		0,0,0,0,0,1,0,0,0,0,
		0,0,0,0,0,0,1,0,0,0,
		0,0,0,0,0,0,1,1,1,1},

	  { 1,1,0,0,0,0,1,1,0,0,
		0,0,0,1,0,0,0,1,1,0,
		0,0,1,0,0,1,0,0,0,0,
		0,1,0,0,1,0,1,1,1,0,
		1,0,0,0,1,0,0,1,0,0,
		0,1,0,0,0,1,0,0,0,0,
		1,1,0,0,0,1,0,1,0,1,
		0,0,0,0,0,0,0,1,0,0,
		0,1,0,0,0,0,1,0,0,0,
		0,1,0,0,0,0,0,1,1,1 },

	  { 1,0,0,1,0,0,1,1,0,0,
		0,1,0,0,0,0,0,0,1,0,
		0,0,1,0,0,1,0,0,1,0,
		0,1,0,0,1,1,1,0,1,0,
		1,0,0,0,1,0,0,1,0,0,
		0,1,0,0,0,1,1,0,1,0,
		0,1,0,1,0,0,0,1,0,1,
		0,0,0,0,0,1,0,0,0,0,
		0,1,0,1,0,0,1,0,0,0,
		0,0,0,0,0,0,1,1,1,1 },

	  { 0,1,0,0,1,0,1,0,1,0,
		0,0,0,1,1,0,0,1,1,0,
		0,0,1,0,0,1,0,0,0,0,
		1,1,0,0,1,0,1,1,1,0,
		1,0,1,0,1,0,0,1,0,0,
		0,1,0,0,0,1,0,0,0,0,
		1,1,1,0,0,1,0,1,0,1,
		0,1,0,0,0,0,1,1,0,0,
		0,0,1,0,1,0,0,0,1,0,
		0,1,0,1,1,0,0,1,0,1 },
	};


	int batch_size = 1;


	int num_AE = 3;
	int eachEncoder_outDim[] = { 50,25,10 };
	StackedAutoEncoder myStackedAE(num_AE, inputDim, eachEncoder_outDim);

	FILE* myfile = fopen("result.txt", "w");


	printf("Start Training Stacked-AutoEncoder \n");
	fprintf(myfile, "Start Training Stacked-AutoEncoder \n");

	for (int idx = 0; idx < num_AE; idx++) {
		int for_breaking = 1;
		for (int epoch = 0; epoch < MAX_EPOCH && for_breaking == 1; epoch++) {
			float error = 0.f;
			for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
				myStackedAE.Back_Propagate(training_data[i], idx);
				error += myStackedAE.Get_Reproduct_Error(idx);
			}
		
			error /= TRAINING_DATA_SIZE;
			if (error < MAX_ERROR) {
				for_breaking = 0;
				break;
			}
			myStackedAE.Weight_Update(LEARNING_RATE, idx);
			


			if ((epoch + 1) % 100== 0) {
				printf("%d-th autoencoder, epoch = %d, error = %f \n", idx + 1, epoch + 1, error);
				fprintf(myfile, "%d-th autoencoder, epoch = %d, error = %f \n", idx + 1, epoch + 1, error);
			}
		}
	}



	printf("Finish Training Stacked-AutoEncoder \n\n");
	fprintf(myfile, "Finish Training Stacked-AutoEncoder \n\n");


	printf("Testing Stacked-AutoEncoder\n");
	fprintf(myfile, "Testing Stacked-AutoEncoder\n");

	for (int number = 0; number < num_AE; number++) {
		fprintf(myfile, "\n\n%d-th autoencoder testing \n", number + 1);
		for (int j = 0; j < TRAINING_DATA_SIZE; j++) {
			myStackedAE.Decoding(training_data[j], number);
			float *encoded = myStackedAE.Get_Encoding_Result(number);
			float *reprod = myStackedAE.Get_Decoding_Result(number);
			float *pinput = myStackedAE.Get_Input(number);
			if (number == 0) {
				for (int k = 0; k < inputDim; k++) {
					fprintf(myfile, "\t(%f, %f)\n", training_data[number][k], reprod[k]);
				}
			}
			else {
				for (int k = 0; k < eachEncoder_outDim[number]; k++) {
					fprintf(myfile, "\t(%f, %f)\n", pinput[k], reprod[k]);
				}
			}
		}
	}
	fclose(myfile);
}