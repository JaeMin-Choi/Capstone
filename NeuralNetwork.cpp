
#include <iostream>

#include "Layer.h"
#include "NeuralNetwork.h"
using namespace std;


NeuralNetwork::NeuralNetwork() {
	numberOfLayer = 0;
	aLayer = NULL;
}

NeuralNetwork::NeuralNetwork(int inputDim, int numLayer, int* eachLayer_outputDim) {
	numberOfLayer = 0;
	aLayer = NULL;

	Init(inputDim, numLayer, eachLayer_outputDim);
}


void NeuralNetwork::Delete() {
	if (aLayer != NULL) {
		delete[] aLayer;
		aLayer = NULL;
	}
	numberOfLayer = 0;
}

void NeuralNetwork::Init(int inputDim, int numLayer, int* eachLayer_outputDim) {
	if (Is_Inited())
		Delete();

	numberOfLayer = numLayer;

	try {
		aLayer = new Layer[numberOfLayer];
	}
	catch (bad_alloc& ba) {
		printf("bad memory allocation in func : %s, file : %s, line number : %d\n", __FUNCTION__, __FILE__, __LINE__);
		exit(-1);
	}

	aLayer[0].Init(inputDim, eachLayer_outputDim[0]);

	for (int i = 1; i < numberOfLayer; i++)
		aLayer[i].Init(eachLayer_outputDim[i - 1], eachLayer_outputDim[i]);

}

//Propagate하는 함수. Layer의 처음에는 사용자로부터 받은 input을 넘겨주고, 그 다음부터는 그 전 layer의 output을 input으로 줌.
void NeuralNetwork::Propagate(float* input) {
	aLayer[0].Propagate(input);
	for (int i = 1; i < numberOfLayer; i++)
		aLayer[i].Propagate(aLayer[i - 1].Get_Output());

}

//back propagate 하는 것. 하고 나면 gradient가 계산된다
void NeuralNetwork::Back_Propagate(float* input, float* desiredOutput) {
	Propagate(input);

	for (int i = numberOfLayer - 1; i >= 0; i--) {
		// i 가 현재 최상단 layer를 가리키고 있다면
		if (i == numberOfLayer - 1)
			aLayer[i].Compute_Top_DeltaBar(desiredOutput);
		// i 가 현재 최상단 layer가 아니라면, 윗 레이어에게 해당 layer의 deltabar를 구하도록 부탁해야함.
		else
			aLayer[i + 1].Compute_PrevDeltaBar(aLayer[i].Get_DeltaBar());
		//위 if-else문은 deltabar를 구하는 과정임. 

		//deltabar를 구했다면, gradient를 구한다. 아래 Compute_Gradient를 호출하면 deltabar로부터 delta를 구하고, delta로부터 gradient를 구한다.
		aLayer[i].Compute_Gradient();
	}
}

void NeuralNetwork::Weight_Update(float learningRate) {
	for (int i = 0; i < numberOfLayer; i++)
		aLayer[i].Weight_Update(learningRate);
}
