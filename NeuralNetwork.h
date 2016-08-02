
class NeuralNetwork {
private:
	int numberOfLayer;
	Layer* aLayer;

public:
	NeuralNetwork();
	NeuralNetwork(int inputDim, int numLayer, int* eachLayer_outputDim);
	~NeuralNetwork() { Delete(); }
	void Delete();

	void Init(int inputDim, int numLayer, int* eachLayer_outputDim);
	int Is_Inited() { return aLayer != NULL; } //aLayer가 할당되어 있다면, return 1, 할당되어있지 않다면(==NULL) return 0

	void Propagate(float* input);

	void Back_Propagate(float* input, float* desiredOutput);

	void Weight_Update(float learningRate);

	Layer& operator[] (int idx) { return aLayer[idx]; }

	float Get_Error(float* desired_output) { return aLayer[numberOfLayer - 1].Compute_Error(desired_output); }

	float* Get_Output() { return aLayer[numberOfLayer - 1].Get_Output(); }
};

