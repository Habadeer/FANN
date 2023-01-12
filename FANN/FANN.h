#ifndef ARTIFICIALNEURALNETWORK_H
#define ARTIFICIALNEURALNETWORK_H

#include <vector>
#include <stdint.h>
#include <iostream>

struct LearningData
{
    std::vector<double> input;
    std::vector<double> desiredOutput;
};

struct FunctionParams
{
    double a;
    double b;
    double c;
};

enum EActivationFunction
{
    UNDEFINED = 0,
    LINEAR = 1,
    UNISIGMOID = 2,
    BISIGMOID = 3,
    GAUSS = 4
};

enum EFunctionDerivative
{
    FUNCTION = 0,
    FUNCTION_DERIVATIVE = 1
};

class ArtificialNeuralNetwork
{
public:
    class NeuronLayer
    {
    public:
        class Neuron
        {
        public:
            ArtificialNeuralNetwork* ANN;
            std::vector<double> v_Weights;
            double m_NeuronError;

            Neuron(uint32_t numberOfInputs, ArtificialNeuralNetwork* networkPtr);
            double neuronImpulse(const std::vector<double>& inputs);
            
            double getWeight(uint32_t weightNumber);
            void setNeuronError(double error);
            void calibrateNeuronWeight(std::vector<double> inputs);
            double functionOutput(EFunctionDerivative select, double x);
        };
        ArtificialNeuralNetwork* ANN;
        std::vector<Neuron> v_Neurons;
        NeuronLayer(uint32_t numberOfNeurons, uint32_t numberOfNeuronsInPreviousLayer, ArtificialNeuralNetwork* networkPtr);
        std::vector<double> layerImpulse(const std::vector<double>& previousLayerInput);
        std::vector<double> calculatePreviousLayerError(std::vector<double> layerError, int numberOfNeuronsInPreviousLayer);
        void setLayerError(std::vector<double> layerError);
        void calibrateLayerWeights(std::vector<double> inputs);


    };
    std::vector<uint64_t> v_NetworkShape;
    std::vector<NeuronLayer> v_NeuronLayers;
    double learningSpeed;
    EActivationFunction ActivationFunctionSelection;
    FunctionParams par;
    ArtificialNeuralNetwork(std::vector<uint64_t> networkShape, FunctionParams functionParameters, EActivationFunction function = BISIGMOID, double lspeed = 0.001);


    
    //static double getLearningSpeed();

    std::vector<double> networkImpulse(std::vector<double> inputs);

    void calibrateNetworkWeights(std::vector<LearningData> learningSeries, uint64_t maxEpochNumber, double maxError);
    double calibrateWeightsSeries(std::vector<double> inputs, std::vector<double> desiredOutputs);


};

#endif // ARTIFICIALNEURALNETWORK_H
