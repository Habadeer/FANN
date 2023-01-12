#pragma once

#include <vector>
#include <stdint.h>
#include <iostream>

namespace AI
{
    struct LearningData
    {
        std::vector<double> input;
        std::vector<double> desiredOutput;
    };

    struct FunctionParams
    {
        FunctionParams(double _a = 0, double _b = 0, double _c = 0) : a(_a), b(_b), c(_c) {}
        double a;
        double b;
        double c;
    };

    enum class EActivationFunction : short
    {
        UNDEFINED = 0,
        LINEAR,
        UNISIGMOID,
        BISIGMOID,
        GAUSS
    };

    enum class EFunctionDerivative : short
    {
        FUNCTION = 0,
        FUNCTION_DERIVATIVE
    };

    class ArtificialNeuralNetwork
    {
    private:
        class NeuronLayer;
        std::vector<uint64_t> v_NetworkShape;
        std::vector<NeuronLayer> v_NeuronLayers;
        double learningSpeed;
        EActivationFunction ActivationFunctionSelection;
        FunctionParams par;

        double calibrateWeightsSeries(std::vector<double> inputs, std::vector<double> desiredOutputs);

        class NeuronLayer
        {
        private:
            class Neuron;
            ArtificialNeuralNetwork &network;
            std::vector<Neuron> v_Neurons;

            class Neuron
            {
            private:
                ArtificialNeuralNetwork &network;
                std::vector<double> v_Weights;
                double m_NeuronError;

            public:
                Neuron(uint32_t numberOfInputs, ArtificialNeuralNetwork &net);
                double neuronImpulse(const std::vector<double> &inputs);

                double getWeight(uint32_t weightNumber);
                void setNeuronError(double error);
                void calibrateNeuronWeight(std::vector<double> inputs);
                double functionOutput(EFunctionDerivative select, double x);
            };

        public:
            NeuronLayer(uint32_t numberOfNeurons, uint32_t numberOfNeuronsInPreviousLayer, ArtificialNeuralNetwork &net);
            std::vector<double> layerImpulse(const std::vector<double> &previousLayerInput);
            void calibrateLayerWeights(std::vector<double> inputs);
            std::vector<double> calculatePreviousLayerError(std::vector<double> layerError, int numberOfNeuronsInPreviousLayer);
            void setLayerError(std::vector<double> layerError);
        };

    public:
        ArtificialNeuralNetwork(std::vector<uint64_t> networkShape, FunctionParams functionParameters, EActivationFunction function, double lspeed);

        std::vector<double> networkImpulse(std::vector<double> inputs);

        void calibrateNetworkWeights(std::vector<LearningData> learningSeries, uint64_t maxEpochNumber, double maxError);
    };

}