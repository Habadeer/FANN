#include "FANN.h"

#include <fstream>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <cmath>

namespace AI
{
    ArtificialNeuralNetwork::ArtificialNeuralNetwork(std::vector<uint64_t> networkShape, FunctionParams functionParameters, EActivationFunction function, double lspeed) : learningSpeed(lspeed), ActivationFunctionSelection(function), par(functionParameters), v_NetworkShape(networkShape)
    {
        for (uint16_t i = 1; i < networkShape.size(); i++)
        {
            v_NeuronLayers.push_back(NeuronLayer(networkShape.at(i), networkShape.at(i - 1), *this));
        }
    }

    ArtificialNeuralNetwork::NeuronLayer::NeuronLayer(uint32_t numberOfNeurons, uint32_t numberOfNeuronsInPreviousLayer, ArtificialNeuralNetwork &net) : network(net)
    {
        for (uint16_t i = 0; i < numberOfNeurons; i++)
        {
            v_Neurons.push_back(Neuron(numberOfNeuronsInPreviousLayer, network));
        }
    }

    ArtificialNeuralNetwork::NeuronLayer::Neuron::Neuron(uint32_t numberOfInputs, ArtificialNeuralNetwork &net) : network(net)
    {
        // Each of neuron weights is initialised with value between -1 and 1
        // Number of inputs is increased by 1 for including bias.
        for (uint32_t i = 0; i < numberOfInputs + 1; i++)
        {
            v_Weights.push_back(double(std::rand()) / RAND_MAX * 2 - 1);
        }
    }

    std::vector<double> ArtificialNeuralNetwork::networkImpulse(std::vector<double> inputs)
    {

        if (inputs.size() != v_NetworkShape.at(0))
        {
            std::cout << inputs.size() << " " << v_NetworkShape.at(0) << std::endl;
            return inputs;
        }
        std::vector<double> layerIO = inputs;
        for (uint16_t i = 0; i < v_NetworkShape.size() - 1; i++)
        {
            layerIO = v_NeuronLayers.at(i).layerImpulse(layerIO);
        }
        return layerIO;
    }

    std::vector<double> ArtificialNeuralNetwork::NeuronLayer::layerImpulse(const std::vector<double> &previousLayerInput)
    {
        std::vector<double> layerOutputs;
        for (uint16_t i = 0; i < v_Neurons.size(); i++)
        {
            layerOutputs.push_back(v_Neurons.at(i).neuronImpulse(previousLayerInput));
        }
        return layerOutputs;
    }

    double ArtificialNeuralNetwork::NeuronLayer::Neuron::neuronImpulse(const std::vector<double> &inputs)
    {
        double inputXWeightsSum = 0;

        for (uint16_t i = 0; i < inputs.size(); i++)
        {
            inputXWeightsSum += inputs.at(i) * v_Weights.at(i);
        }
        inputXWeightsSum += v_Weights.back();
        return functionOutput(EFunctionDerivative::FUNCTION, inputXWeightsSum);
    }

    double ArtificialNeuralNetwork::NeuronLayer::Neuron::functionOutput(EFunctionDerivative select, double input)
    {
        switch (network.ActivationFunctionSelection)
        {
        case EActivationFunction::LINEAR:
            if (select == EFunctionDerivative::FUNCTION)
            {
                return network.par.a * input + network.par.b;
            }
            return network.par.a;
        case EActivationFunction::UNISIGMOID:
            if (select == EFunctionDerivative::FUNCTION)
            {
                return 1 / (1 + exp(-network.par.a * input));
            }
            return (network.par.a * exp(-network.par.a * input)) / (pow(exp(-network.par.a * input) + 1, 2));
        case EActivationFunction::BISIGMOID:
            if (select == EFunctionDerivative::FUNCTION)
            {
                return (1 - exp(-network.par.a * input)) / (1 + exp(-network.par.a * input));
            }
            return (2 * network.par.a * exp(network.par.a * input) / pow(exp(network.par.a * input) + 1, 2));
        case EActivationFunction::GAUSS:
            if (select == EFunctionDerivative::FUNCTION)
            {
                return network.par.a * exp(-pow(input - network.par.b, 2) / (2 * pow(network.par.c, 2)));
            }
            return (-network.par.a * (input - network.par.b) * exp(-pow((network.par.b - input), 2) / (2 * pow(network.par.c, 2)))) / pow(network.par.c, 2);
        default:
            return 0;
        }
    }

    void ArtificialNeuralNetwork::calibrateNetworkWeights(std::vector<LearningData> learningSeries, uint64_t maxEpochNumber, double maxError)
    {
        uint64_t epochCounter = 0;
        double error = 0;
        double deltaError = 0;

        do
        {
            deltaError = error;
            epochCounter++;

            for (uint32_t i = 0; i < learningSeries.size(); i++)
            {
                error += calibrateWeightsSeries(learningSeries.at(i).input, learningSeries.at(i).desiredOutput);
            }
            error = error / learningSeries.size();
            std::random_shuffle(learningSeries.begin(), learningSeries.end());
            deltaError -= error;
            std::cout << std::endl;
            std::cout << "I nr " << epochCounter << " err : " << error << " LS: " << learningSpeed << " dE " << deltaError << std::endl;
            // if(deltaError < 0.2)
            // {
            //     learningSpeed *= 1.0001;
            // }
            if (deltaError > 0.002)
            {
                learningSpeed *= 0.9999;
            }

        } while (error > maxError && epochCounter < maxEpochNumber);
        std::cout << "Calibrated. Epoch number: " << epochCounter << std::endl;
        std::cout << "Error = " << error << ", DeltaError = " << deltaError << std::endl;
        std::cout << "LearningSpeed = " << learningSpeed << std::endl;
    }

    double ArtificialNeuralNetwork::calibrateWeightsSeries(std::vector<double> inputs, std::vector<double> desiredOutputs)
    {
        if (inputs.size() != v_NetworkShape.front() || desiredOutputs.size() != v_NetworkShape.back())
        {
            std::cout << "Wrong inputs or outputs dimension \n";
        }

        double errorSum = 0;
        std::vector<double> layerIO = inputs;
        std::vector<double> error;

        error = networkImpulse(inputs);

        for (uint16_t i = 0; i < v_NetworkShape.back(); i++)
        {
            error.at(i) = desiredOutputs.at(i) - error.at(i);
        }
        v_NeuronLayers.back().setLayerError(error);

        for (uint16_t i = 0; i < v_NetworkShape.size() - 2; i++)
        {
            error = v_NeuronLayers.at(v_NetworkShape.size() - i - 2).calculatePreviousLayerError(error, v_NetworkShape.at(v_NetworkShape.size() - i - 2));
            v_NeuronLayers.at(v_NetworkShape.size() - i - 3).setLayerError(error);
        }

        for (uint16_t i = 0; i < v_NetworkShape.size() - 1; i++)
        {
            v_NeuronLayers.at(i).calibrateLayerWeights(layerIO);
            layerIO = v_NeuronLayers.at(i).layerImpulse(layerIO);
        }

        error = networkImpulse(inputs);
        for (uint16_t i = 0; i < v_NetworkShape.back(); i++)
        {
            error.at(i) = desiredOutputs.at(i) - error.at(i);
            errorSum += fabs(error.at(i));
        }

        return errorSum;
    }

    std::vector<double> ArtificialNeuralNetwork::NeuronLayer::calculatePreviousLayerError(std::vector<double> layerError, int numberOfNeuronsInPreviousLayer)
    {
        std::vector<double> previousLayerError;
        for (uint16_t i = 0; i < numberOfNeuronsInPreviousLayer; i++)
        {
            double ErrorSum = 0;
            for (uint16_t j = 0; j < v_Neurons.size(); j++)
            {
                ErrorSum += layerError.at(j) * v_Neurons.at(j).getWeight(i);
            }
            previousLayerError.push_back(ErrorSum);
            ErrorSum = 0;
        }
        return previousLayerError;
    }

    void ArtificialNeuralNetwork::NeuronLayer::setLayerError(std::vector<double> layerError)
    {
        if (layerError.size() == v_Neurons.size())
        {
            for (uint16_t i = 0; i < v_Neurons.size(); i++)
            {
                v_Neurons.at(i).setNeuronError(layerError.at(i));
            }
        }
        else
        {
            std::cout << "SetLayerError failed: Vectors have wrong dimention\n";
        }
    }

    void ArtificialNeuralNetwork::NeuronLayer::calibrateLayerWeights(std::vector<double> inputs)
    {
        for (uint16_t i = 0; i < v_Neurons.size(); i++)
        {
            v_Neurons.at(i).calibrateNeuronWeight(inputs);
        }
    }

    double ArtificialNeuralNetwork::NeuronLayer::Neuron::getWeight(uint32_t weightNumber)
    {
        return v_Weights.at(weightNumber);
    }

    void ArtificialNeuralNetwork::NeuronLayer::Neuron::setNeuronError(double error)
    {
        m_NeuronError = error;
    }

    void ArtificialNeuralNetwork::NeuronLayer::Neuron::calibrateNeuronWeight(std::vector<double> inputs)
    {
        for (uint16_t i = 0; i < v_Weights.size() - 1; i++)
        {
            v_Weights.at(i) += network.learningSpeed * m_NeuronError * functionOutput(EFunctionDerivative::FUNCTION_DERIVATIVE, v_Weights.at(i) * inputs.at(i)) * inputs.at(i);
        }
        v_Weights.back() += network.learningSpeed * m_NeuronError * functionOutput(EFunctionDerivative::FUNCTION_DERIVATIVE, v_Weights.back());
    }

}
