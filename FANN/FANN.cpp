#include "FANN.h"

#include <fstream>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <cmath>

namespace AI
{
    ArtificialNeuralNetwork::ArtificialNeuralNetwork(std::vector<uint64_t> networkShape, FunctionParams functionParameters, EActivationFunction function, double lspeed) : m_learningSpeed(lspeed), m_ActivationFunctionSelection(function), m_FuncParameters(functionParameters), m_NetworkShape(networkShape)
    {
        for (uint16_t i = 1; i < networkShape.size(); i++)
        {
            m_NeuronLayers.push_back(NeuronLayer(networkShape.at(i), networkShape.at(i - 1), *this));
        }
    }

    ArtificialNeuralNetwork::NeuronLayer::NeuronLayer(uint32_t numberOfNeurons, uint32_t numberOfNeuronsInPreviousLayer, ArtificialNeuralNetwork &net) : m_network(net)
    {
        for (uint16_t i = 0; i < numberOfNeurons; i++)
        {
            m_Neurons.push_back(Neuron(numberOfNeuronsInPreviousLayer, m_network));
        }
    }

    ArtificialNeuralNetwork::NeuronLayer::Neuron::Neuron(uint32_t numberOfInputs, ArtificialNeuralNetwork &net) : m_network(net)
    {
        // Each of neuron weights is initialised with value between -1 and 1
        // Number of inputs is increased by 1 for including bias.
        for (uint32_t i = 0; i < numberOfInputs + 1; i++)
        {
            m_Weights.push_back(double(std::rand()) / RAND_MAX * 2 - 1);
        }
    }

    std::vector<double> ArtificialNeuralNetwork::networkImpulse(std::vector<double> inputs)
    {

        if (inputs.size() != m_NetworkShape.at(0))
        {
            std::cout << inputs.size() << " " << m_NetworkShape.at(0) << std::endl;
            return inputs;
        }
        std::vector<double> layerIO = inputs;
        for (uint16_t i = 0; i < m_NetworkShape.size() - 1; i++)
        {
            layerIO = m_NeuronLayers.at(i).layerImpulse(layerIO);
        }
        return layerIO;
    }

    std::vector<double> ArtificialNeuralNetwork::NeuronLayer::layerImpulse(const std::vector<double> &previousLayerInput)
    {
        std::vector<double> layerOutputs;
        for (uint16_t i = 0; i < m_Neurons.size(); i++)
        {
            layerOutputs.push_back(m_Neurons.at(i).neuronImpulse(previousLayerInput));
        }
        return layerOutputs;
    }

    double ArtificialNeuralNetwork::NeuronLayer::Neuron::neuronImpulse(const std::vector<double> &inputs)
    {
        double inputXWeightsSum = 0;

        for (uint16_t i = 0; i < inputs.size(); i++)
        {
            inputXWeightsSum += inputs.at(i) * m_Weights.at(i);
        }
        inputXWeightsSum += m_Weights.back();
        return functionOutput(EFunctionDerivative::FUNCTION, inputXWeightsSum);
    }

    double ArtificialNeuralNetwork::NeuronLayer::Neuron::functionOutput(EFunctionDerivative select, double input)
    {
        switch (m_network.m_ActivationFunctionSelection)
        {
        case EActivationFunction::LINEAR:
            if (select == EFunctionDerivative::FUNCTION)
            {
                return m_network.m_FuncParameters.a * input + m_network.m_FuncParameters.b;
            }
            return m_network.m_FuncParameters.a;
        case EActivationFunction::UNISIGMOID:
            if (select == EFunctionDerivative::FUNCTION)
            {
                return 1 / (1 + exp(-m_network.m_FuncParameters.a * input));
            }
            return (m_network.m_FuncParameters.a * exp(-m_network.m_FuncParameters.a * input)) / (pow(exp(-m_network.m_FuncParameters.a * input) + 1, 2));
        case EActivationFunction::BISIGMOID:
            if (select == EFunctionDerivative::FUNCTION)
            {
                return (1 - exp(-m_network.m_FuncParameters.a * input)) / (1 + exp(-m_network.m_FuncParameters.a * input));
            }
            return (2 * m_network.m_FuncParameters.a * exp(m_network.m_FuncParameters.a * input) / pow(exp(m_network.m_FuncParameters.a * input) + 1, 2));
        case EActivationFunction::GAUSS:
            if (select == EFunctionDerivative::FUNCTION)
            {
                return m_network.m_FuncParameters.a * exp(-pow(input - m_network.m_FuncParameters.b, 2) / (2 * pow(m_network.m_FuncParameters.c, 2)));
            }
            return (-m_network.m_FuncParameters.a * (input - m_network.m_FuncParameters.b) * exp(-pow((m_network.m_FuncParameters.b - input), 2) / (2 * pow(m_network.m_FuncParameters.c, 2)))) / pow(m_network.m_FuncParameters.c, 2);
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
            // TODO ajdust learning speed better
            // if(deltaError < 0.2)
            // {
            //     m_learningSpeed *= 1.0001;
            // }
            if (deltaError > 0.002)
            {
                m_learningSpeed *= 0.9999;
            }

        } while (error > maxError && epochCounter < maxEpochNumber);
        std::cout << "Calibrated. Epoch number: " << epochCounter << std::endl;
        std::cout << "Error = " << error << ", DeltaError = " << deltaError << std::endl;
        std::cout << "LearningSpeed = " << m_learningSpeed << std::endl;
    }

    double ArtificialNeuralNetwork::calibrateWeightsSeries(std::vector<double> inputs, std::vector<double> desiredOutputs)
    {
        if (inputs.size() != m_NetworkShape.front() || desiredOutputs.size() != m_NetworkShape.back())
        {
            std::cout << "Wrong inputs or outputs dimension \n";
        }

        double errorSum = 0;
        std::vector<double> layerIO = inputs;
        std::vector<double> error;

        error = networkImpulse(inputs);

        for (uint16_t i = 0; i < m_NetworkShape.back(); i++)
        {
            error.at(i) = desiredOutputs.at(i) - error.at(i);
        }
        m_NeuronLayers.back().setLayerError(error);

        for (uint16_t i = 0; i < m_NetworkShape.size() - 2; i++)
        {
            error = m_NeuronLayers.at(m_NetworkShape.size() - i - 2).calculatePreviousLayerError(error, m_NetworkShape.at(m_NetworkShape.size() - i - 2));
            m_NeuronLayers.at(m_NetworkShape.size() - i - 3).setLayerError(error);
        }

        for (uint16_t i = 0; i < m_NetworkShape.size() - 1; i++)
        {
            m_NeuronLayers.at(i).calibrateLayerWeights(layerIO);
            layerIO = m_NeuronLayers.at(i).layerImpulse(layerIO);
        }

        error = networkImpulse(inputs);
        for (uint16_t i = 0; i < m_NetworkShape.back(); i++)
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
            for (uint16_t j = 0; j < m_Neurons.size(); j++)
            {
                ErrorSum += layerError.at(j) * m_Neurons.at(j).getWeight(i);
            }
            previousLayerError.push_back(ErrorSum);
            ErrorSum = 0;
        }
        return previousLayerError;
    }

    void ArtificialNeuralNetwork::NeuronLayer::setLayerError(std::vector<double> layerError)
    {
        if (layerError.size() == m_Neurons.size())
        {
            for (uint16_t i = 0; i < m_Neurons.size(); i++)
            {
                m_Neurons.at(i).setNeuronError(layerError.at(i));
            }
        }
        else
        {
            std::cout << "SetLayerError failed: Vectors have wrong dimention\n";
        }
    }

    void ArtificialNeuralNetwork::NeuronLayer::calibrateLayerWeights(std::vector<double> inputs)
    {
        for (uint16_t i = 0; i < m_Neurons.size(); i++)
        {
            m_Neurons.at(i).calibrateNeuronWeight(inputs);
        }
    }

    double ArtificialNeuralNetwork::NeuronLayer::Neuron::getWeight(uint32_t weightNumber)
    {
        return m_Weights.at(weightNumber);
    }

    void ArtificialNeuralNetwork::NeuronLayer::Neuron::setNeuronError(double error)
    {
        m_NeuronError = error;
    }

    void ArtificialNeuralNetwork::NeuronLayer::Neuron::calibrateNeuronWeight(std::vector<double> inputs)
    {
        for (uint16_t i = 0; i < m_Weights.size() - 1; i++)
        {
            m_Weights.at(i) += m_network.m_learningSpeed * m_NeuronError * functionOutput(EFunctionDerivative::FUNCTION_DERIVATIVE, m_Weights.at(i) * inputs.at(i)) * inputs.at(i);
        }
        m_Weights.back() += m_network.m_learningSpeed * m_NeuronError * functionOutput(EFunctionDerivative::FUNCTION_DERIVATIVE, m_Weights.back());
    }

}
