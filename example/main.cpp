#include <vector>
#include <map>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <ctime>
#include <fstream>

#define WITHOUT_NUMPY
#include "matplotlibcpp.h"
#include "../FANN/FANN.h"

int main()
{
    // Add seed for random functions
    srand(time(NULL));

    // Define shape of neural network
    std::vector<uint64_t> shape = {1, 8, 8, 8, 1};

    // Define type of activation function
    AI::EActivationFunction funcType = AI::EActivationFunction::BISIGMOID;

    // Define parameters for activation function (a = 1.4, b = 0, c = 0)
    AI::FunctionParams parameters(1.4);

    // Define initial learning speed (steps on function derivative)
    double lspeed = 0.001;

    // Create neural network
    AI::ArtificialNeuralNetwork neuralNetwork(shape, parameters, funcType, lspeed);

    // Data vector for learning purpose
    std::vector<AI::LearningData> data;

    // Vectors for storing learning data
    std::vector<double> time;
    std::vector<double> sin;

    // Fill data vector with random sine values


    for (int i = 0; i < 20; i++)
    {
        // Vectors of inputs and outputs needed to populate learning data
        std::vector<double> inputVector;  // input
        std::vector<double> outputVector; // desired output

        // Single input with desired output
        AI::LearningData d;

        time.push_back(static_cast<double>(rand()) / RAND_MAX * 2 * M_PI);
        sin.push_back(std::sin(time.back()));

        inputVector.push_back(time.back());
        outputVector.push_back(sin.back());

        d.input = inputVector;
        d.desiredOutput = outputVector;

        data.push_back(d);
    }


    // Calibrate weights (network training), provide learning data, max iteration number and max error
    neuralNetwork.calibrateNetworkWeights(data, 10000, 0.05);

    std::vector<double> outputForPlotting;
    std::vector<double> inputForPlotting;

    // Get outputs from trained network
    for (int i = 0; i < 100; i++)
    {
        std::vector<double> input;
        inputForPlotting.push_back(static_cast<double>(i) / 100 * 2 * M_PI);
        input.push_back(inputForPlotting.back());
        outputForPlotting.push_back(neuralNetwork.networkImpulse(input).at(0));
    }

    matplotlibcpp::plot(inputForPlotting, outputForPlotting);
    matplotlibcpp::plot(time, sin, "*");
    matplotlibcpp::show();

    return 0;
}
