#include<vector>
#include<map>
#include<cstdint>
#include<iostream>
#include<cmath>
#include<ctime>
#include<fstream>
#include "../FANN/FANN.h"

#define WITHOUT_NUMPY
#include "matplotlibcpp.h"

#define _USE_MATH_DEFINES

namespace plt = matplotlibcpp;
int main() 
{
    srand(time(NULL));
    std::vector<uint64_t> shape = {1, 8, 8, 8, 1};
    FunctionParams parm;
    parm.a = 1.4;

    ArtificialNeuralNetwork a(shape, parm);

    // plot
    std::vector<LearningData> data;
    std::vector<double> tempTime;
    std::vector<double> tempSin;

    std::vector<double> time;
    std::vector<double> sin;

    for(int i = 0; i < 20; i++)
    {
        double in = double((double)rand() / RAND_MAX) * 2 * M_PI;
        double out = std::sin(in);
        tempTime.push_back(in);
        time.push_back(in);
        tempSin.push_back(out);
        sin.push_back(out);
        LearningData d;
        d.input = tempTime;
        d.desiredOutput = tempSin;
        data.push_back(d);

        tempTime.clear();
        tempSin.clear();
    }

    std::cout << "nauka" << std::endl;
    a.calibrateNetworkWeights(data, 10000, 0.05);
    std::cout << "po nauce" << std::endl;
    
    std::vector<double> output;
    std::vector<double> time2;
    for(int i = 0; i < 100; i++)
    {
        std::vector<double> in;
        in.push_back((double)i/100 * 2 * M_PI );
        time2.push_back(in.at(0));
        output.push_back(a.networkImpulse(in).at(0));

        //std::cout << "in = " << time.at(i) << "out = " << output.at(i) << std::endl;
    }

    plt::plot(time, sin, "*");
    plt::plot(time2, output);
    plt::plot(time, sin, "*");
    plt::show();
    
    return 0;
}

