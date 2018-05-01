//
// Created by lukasz on 25.12.17.
//

#include <getopt.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include "subspaceClustering.h"

using namespace std;
using namespace subspaceClusteringParallel;

std::string inputfile, outputDir = "./output.txt";
unsigned int n_clusters = 10, iters = 5, bits = 0, n_thread = 1, l_ratio = 1, r_ratio = 95;

void PrintHelp() {
    std::cout <<
              "--inputfile -f:      Input file\n"
              "--n_clusters -k:     Number of clusters\n"
              "--iters -i:          Number of iterations\n"
              "--bits -b:           Number of bits\n"
              "--t -t               Number of threads\n"
              "--l_ratio -l         Left side of range of compression ratio\n"
              "--r_ratio -r         Right side of range of compression ratio\n"
              "--output -o:         Path to output directory\n";
    exit(1);
}

void ProcessArgs(int argc, char **argv) {
    const char *const short_opts = "f:k:i:b:t:l:r:o:h";
    const option long_opts[] = {
            {"inputfile",  1, nullptr, 'f'},
            {"n_clusters", 1, nullptr, 'k'},
            {"iters",      0, nullptr, 'i'},
            {"bits",       0, nullptr, 'b'},
            {"t",          0, nullptr, 't'},
            {"l_ratio",    1, nullptr, 'l'},
            {"r_ratio",    1, nullptr, 'r'},
            {"output",     0, nullptr, 'o'},
            {"help",       0, nullptr, 'h'},
            {nullptr,      0, nullptr, 0}
    };

    while (true) {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

        if (-1 == opt)
            break;

        switch (opt) {
            case 'f':
                inputfile = std::string(optarg);
                std::cout << "Input file: " << inputfile << std::endl;
                break;

            case 'k':
                n_clusters = static_cast<unsigned int>(std::stoi(optarg));
                std::cout << "Number of clusters:" << n_clusters << std::endl;
                break;

            case 'i':
                iters = static_cast<unsigned int>(std::stoi(optarg));
                std::cout << "Number of iterations: " << iters << std::endl;
                break;

            case 'b':
                bits = static_cast<unsigned int>(std::stoi(optarg));
                std::cout << "Number of bits:" << bits << std::endl;
                break;

            case 't':
                n_thread = static_cast<unsigned int>(std::stoi(optarg));
                std::cout << "Number of threads:" << n_thread << std::endl;
                break;

            case 'l':
                l_ratio = static_cast<unsigned int>(std::stoi(optarg));
                if (l_ratio > 100) {
                    std::cout << "Left side of range of compression ratio should be value less 100!\n";
                    exit(1);
                }
                std::cout << "Left side of range of compression ratio: " << l_ratio << std::endl;
                break;

            case 'r':
                r_ratio = static_cast<unsigned int>(std::stoi(optarg));
                if (l_ratio > r_ratio || r_ratio > 100) {
                    std::cout
                            << "Right side of range of compression ratio should be value less 100 and left side of range of compression ratio!\n";
                    exit(1);
                }
                std::cout << "Right value of range of compression ratio: " << r_ratio << std::endl;
                break;

            case 'o':
                outputDir = std::string(optarg);
                std::cout << "Path to output directory: " << outputDir << std::endl;
                break;

            case 'h': // -h or --help
            default:
                PrintHelp();
                break;
        }
    }
}


int main(int argc, char **argv) {
    ProcessArgs(argc, argv);

    int size[2];

    data_param(inputfile, *size, *(size + 1)); // n_samples = size[0], n_features = size[1]
    auto *data = new double[size[0] * size[1]];
    readfile(data, inputfile);

    ContainerClusters c;

    c.Hartigan_parallel_ratio(iters, size, data, n_clusters, n_thread, bits, l_ratio, r_ratio, outputDir);

    delete[] data;
    return 0;
}
