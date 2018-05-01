//
// Created by lukasz on 19.03.17.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include "subspaceClustering.h"

using namespace subspaceClusteringParallel;


void data_param(std::string filename, int &n_samples, int &dim, const char separator) {
    std::ifstream in(filename, std::ios::in);
    std::string line;
    n_samples = 0;

    if (!in) {
        std::cout << "Cannot open input file.\n";
        exit(1);
    }

    std::getline(in, line);
    dim = (int) std::count(line.begin(), line.end(), separator);
    ++dim;
    ++n_samples;
    while (std::getline(in, line))
        ++n_samples;

    in.clear();
    in.close();

    if (!in.good()) {
        std::cout << "A file error occurred.\n";
        exit(1);
    }
}


void readfile(double *data, std::string filename, char separator) {
    std::ifstream in(filename, std::ios::in);
    std::string line, item;
    int i;

    if (!in) {
        std::cout << "Cannot open input file.\n";
        exit(1);
    }

    i = 0;
    std::stringstream linestream;
    while (std::getline(in, line)) {
        linestream << line;
        while (std::getline(linestream, item, separator))
            if (line.length() > 0)
                data[i++] = std::stod(item);
        linestream.clear();
    }

    in.clear();
    in.close();

    if (!in.good()) {
        std::cout << "A file error occurred.\n";
        exit(1);
    }
}


/*
 * This function reads data from file.
 */
double *readData(std::string filename, int &n_samples, int &dim, const char separator) {
    std::ifstream in(filename, std::ios::in);
    std::string line, item;
    double *data;
    int i;

    n_samples = 0;

    if (!in) {
        std::cout << "Cannot open input file.\n";
        exit(1);
    }

    std::getline(in, line);
    dim = (int) std::count(line.begin(), line.end(), separator);
    ++dim;
    ++n_samples;
    while (std::getline(in, line))
        ++n_samples;

    in.clear();
    in.seekg(0, std::ios::beg);

    i = 0;
    data = new double[n_samples * dim];
    std::stringstream linestream;
    while (std::getline(in, line)) {
        linestream << line;
        while (std::getline(linestream, item, separator))
            if (line.length() > 0)
                data[i++] = std::stod(item);
        linestream.clear();
    }

    in.clear();
    in.close();

    if (!in.good()) {
        std::cout << "A file error occurred.\n";
        exit(1);
    }
    return data;
}


int Cluster::N;

/**
 * @param N (<i><b>size_t</b></i>) - size of cluster (what dimension are the data represented)
 */
Cluster::Cluster(int N) {
    Cluster::N = N;
    this->dim = 0.0;
    this->mean.reserve(static_cast<unsigned long>(N));
    this->diagCov.reserve(static_cast<unsigned long>(N));
    this->sortedDiagCov.reserve(static_cast<unsigned long>(N));
    for (int i = 0; i < N; ++i) {
        this->mean.push_back(0);
        this->diagCov.push_back(0);
        this->sortedDiagCov.push_back(0);
    }
    this->weight = 0;
    this->memory = 0.0;
}

/**
 * @param orig (<i><b>Cluster</b></i>)
 */
Cluster::Cluster(const Cluster &orig) {
    this->N = orig.N;
    this->dim = orig.dim;
//    this->mean.clear();
//    this->diagCov.clear();
//    this->sortedDiagCov.clear();
    std::copy(orig.mean.begin(), orig.mean.end(), std::back_inserter(this->mean));
    std::copy(orig.diagCov.begin(), orig.diagCov.end(), std::back_inserter(this->diagCov));
    std::copy(orig.sortedDiagCov.begin(), orig.sortedDiagCov.end(), std::back_inserter(this->sortedDiagCov));
    this->weight = orig.weight;
    this->memory = orig.memory;
}

/**
 * Destructor
 */
Cluster::~Cluster() {
    this->mean.clear();
    this->diagCov.clear();
    this->sortedDiagCov.clear();
}

/**
 * @return dim of Cluster
 */
double Cluster::getDim() const {
    return this->dim;
}

/**
 * @return mean of Cluster
 */
const std::vector<double> &Cluster::getMean() const {
    return this->mean;
}

/**
 * @return main diagonal of covariance matrix Cluster
 */
const std::vector<double> &Cluster::getDiagCov() const {
    return this->diagCov;
}

/**
 * @return weight of Cluster
 */
int Cluster::getWeight() const {
    return this->weight;
}

/**
 * @return memory of Cluster
 */
double Cluster::getMemory() const {
    return this->memory;
}

/**
 * The function changes all (except 'N', 'dim', memory ',' error ') data in a cluster caused by adding or removing a point to/from a cluster.
 *
 * @param point (<i><b>double*</i></b>) - point, which will be added to the cluster
 * @param weightPoint (<i><b>int</i></b>) - weight of this point: '1' when we add, '-1' when we subtract 'point' from the cluster
 */
void Cluster::changePoint(const double *point, const int weightPoint) {
    double pu, pv, temp;
    pv = (double) (this->weight + weightPoint);
    pu = (double) this->weight / pv;
    pv = (double) weightPoint / pv;

    for (int i = 0; i < this->N; ++i) {
        temp = point[i] - this->mean[i];
        this->mean[i] = pu * this->mean[i] + pv * point[i];
        this->diagCov[i] = pv * pu * temp * temp + pu * this->diagCov[i];
    }
    this->weight += weightPoint;
}


/*
 * This function sorts of vector.
 */
void Cluster::sortDiagCov() {
    this->sortedDiagCov.clear();
    std::copy(this->diagCov.begin(), this->diagCov.end(), std::back_inserter(this->sortedDiagCov));
    std::sort(sortedDiagCov.begin(), sortedDiagCov.end(), std::greater<>());
}

/*
 * This function divides the array into two parts. The first part contains 
 * the first n largest elements and the second part contains the other ones.
 */
void Cluster::divide_nth_element(const int totalWeight, const int bits) {
    sortedDiagCov.clear();
    std::copy(this->diagCov.begin(), this->diagCov.end(), std::back_inserter(this->sortedDiagCov));

    double factor;

    if (bits == 0)
        factor = this->memory / this->weight;
    else
        factor = (this->memory + std::log2((double) this->weight / totalWeight) * this->weight) / (bits * this->weight);


    if (factor > 0 && factor < this->N - 1) {
        std::nth_element(sortedDiagCov.begin(), sortedDiagCov.begin() + std::ceil(factor), sortedDiagCov.end(),
                         std::greater<>());
        std::nth_element(sortedDiagCov.begin(), sortedDiagCov.begin() + std::floor(factor),
                         sortedDiagCov.begin() + std::floor(factor),
                         std::greater<>());
    }
}


/**
 * The function (overloading operator =) copies the contents of the 'orig'
 *
 * @param orig (<i><b>Cluster&</b></i>) - cluster
 * @return copy of 'orig'
 */
Cluster &Cluster::operator=(const Cluster &orig) {
    this->dim = orig.dim;
    this->mean.clear();
    this->diagCov.clear();
    this->sortedDiagCov.clear();
    std::copy(orig.mean.begin(), orig.mean.end(), std::back_inserter(this->mean));
    std::copy(orig.diagCov.begin(), orig.diagCov.end(), std::back_inserter(this->diagCov));
    std::copy(orig.sortedDiagCov.begin(), orig.sortedDiagCov.end(), std::back_inserter(this->sortedDiagCov));
    this->weight = orig.weight;
    this->memory = orig.memory;
    return *this;
}

/**
 * @param factor (<i><b>double</i></b>) - ratio of memory to number of points (weight)
 * @return accuracy/error with number of parameters
 */
double Cluster::err(const double factor) const {
    double s = 0.0;
    if (factor >= (double) this->N) return s;
    else if (factor <= std::numeric_limits<double>::epsilon()) {
        for (int i = 0; i < this->N; ++i)
            s += sortedDiagCov[i];
    } else {
        for (int i = this->N - 1; i >= std::ceil(factor); i--)
            s += sortedDiagCov[i];
        s += (std::ceil(factor) - factor) * sortedDiagCov[(int) std::floor(factor)];
    }
    return s;
}

/**
 * Cost function
 *
 * @param totalWeight (<i><b>int</i></b>) - the total number of data
 * @param bits (<i><b>size_t</i></b>) - number of bits needed to memorize one scalar
 * @return cost function for one cluster
 */
double Cluster::errorFUN(const int totalWeight, const int bits) const {
    double factor;
    if (bits == 0)
        factor = this->memory / this->weight;
    else
        factor = (this->memory + std::log2((double) this->weight / totalWeight) * this->weight) / (bits * this->weight);
    factor = this->err(factor) * this->weight;
    return factor;
}

/**
 * The function checks whether the cluster is deleted or not
 *
 * @param totalWeight (<i><b>int</i></b>) - the total number of data
 * @param toleranceFactor (<i><b>double</i></b>) - the coefficient defining the minimum number of points in relation to the total number of data sets
 * @return whether to remove the cluster (true) or not (false)
 */
bool Cluster::unassign(const int totalWeight, const double toleranceFactor) const {
    return this->weight <= std::max(toleranceFactor * totalWeight, 2.0 * this->N);
}

/**
 * Overloaded exit operator for 'Cluster' class
 *
 * @return Prints cluster data: dimension, memory, weight, center, covariance matrix
 */
std::ostream &subspaceClusteringParallel::operator<<(std::ostream &out, const Cluster &c) {
    out << "\nDim:\t" << c.dim;
    out << "\nMemory:\t" << c.memory;
    out << "\nWeight:\t" << c.weight;
    out << "\nMean:\t";
    for (auto &val: c.mean)
        out << val << " ";
    out << std::endl;

    out << "Main diagonal covariance matrix:\t";
    for (auto &val: c.diagCov)
        out << val << " ";
    out << std::endl;

    out << "Sorted diagonal covariance matrix:\t";
    for (auto &val: c.sortedDiagCov)
        out << val << " ";
    out << std::endl;

    return out;
}
