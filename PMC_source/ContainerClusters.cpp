//
// Created by lukasz on 19.03.17.
//

#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <fstream>
#include "subspaceClustering.h"


using namespace subspaceClusteringParallel;

/*
 * This function calculates index for upper matrix without diagonal (indices (i, j) used as in classical 2-dimensional matrix).
 */
inline int index(int i, int j, int dim) {
    int res;
    if (i == j)
        res = -1;
    else if (i < j)
        res = (dim - 1) * i - (i * (i - 1)) / 2 + j - i - 1;
    else
        res = (dim - 1) * j - (j * (j - 1)) / 2 + i - j - 1;
    return res;
}

/**
 * Constructor
 */
ContainerClusters::ContainerClusters() {
    size = 0;
    error = std::numeric_limits<double>::infinity();
}

/**
 * Copy constructor
 */
ContainerClusters::ContainerClusters(const ContainerClusters &orig) {
    this->size = orig.size;
    this->error = orig.error;
    this->tableClusters.reserve(this->size);
    for (auto cl: orig.tableClusters)
        this->tableClusters.emplace_back(new Cluster(*cl));
}


/**
 * Cleaning the contents of the container
 */
void ContainerClusters::clear() {
//    this->size = 0;
//    this->error = std::numeric_limits<double>::infinity();
    for (Cluster *cl: this->tableClusters)
        delete cl;
    this->tableClusters.clear();
}


/**
 * Destructor
 */
ContainerClusters::~ContainerClusters() {
    this->clear();
}

/**
 * @return size of Container
 */
unsigned int ContainerClusters::getSize() const {
    return this->size;
}

/**
 * @return error of Container
 */
double ContainerClusters::getError() const {
    return this->error;
}

/**
 * reset error clustring
 */
void ContainerClusters::resetError() {
    this->error = std::numeric_limits<double>::infinity();
}

/**
 * @return the container of clusters
 */
const std::vector<Cluster *> &ContainerClusters::getContainer() const {
    return this->tableClusters;
}

/**
 * The (overload operator =) function copies the values of the 'orig' cluster container, deleting the previous data
 *
 * @param orig (<i><b>ContainerClusters&</b></i>) - the container of clusters
 * @return copy 'orig'
 */
ContainerClusters &ContainerClusters::operator=(const ContainerClusters &orig) {
    this->clear();
    this->size = orig.size;
    this->error = orig.error;
    this->tableClusters.reserve(this->size);
    for (Cluster *cl: orig.tableClusters)
        this->tableClusters.emplace_back(new Cluster(*cl));

    return *this;
}

/**
 * Create clusters, randomly linking points to clusters
 *
 * @param size (<i><b>size_t[]</b></i>) - size of number of samples and number of features
 * @param data (<i><b>double*</b></i>) - 2-dimensional array, which save data <i>row-major order</i>
 * @param groups (<i><b>size_t*</b></i>) - list of id clusters (values: 0,1,...,howClusters-1;  <i>row-major order</i>)
 * @param howClusters (<i><b>size_t</b></i>) - determines how much we want to create clusters
 * @param allMemory (<i><b>double</b></i>) - general memory we have at our disposal
 */
void ContainerClusters::createCluster(const int size[], const double *data, const int *group,
                                      const unsigned int howClusters, const double allMemory, const int bits) {
    this->size = howClusters;
    this->tableClusters.reserve(howClusters);
    int i;

    for (i = 0; i < howClusters; ++i)
        this->tableClusters.emplace_back(new Cluster(size[1]));

    for (i = 0; i < size[0]; ++i)
        this->tableClusters[group[i]]->changePoint(data + i * Cluster::N, 1);

    for (i = 0; i < howClusters; ++i) {
        tableClusters[i]->memory = allMemory * (double) tableClusters[i]->weight / size[0];
        //        tableClusters[i]->sortDiagCov();
        tableClusters[i]->divide_nth_element(size[0], bits);
    }
}

/**
 * The function calculates the minimum energy of two clusters and their memory
 *
 * @param array (<i><b>double[2]</i></b>) -  array, which saves cost function and memory of second cluster
 * @param idCluster1 (<i><b>Cluster &</i></b>) - index of the first cluster
 * @param idCluster2 (<i><b>Cluster &</i></b>) - index of the second cluster
 * @param totalWeight (<i><b>int</i></b>) - the total number of data
 * @param bits (<i><b>size_t</i></b>) - number of bits needed to memorize one scalar
 */
void ContainerClusters::errorsTWOclusters(double *array, const int idCluster1, const int idCluster2,
                                          const int totalWeight, const int bits) const {
    double ilewsp;

    if (bits == 0)
        ilewsp = this->tableClusters[idCluster1]->memory + this->tableClusters[idCluster2]->memory;
    else
        ilewsp = (this->tableClusters[idCluster1]->memory + this->tableClusters[idCluster2]->memory +
                  std::log2((double) this->tableClusters[idCluster1]->weight / totalWeight) *
                  this->tableClusters[idCluster1]->weight +
                  std::log2((double) this->tableClusters[idCluster2]->weight / totalWeight) *
                  this->tableClusters[idCluster2]->weight) / bits;

    //error two clusters
    array[0] = std::numeric_limits<double>::infinity();
    //memory of cluster 'idCluster2'
    array[1] = this->tableClusters[idCluster2]->memory;

    if (ilewsp > std::numeric_limits<double>::epsilon()) {
        double dim1, dim2;
        auto *temp = new double[4];

        // old dim1, dim2
        dim1 = this->tableClusters[idCluster1]->memory / this->tableClusters[idCluster1]->weight;
        dim2 = this->tableClusters[idCluster2]->memory / this->tableClusters[idCluster2]->weight;

        dim1 = std::round(dim1);
        dim2 = std::round(dim2);
        for (int i = -1; i <= 1; ++i) {
            if (dim1 + i < 0 || dim2 + i < 0 || dim1 + i > Cluster::N || dim2 + i > Cluster::N) continue;
            temp[0] = (dim1 + i) * this->tableClusters[idCluster1]->weight;
            temp[2] = (dim2 + i) * this->tableClusters[idCluster2]->weight;

            if (temp[0] <= ilewsp) {
                temp[1] = ilewsp - temp[0];
                temp[3] = tableClusters[idCluster1]->err(dim1 + i) * tableClusters[idCluster1]->weight +
                          tableClusters[idCluster2]->err(temp[1] / tableClusters[idCluster2]->weight) *
                          tableClusters[idCluster2]->weight;
                if (temp[3] < array[0]) {
                    array[1] = temp[1];
                    array[0] = temp[3];
                }
            }

            if (temp[2] <= ilewsp) {
                temp[1] = ilewsp - temp[2];
                temp[3] = this->tableClusters[idCluster1]->err(temp[1] / this->tableClusters[idCluster1]->weight) *
                          this->tableClusters[idCluster1]->weight +
                          this->tableClusters[idCluster2]->err(dim2 + i) * this->tableClusters[idCluster2]->weight;

                if (temp[3] < array[0]) {
                    array[1] = temp[2];
                    array[0] = temp[3];
                }
            }
        }

        delete[] temp;

        if (bits != 0)
            array[1] = -std::log2((double) this->tableClusters[idCluster2]->weight / totalWeight) *
                       this->tableClusters[idCluster2]->weight + bits * array[1];
        // new memory of cluster 'idCluster1' = old memories Cluster1+Cluster2 - new memory Cluster2
    }
}

/**
 * The function improves the size and memory of two clusters
 *
 * @param idCluster1 (<i><b>Cluster &</i></b>) - index of the first cluster
 * @param idCluster2 (<i><b>Cluster &</i></b>) - index of the second cluster
 * @param totalWeight (<i><b>int</i></b>) - the total number of data
 * @param bits (<i><b>size_t</i></b>) - number of bits needed to memorize one scalar
 *
 * @return returns 0 or 1, 0 when cluster 'idCluster1' has an integer dimension, 1 for the second case
 */
int ContainerClusters::updateDim(const int idCluster1, const int idCluster2, const int totalWeight,
                                 const int bits) {
    int ret = 0;
    double ilewsp;
    if (bits == 0)
        ilewsp = this->tableClusters[idCluster1]->memory + this->tableClusters[idCluster2]->memory;
    else
        ilewsp = (this->tableClusters[idCluster1]->memory + this->tableClusters[idCluster2]->memory +
                  std::log2((double) this->tableClusters[idCluster1]->weight / totalWeight) *
                  this->tableClusters[idCluster1]->weight +
                  std::log2((double) this->tableClusters[idCluster2]->weight / totalWeight) *
                  this->tableClusters[idCluster2]->weight) / bits;

    double tempM = -1.0, blad = std::numeric_limits<double>::infinity();
    if (bits == 0) {
        this->tableClusters[idCluster1]->dim =
                this->tableClusters[idCluster1]->memory / this->tableClusters[idCluster1]->weight;
        this->tableClusters[idCluster2]->dim =
                this->tableClusters[idCluster2]->memory / this->tableClusters[idCluster2]->weight;
    } else {
        this->tableClusters[idCluster1]->dim = (this->tableClusters[idCluster1]->memory +
                                                this->tableClusters[idCluster1]->weight *
                                                std::log2((double) this->tableClusters[idCluster1]->weight /
                                                          totalWeight)) /
                                               (bits * this->tableClusters[idCluster1]->weight);
        this->tableClusters[idCluster2]->dim = (this->tableClusters[idCluster2]->memory +
                                                this->tableClusters[idCluster2]->weight *
                                                std::log2((double) this->tableClusters[idCluster2]->weight /
                                                          totalWeight)) /
                                               (bits * this->tableClusters[idCluster2]->weight);
    }


    // klaster z niecalkowitym wymiarem
    if (this->tableClusters[idCluster1]->dim - std::floor(this->tableClusters[idCluster1]->dim) >
        std::numeric_limits<double>::epsilon())
        ret = 1;

    if (ilewsp > std::numeric_limits<double>::epsilon()) {
        double dim1, dim2;
        auto *temp = new double[4];

        dim1 = std::round(this->tableClusters[idCluster1]->dim);
        dim2 = std::round(this->tableClusters[idCluster2]->dim);
        for (int i = -1; i <= 1; ++i) {
            if (dim1 + i < 0 || dim2 + i < 0 || dim1 + i > Cluster::N || dim2 + i > Cluster::N) continue;
            temp[0] = (dim1 + i) * this->tableClusters[idCluster1]->weight;
            temp[2] = (dim2 + i) * this->tableClusters[idCluster2]->weight;

            if (temp[0] <= ilewsp) {
                temp[1] = ilewsp - temp[0];
                temp[3] = this->tableClusters[idCluster1]->err(dim1 + i) * this->tableClusters[idCluster1]->weight +
                          this->tableClusters[idCluster2]->err(temp[1] / this->tableClusters[idCluster2]->weight) *
                          this->tableClusters[idCluster2]->weight;
                if (temp[3] < blad) {
                    tempM = temp[1];
                    blad = temp[3];
                    this->tableClusters[idCluster1]->dim = dim1 + i;
                    this->tableClusters[idCluster2]->dim = temp[1] / this->tableClusters[idCluster2]->weight;
                    ret = 0;
                }
            }

            if (temp[2] <= ilewsp) {
                temp[1] = ilewsp - temp[2];
                temp[3] = this->tableClusters[idCluster1]->err(temp[1] / this->tableClusters[idCluster1]->weight) *
                          this->tableClusters[idCluster1]->weight +
                          this->tableClusters[idCluster2]->err(dim2 + i) * this->tableClusters[idCluster2]->weight;

                if (temp[3] < blad) {
                    tempM = temp[2];
                    blad = temp[3];
                    this->tableClusters[idCluster1]->dim = temp[1] / this->tableClusters[idCluster1]->weight;
                    this->tableClusters[idCluster2]->dim = dim2 + i;
                    ret = 1;
                }
            }
        }

        delete[] temp;

        this->tableClusters[idCluster1]->memory =
                this->tableClusters[idCluster1]->memory + this->tableClusters[idCluster2]->memory;
        if (bits == 0)
            this->tableClusters[idCluster2]->memory = tempM;
        else
            this->tableClusters[idCluster2]->memory =
                    -std::log2((double) this->tableClusters[idCluster2]->weight / totalWeight) *
                    this->tableClusters[idCluster2]->weight + bits * tempM;
        this->tableClusters[idCluster1]->memory -= this->tableClusters[idCluster2]->memory;
    }
    return ret;
}

/**
 * Hartigan iteration
 *
 * @param size (<i><b>size_t[]</b></i>) - size of data
 * @param data (<i><b>double*</b></i>) - 2-dimensional array, which save data <i>row-major order</i>
 * @param groups (<i><b>size_t*</b></i>) - list of id clusters (values: 0,1,...,howClusters-1;  <i>row-major order</i>)
 * @param howClusters (<i><b>size_t</b></i>) - determines how much we want to create clusters
 * @param allMemory (<i><b>double</b></i>) -  general memory we have at our disposal
 * @param bits (<i><b>size_t</i></b>) - number of bits needed to memorize one scalar
 */
void ContainerClusters::stepHartigan(int *group, std::vector<int> &activeClusters, const int size[],
                                     const double *data, const double allMemory, const int bits) {
    bool outside, T = false;
    auto howClusters = static_cast<unsigned int>(activeClusters.back() + 1);
    this->createCluster(size, data, group, howClusters, allMemory);

    std::vector<int>::iterator it, min_it;

    int l, j, m, i, min_weight, id_i;
    double temp, c, tempMemory = 0.;
    auto *point = new double[size[1]];
    double array[2];

    auto *copy_container = new ContainerClusters();

    int stop = size[0]; // zatrzymuje petle for jesli nie zmieni sie zaden punkt miedzy 'stop <-> stop'
    bool check = true;

    int items = 0;

    double error2cluster = -1.;
    auto *errorsTWOclustersArray = new double[(howClusters * (howClusters - 1)) / 2];
    std::fill(errorsTWOclustersArray, errorsTWOclustersArray + (howClusters * (howClusters - 1)) / 2, -1.0);

    while (!T && items < 50) {
        items++;
        T = true;
        for (m = 0; m < size[0] && !(m == stop && check); ++m) {
            check = true;
            std::copy(data + m * size[1], data + (m + 1) * size[1], point);
            l = group[m];

            if (bits != 0) {
                outside = true;
                for (it = activeClusters.begin(); it != activeClusters.end(); it++)
                    if (l == *it) {
                        outside = false;
                        break;
                    }

                if (outside) {
                    temp = this->tableClusters[l]->memory / this->tableClusters[l]->weight;
                    l = activeClusters.front();
                    group[m] = l;
                    this->tableClusters[l]->changePoint(point, 1);
                    this->tableClusters[l]->memory += temp;

                    T = false;
                    stop = m + 1; // stop <- nastapila zmiana
                    check = false;
                }
            }

            *copy_container = *this;
            copy_container->tableClusters[l]->changePoint(point, -1);
            //            copy_container->tableClusters[l]->sortDiagCov();
            copy_container->tableClusters[l]->divide_nth_element(size[0], bits);

            j = l;
            c = 0.0;

            for (it = activeClusters.begin(); it != activeClusters.end(); it++) {
                if (*it != l) {
                    id_i = index(l, *it, howClusters);
                    if (errorsTWOclustersArray[id_i] == -1) {
                        this->errorsTWOclusters(array, l, *it, size[0], bits);
                        temp = array[0];
                        errorsTWOclustersArray[id_i] = temp;
                    } else
                        temp = errorsTWOclustersArray[id_i];

                    copy_container->tableClusters[*it]->changePoint(point, 1);
                    //                    copy_container->tableClusters[*it]->sortDiagCov();
                    copy_container->tableClusters[*it]->divide_nth_element(size[0], bits);

                    copy_container->errorsTWOclusters(array, l, *it, size[0], bits);

                    if (array[0] < (c + temp)) {
                        j = *it;
                        c = array[0] - temp;
                        tempMemory = array[1];
                        error2cluster = array[0];
                    }
                }
                if (c == -std::numeric_limits<double>::infinity()) break;
            }

            if (j != l) {
                T = false;
                stop = m + 1; // stop <- nastapila zmiana
                check = false;
                group[m] = j;

                for (int kk = 0; kk < howClusters; ++kk) {
                    if (l != kk) {
                        id_i = index(l, kk, howClusters);
                        errorsTWOclustersArray[id_i] = -1.0;
                    }
                    if (j != kk) {
                        id_i = index(j, kk, howClusters);
                        errorsTWOclustersArray[id_i] = -1.0;
                    }
                }
                id_i = index(l, j, howClusters);
                errorsTWOclustersArray[id_i] = error2cluster;

                this->tableClusters[l]->weight = copy_container->tableClusters[l]->weight;
                this->tableClusters[j]->weight = copy_container->tableClusters[j]->weight;
                for (int ii = 0; ii < size[1]; ++ii) {
                    this->tableClusters[l]->mean[ii] = copy_container->tableClusters[l]->mean[ii];
                    this->tableClusters[j]->mean[ii] = copy_container->tableClusters[j]->mean[ii];
                    this->tableClusters[l]->diagCov[ii] = copy_container->tableClusters[l]->diagCov[ii];
                    this->tableClusters[j]->diagCov[ii] = copy_container->tableClusters[j]->diagCov[ii];
                    this->tableClusters[l]->sortedDiagCov[ii] = copy_container->tableClusters[l]->sortedDiagCov[ii];
                    this->tableClusters[j]->sortedDiagCov[ii] = copy_container->tableClusters[j]->sortedDiagCov[ii];
                }

                this->tableClusters[l]->memory =
                        this->tableClusters[l]->memory + this->tableClusters[j]->memory - tempMemory;
                this->tableClusters[j]->memory = tempMemory;
            }
        }

        // Delete cluster which have a small number of points
        if (bits != 0) {
            min_weight = size[0];
            for (it = activeClusters.begin(); it != activeClusters.end(); it++)
                if (min_weight >= this->tableClusters[*it]->weight) {
                    min_it = it;
                    min_weight = this->tableClusters[*it]->weight;
                }
            if (this->tableClusters[*min_it]->unassign(size[0])) {
                activeClusters.erase(min_it);
                T = false;
            }
            if (activeClusters.empty()) {
                printf("Delete all clusters. Small number of points in relation to the dimension of the data.\t");
                T = true;
            }
        }
    }
    delete copy_container;
    delete[] point;
    delete[] errorsTWOclustersArray;

    //update dimensions and memory of clusters
    if (activeClusters.size() == 1) {
        this->tableClusters[activeClusters.front()]->dim =
                this->tableClusters[activeClusters.front()]->memory / size[0];
        if (bits != 0)
            this->tableClusters[activeClusters.front()]->dim /= bits;
    } else {
        std::vector<int> vec;
        vec.reserve(activeClusters.size());
        for (auto val: activeClusters)
            vec.push_back(val);
//        for (it = activeClusters.begin(); it != activeClusters.end(); ++it)
//            vec.push_back(*it);
        while (vec.size() > 1) {
            it = vec.begin();
            i = this->updateDim(*it, *(it + 1), size[0], bits);
            vec.erase(it + i);
        }
        vec.clear();
    }

    this->error = 0.0;
    if (bits != 0) {
        for (i = this->size - 1; i >= 0; i--) {
            outside = true;
            for (auto rit = activeClusters.rbegin(); rit != activeClusters.rend(); ++rit)
                if (i == *rit) {
                    outside = false;
                    break;
                }
            if (outside) {
                this->tableClusters.erase(this->tableClusters.begin() + i);
                this->size -= 1;
            } else {
                this->error += this->tableClusters[i]->errorFUN(size[0], bits);
            }
        }
    } else {
        for (i = this->size - 1; i >= 0; i--)
            this->error += this->tableClusters[i]->errorFUN(size[0], bits);
    }
}

/**
 *
 * @param group (<i><b>size_t*</b></i>) - list of id clusters (values: 0,1,...,howClusters-1;  <i>row-major order</i>)
 * @param size (<i><b>size_t[]</b></i>) - size of data
 * @param data (<i><b>double*</b></i>) - 2-dimensional array, which save data <i>row-major order</i>
 * @param howClusters (<i><b>size_t</b></i>) - determines how much we want to create clusters
 * @param allMemory (<i><b>double</b></i>) -  general memory we have at our disposal
 * @param bits (<i><b>size_t</i></b>) - number of bits needed to memorize one scalar
 * @param iteration (<i><b>size_t</i></b>) - number of iterations
 */
void
ContainerClusters::Hartigan(int *group, const int size[], const double *data, const unsigned int howClusters,
                            const double allMemory, const int iteration, const int bits) {

//    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
//    std::mt19937 gen(seed);
////    std::random_device rd;
////    std::mt19937 gen(rd());
//    std::uniform_int_distribution<> dis(0, howClusters - 1);

    auto *TEMPgroup = new int[size[0]];
    auto *temp = new ContainerClusters();

    std::vector<int> TEMPactiveClusters(howClusters);

    std::vector<int> activeClusters;
    if (bits != 0)
        activeClusters.reserve(howClusters);

    int i, j;

    for (i = 0; i < iteration; ++i) {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dis(0, howClusters - 1);

        for (j = 0; j < howClusters; ++j)
            TEMPactiveClusters.push_back(j);
        for (j = 0; j < size[0]; ++j)
            TEMPgroup[j] = dis(gen);

        temp->stepHartigan(TEMPgroup, TEMPactiveClusters, size, data, allMemory, bits);

        if (temp->error < this->error) {
            *this = *temp;
            for (j = 0; j < size[0]; ++j)
                group[j] = TEMPgroup[j];
            if (bits != 0) activeClusters.assign(TEMPactiveClusters.begin(), TEMPactiveClusters.end());
        }

        TEMPactiveClusters.clear();
        temp->size = 0;
        temp->resetError();
        temp->clear();
    }

    if (bits != 0 && this->size != howClusters) {
        j = 0;
        for (i = 0; i < size[0]; ++i) {
            for (const int &val: activeClusters) {
                if (group[i] == val) {
                    group[i] = j;
                    break;
                }
                j++;
            }
            j = 0;
        }
    }

    delete temp;
    delete[] TEMPgroup;
    if (bits != 0) activeClusters.clear();
}


/**
 *
 *
 * @param group (<i><b>int*</b></i>) - tablica (o wartosciach: 0,1,...,howClusters-1;  <i>row-major order</i>) okreslajaca poczatkowe polozenie poszczegolnych punktow ze zbioru 'data'
 * @param size (<i><b>int[]</b></i>) - tablica dwuwymiarowa okrslajaca (odpowiednio): wymiar w jakim sa przedstawione dane oraz ilosc danych
 * @param data (<i><b>double*</b></i>) - dane <i>row-major order</i>
 * @param howClusters (<i><b>int</b></i>) - okresla ile chcemy stworzyc klastrow
 * @param allMemory (<i><b>double</b></i>) - ogolna pamiec jaka mamy do dyspozycji
 * @param n_threads (<i><b>int</i></b>) - liczba watkow
 * @param iteration (<i><b>int</i></b>) - zmienna okresla ilosc iteracji
 * @param bits (<i><b>int</i></b>) - bity to ilosc bitow potrzebna do pamietania jednego skalara
 */
void
ContainerClusters::Hartigan_parallel(int *group, const int *size, const double *data,
                                     const unsigned int howClusters, const double allMemory, unsigned int n_threads,
                                     int iteration,
                                     const int bits) {

    unsigned int numCPU = std::thread::hardware_concurrency();

    n_threads = (n_threads <= numCPU) ? n_threads : numCPU;
    std::vector<std::thread> threads;
//    std::vector<std::unique_ptr<ContainerClusters>> array_containerCL;
    std::vector<ContainerClusters *> array_containerCL;

    auto *TEMPgrups = new int[n_threads * size[0]];

    int id_min_error;
    double min_value; // this->error
    int i, j = 0, n_iteration_thread = (int) std::ceil(iteration / double(n_threads));

    while (iteration > 0) {
        i = (iteration > n_iteration_thread) ? n_iteration_thread : iteration;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        array_containerCL.emplace_back(new ContainerClusters());
        threads.emplace_back(
                std::thread(&ContainerClusters::Hartigan, array_containerCL.back(), TEMPgrups + size[0] * j++,
                            size, data, howClusters, allMemory, i, bits));
        iteration -= i;
    }

    for (auto &thread : threads)
//    if (thread.joinable())
        thread.join();

    id_min_error = -1;
    min_value = this->error;
    for (i = 0; i < array_containerCL.size(); i++)
        if (min_value > array_containerCL[i]->getError()) {
            id_min_error = i;
            min_value = array_containerCL[i]->getError();
        }
    if (id_min_error != -1) {
        *this = *array_containerCL[id_min_error];
        for (j = 0; j < size[0]; ++j)
            group[j] = TEMPgrups[id_min_error * size[0] + j];
    }
    threads.clear();
    for (ContainerClusters *con: array_containerCL)
        delete con;
    array_containerCL.clear();

    delete[] TEMPgrups;
}


/**
 *
 * @param data
 * @param group
 * @param size
 * @param howClusters
 * @param degreeOfCompression
 * @param iteration
 * @param bits
 * @param allMemory
 */
void ContainerClusters::run(double *data, int *group, const int size[], const unsigned int howClusters,
                            const double degreeOfCompression, const int iteration, const int bits) {
    double allMemory = (bits == 0) ? degreeOfCompression * size[0] * size[1] : degreeOfCompression * bits * size[0] *
                                                                               size[1];
    this->Hartigan(group, size, data, howClusters, allMemory, iteration, bits);
    //    this->compression(data, size, group);
}


void
ContainerClusters::Hartigan_init_group(int *group, const int size[], const double *data, const unsigned int howClusters,
                                       std::vector<double> &comp_ratio, unsigned int n_threads, const int bits,
                                       std::string &path2dir) {

    unsigned int numCPU = std::thread::hardware_concurrency();

    n_threads = (n_threads <= numCPU) ? n_threads : numCPU;
    std::vector<std::thread> threads;
    std::vector<ContainerClusters *> array_containerCL;

    auto *TEMPgrups = new int[n_threads * size[0]];

    int id_min_error;
    double min_value; // this->error
    auto iteration = static_cast<int>(comp_ratio.size());
    int i, j, k, n_iteration_thread = (int) std::ceil(iteration / double(n_threads));

    auto launch = [&comp_ratio](ContainerClusters *con, int *bestgroup, const int *gr, const int *size_,
                                const double *data_,
                                unsigned int howClusters_, int bits_, int left, int right,
                                std::string &path2dir) -> void {
        double val_mem_mse;
        int ii, jj, kk;
        std::ofstream plik1, plik2;
        std::string fileName;

        auto *TEMPgroup = new int[size_[0]];
        auto *temp = new ContainerClusters();

        std::vector<int> TEMPactiveClusters(howClusters_);

        for (ii = left; ii < right; ++ii) {
            val_mem_mse = comp_ratio[ii] * size_[0] * size_[1];
            if (bits_ != 0) val_mem_mse *= bits_;

            for (jj = 0; jj < howClusters_; ++jj)
                TEMPactiveClusters.push_back(jj);
            for (jj = 0; jj < size_[0]; ++jj)
                TEMPgroup[jj] = gr[jj];

            temp->stepHartigan(TEMPgroup, TEMPactiveClusters, size_, data_, val_mem_mse, bits_);

            if (bits_ != 0 && temp->getSize() != howClusters_) {
                jj = 0;
                for (kk = 0; kk < size_[0]; ++kk) {
                    for (const int &val: TEMPactiveClusters) {
                        if (TEMPgroup[kk] == val) {
                            TEMPgroup[kk] = jj;
                            break;
                        }
                        jj++;
                    }
                    jj = 0;
                }
            }

            if (temp->getSize() != 0) {
                val_mem_mse = 0.0;

                for (kk = 0; kk < size_[0]; kk++)
                    for (jj = 0; jj < size_[1]; jj++)
                        val_mem_mse += std::pow(
                                std::abs(
                                        data_[kk * size_[1] + jj] - temp->getContainer()[TEMPgroup[kk]]->getMean()[jj]),
                                2);

                fileName = path2dir + "PMC_clustering_" + std::to_string(int(comp_ratio[ii] * 100)) + ".txt";
                plik1.open(fileName.c_str(), std::ios::out);
                if (plik1.good()) {
                    for (jj = 0; jj < size_[0]; jj++)
                        plik1 << TEMPgroup[jj] + 1 << "\n";
                    plik1.close();
                } else std::cout << "Dostep do pliku \"" << fileName << "\" zostal zabroniony!" << std::endl;

                fileName = path2dir + "PMC_description_" + std::to_string(int(comp_ratio[ii] * 100)) + ".txt";
                plik2.open(fileName.c_str(), std::ios::out);
                if (plik2.good()) {
                    plik2 << "Pamiec:\n";
                    for (jj = 0; jj < temp->getSize(); jj++)
                        plik2 << temp->getContainer()[jj]->getMemory() << "\n";

                    plik2 << "\nWymiary:\n";
                    for (jj = 0; jj < temp->getSize(); jj++)
                        plik2 << temp->getContainer()[jj]->getDim() << "\n";

                    plik2 << "\nWagi:\n";
                    for (jj = 0; jj < temp->getSize(); jj++)
                        plik2 << temp->getContainer()[jj]->getWeight() << "\n";

                    plik2 << "\n\nWYNIK:\n";
                    jj = 1;
                    for (auto &cl: temp->getContainer()) {
                        plik2 << "dim" << jj << " = " << cl->getDim() << std::endl;
                        plik2 << "mean" << jj << " = np.array([";
                        for (auto &val: cl->getMean())
                            plik2 << val << ",";
                        plik2 << "])\n\n";
                        plik2 << "base" << jj << " = np.array([";
                        for (auto &val: cl->getDiagCov())
                            plik2 << val << ",";
                        plik2 << "])\n\n";
                        jj++;
                    }

//                    plik2 << "\n\nWYNIK:\n";
//                    for (auto &cl: temp->getContainer())
//                        plik2 << *cl << std::endl;
//
//                    plik2 << "\nE = " << temp->getError() << "\n";
//                    plik2 << "Error/n_sample = " << temp->getError() / size_[0] << std::endl;
//                    plik2 << "Error/L2 = " << temp->getError() / val_mem_mse << std::endl;
                    plik2.close();
                } else std::cout << "Dostep do pliku \"" << fileName << "\" zostal zabroniony!" << std::endl;
            }

            if (temp->error < con->error) {
                *con = *temp;
                for (jj = 0; jj < size_[0]; ++jj)
                    bestgroup[jj] = TEMPgroup[jj];
            }

            TEMPactiveClusters.clear();
            temp->size = 0;
            temp->resetError();
            temp->clear();
        }

        delete temp;
        delete[] TEMPgroup;
    };

    j = 0;
    k = 0;
    while (iteration > 0) {
        i = (iteration > n_iteration_thread) ? n_iteration_thread : iteration;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        array_containerCL.emplace_back(new ContainerClusters());

        threads.emplace_back(
                std::thread(launch, std::ref(array_containerCL.back()), TEMPgrups + size[0] * k++, group, size, data,
                            howClusters, bits, j, j + i, std::ref(path2dir)));
        j += i;
        iteration -= i;
    }

    for (auto &thread : threads)
//    if (thread.joinable())
        thread.join();

    id_min_error = -1;
    min_value = this->error;
    for (i = 0; i < array_containerCL.size(); i++)
        if (min_value > array_containerCL[i]->getError()) {
            id_min_error = i;
            min_value = array_containerCL[i]->getError();
        }
    if (id_min_error != -1) {
        *this = *array_containerCL[id_min_error];
        for (j = 0; j < size[0]; ++j)
            group[j] = TEMPgrups[id_min_error * size[0] + j];
    }
    threads.clear();
    for (ContainerClusters *con: array_containerCL)
        delete con;
    array_containerCL.clear();

    delete[] TEMPgrups;
}


void
ContainerClusters::Hartigan_parallel_ratio(int iterations, const int size[], const double *data,
                                           unsigned int howClusters, unsigned int n_threads, const int bits,
                                           int left_ratio, const int right_ratio, std::string &path2dir) {

    auto launch = [](int iters, const int *size_, const double *data_, unsigned int howClusters_, int bits_,
                     int left, int right, std::string &path2dir) -> void {

        double val_mem_mse;
        int ii, jj, kk;
        std::ofstream plik1, plik2;
        std::string fileName;

        auto *TEMPgroup = new int[size_[0]];
        auto *temp = new ContainerClusters();

        for (ii = left; ii < right; ++ii) {
            val_mem_mse = ii / 100. * size_[0] * size_[1];
            if (bits_ != 0) val_mem_mse *= bits_;

            temp->Hartigan(TEMPgroup, size_, data_, howClusters_, val_mem_mse, iters, bits_);

            if (temp->getSize() != 0) {
                val_mem_mse = 0.0;

                for (kk = 0; kk < size_[0]; kk++)
                    for (jj = 0; jj < size_[1]; jj++)
                        val_mem_mse += std::pow(
                                std::abs(
                                        data_[kk * size_[1] + jj] - temp->getContainer()[TEMPgroup[kk]]->getMean()[jj]),
                                2);

                fileName = path2dir + "PMC_clustering_" + std::to_string(ii) + ".txt";
                plik1.open(fileName.c_str(), std::ios::out);
                if (plik1.good()) {
                    for (jj = 0; jj < size_[0]; jj++)
                        plik1 << TEMPgroup[jj] + 1 << "\n";
                    plik1.close();
                } else std::cout << "Dostep do pliku \"" << fileName << "\" zostal zabroniony!" << std::endl;

                fileName = path2dir + "PMC_description_" + std::to_string(ii) + ".txt";
                plik2.open(fileName.c_str(), std::ios::out);
                if (plik2.good()) {
                    plik2 << "Pamiec:\n";
                    for (jj = 0; jj < temp->getSize(); jj++)
                        plik2 << temp->getContainer()[jj]->getMemory() << "\n";

                    plik2 << "\nWymiary:\n";
                    for (jj = 0; jj < temp->getSize(); jj++)
                        plik2 << temp->getContainer()[jj]->getDim() << "\n";

                    plik2 << "\nWagi:\n";
                    for (jj = 0; jj < temp->getSize(); jj++)
                        plik2 << temp->getContainer()[jj]->getWeight() << "\n";

                    plik2 << "\n\nWYNIK:\n";
                    jj = 1;
                    for (auto &cl: temp->getContainer()) {
                        plik2 << "dim" << jj << " = " << cl->getDim() << std::endl;
                        plik2 << "mean" << jj << " = np.array([";
                        for (auto &val: cl->getMean())
                            plik2 << val << ",";
                        plik2 << "])\n\n";
                        plik2 << "base" << jj << " = np.array([";
                        for (auto &val: cl->getDiagCov())
                            plik2 << val << ",";
                        plik2 << "])\n\n";
                        jj++;
                    }

//                    plik2 << "\n\nWYNIK:\n";
//                    for (auto &cl: temp->getContainer())
//                        plik2 << *cl << std::endl;
//
//                    plik2 << "\nE = " << temp->getError() << "\n";
//                    plik2 << "Error/n_sample = " << temp->getError() / size_[0] << std::endl;
//                    plik2 << "Error/L2 = " << temp->getError() / val_mem_mse << std::endl;
                    plik2.close();
                } else std::cout << "Dostep do pliku \"" << fileName << "\" zostal zabroniony!" << std::endl;
            }

            temp->size = 0;
            temp->resetError();
            temp->clear();
        }

        delete temp;
        delete[] TEMPgroup;
    };

    unsigned int numCPU = std::thread::hardware_concurrency();

    n_threads = (n_threads <= numCPU) ? n_threads : numCPU;
    std::vector<std::thread> threads;

    auto iteration = std::abs(right_ratio - left_ratio);
    int i, n_iteration_thread = (int) std::ceil(iteration / double(n_threads));

    while (right_ratio - left_ratio > 0) {
        i = (iteration > n_iteration_thread) ? n_iteration_thread : iteration;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        threads.emplace_back(
                std::thread(launch, iterations, size, data, howClusters, bits, left_ratio, left_ratio + i,
                            std::ref(path2dir)));
        left_ratio += i;
        iteration -= i;
    }

    for (auto &thread : threads)
//    if (thread.joinable())
        thread.join();

    threads.clear();
}