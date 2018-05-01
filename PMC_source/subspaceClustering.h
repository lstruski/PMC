//
// Created by lukasz on 19.03.17.
//

#ifndef NEW_PROJECT_SUBSPACECLUSTERING_H
#define NEW_PROJECT_SUBSPACECLUSTERING_H


#include <vector>
#include <ostream>

void data_param(std::string filename, int &n_samples, int &dim, char separator = ',');

void readfile(double *data, std::string filename, char separator = ',');

double *readData(std::string filename, int &n_samples, int &dim, char separator = ',');


namespace subspaceClusteringParallel {

    class Cluster;

    /**
     * @class ContainerClusters
     *
     */
    class ContainerClusters {
        unsigned int size;
        double error;
        std::vector<Cluster *> tableClusters;

    public:
        ContainerClusters();

        ContainerClusters(const ContainerClusters &orig);

        virtual ~ContainerClusters();

        unsigned int getSize() const;

        double getError() const;

        void resetError();

        const std::vector<Cluster *> &getContainer() const;

        void Hartigan(int *group, const int size[], const double *data, unsigned int howClusters,
                      double allMemory, int iteration = 5, int bits = 0);

        void Hartigan_parallel(int *group, const int size[], const double *data, unsigned int howClusters,
                               double allMemory, unsigned int n_threads, int iteration = 5, int bits = 0);

        void Hartigan_init_group(int *group, const int size[], const double *data, unsigned int howClusters,
                                 std::vector<double> &comp_ratio, unsigned int n_threads, int bits,
                                 std::string &path2dir);

        void Hartigan_parallel_ratio(int iterations, const int size[], const double *data, unsigned int howClusters,
                                     unsigned int n_threads, int bits, int left, int right, std::string &path2dir);

        void run(double *data, int *group, const int size[], unsigned int howClusters,
                 double degreeOfCompression, int iteration = 5, int bits = 0);

    private:
        void clear();

        ContainerClusters &operator=(const ContainerClusters &orig);

        void stepHartigan(int *group, std::vector<int> &activeClusters, const int size[],
                          const double *data, double allMemory, int bits = 0);

        void
        createCluster(const int size[], const double *data, const int *group, unsigned int howClusters,
                      double allMemory, int bits = 0);

        void
        errorsTWOclusters(double *array, int idCluster1, int idCluster2, int totalWeight,
                          int bits = 0) const;

        int updateDim(int idCluster1, int idCluster2, int totalWeight,
                      int bits = 0);
    };

    /**
     * @class Cluster
     *
     */
    class Cluster {
        double dim;
        std::vector<double> mean;
        std::vector<double> diagCov;
        std::vector<double> sortedDiagCov;
        int weight;
        double memory;

        friend class ContainerClusters;

        friend std::ostream &operator<<(std::ostream &out, const Cluster &c);

    public:
        static int N;

        explicit Cluster(int N);

        Cluster(const Cluster &orig);

        virtual ~Cluster();

        Cluster &operator=(const Cluster &orig);

        double getDim() const;

        int getWeight() const;

        double getMemory() const;

        const std::vector<double> &getMean() const;

        const std::vector<double> &getDiagCov() const;

    private:

        void sortDiagCov();

        void divide_nth_element(int totalWeight, int bits = 0);

        void changePoint(const double *point, int weightPoint);

        double err(double factor) const;

        double errorFUN(int totalWeight, int bits = 0) const;

        bool unassign(int totalWeight, double toleranceFactor = 0.02) const;
    };

    std::ostream &operator<<(std::ostream &out, const Cluster &c);
}


#endif //PMC_SUBSPACECLUSTERINGPARALLEL_H
