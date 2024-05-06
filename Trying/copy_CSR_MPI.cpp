#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <cstring>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;
typedef Eigen::VectorXd Vec;
typedef Eigen::Triplet<double> Triplet;

class MapMatrix {
public:
    typedef std::pair<int, int> IndexPair;
    std::map<IndexPair, double> data;
    int nbrow, nbcol;

    MapMatrix(int nr, int nc) : nbrow(nr), nbcol(nc) {}

    double& operator()(int row, int col) {
        return data[{row, col}];
    }

    // Convert to CSR format
    SpMat toCSR() const {
        std::vector<Triplet> triplets;
        triplets.reserve(data.size());
        for (const auto& item : data) {
            triplets.emplace_back(item.first.first, item.first.second, item.second);
        }
        SpMat mat(nbrow, nbcol);
        mat.setFromTriplets(triplets.begin(), triplets.end());
        return mat;
    }
};

class DistributedMatrix {
public:
    SpMat localMatrix;
    Eigen::SimplicialLLT<SpMat> localCholesky;
    int globalRows;
    int localRows;
    int rank;
    int size;

    DistributedMatrix(int globalRows, int localRows, int rank, int size)
        : localMatrix(localRows, globalRows), globalRows(globalRows), localRows(localRows), rank(rank), size(size) {}

    void insert(int row, int col, double value) {
        assert(row < localRows);
        localMatrix.insert(row, col) = value;
    }

    void finalize() {
        localMatrix.makeCompressed();
        for (int i = 0; i < localRows; ++i) {
            if (localMatrix.coeffRef(i, i) == 0) {
                std::cerr << "Rank: " << rank << " Warning: Diagonal element at " << i << " is zero. Adjusting to small non-zero value.\n";
                localMatrix.coeffRef(i, i) = 1e-10;
            }
        }
        // Compute the Cholesky decomposition directly on the symmetric matrix
        localCholesky.compute(localMatrix);
        if (localCholesky.info() != Eigen::Success) {
            std::cerr << "Decomposition failed!\n";
        }
    }

    

    Vec multiply(const Vec& x) const {
        assert(x.size() == globalRows);
        Vec local_x = x.segment(rank * localRows, localRows);
        Vec local_result = localMatrix * local_x;
        Vec global_result(globalRows);
        MPI_Allreduce(local_result.data(), global_result.data(), localRows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return global_result;
    }

    Vec applyPreconditioner(const Vec& r) const {
        return localCholesky.solve(r);
    }
};

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], option) == 0) {
            return std::stoi(argv[i + 1]);
        }
    }
    return default_value;
}

double dotProduct(const Vec& u, const Vec& v, MPI_Comm comm) {
    double localDot = u.dot(v);
    double globalDot = 0;
    MPI_Allreduce(&localDot, &globalDot, 1, MPI_DOUBLE, MPI_SUM, comm);
    return globalDot;
}

double norm(const Vec& u, MPI_Comm comm) {
    return std::sqrt(dotProduct(u, u, comm));
}

Vec conjugateGradient(const DistributedMatrix& A, const Vec& b, double tol, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    Vec x_global(A.globalRows);
    x_global.setZero();
    Vec r = b - A.multiply(x_global);
    Vec z = A.applyPreconditioner(r);
    Vec p = z;
    double rsold = dotProduct(z, r, comm);
    int iteration = 0;
    while (true) {
        Vec Ap = A.multiply(p);
        double pAp = dotProduct(p, Ap, comm);
        double alpha = rsold / pAp;
        x_global += alpha * p;
        Vec r_new = r - alpha * Ap;
        double residualNorm = std::sqrt(dotProduct(r_new, r_new, comm));
        if (residualNorm < tol) {
            if (rank == 0) {
                std::cout << "Converged after " << iteration << " iterations with residual norm " << residualNorm << ".\n";
            }
            break;
        }
        Vec z_new = A.applyPreconditioner(r_new);
        double rsnew = dotProduct(z_new, r_new, comm);
        double beta = rsnew / rsold;
        p = z_new + beta * p;
        r = r_new;
        z = z_new;
        rsold = rsnew;
        iteration++;
    }
    return x_global;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int globalRows = find_int_arg(argc, argv, "-N", 1000);
    int localRows = globalRows / size;

    MapMatrix localMapMatrix(globalRows, localRows);

    for (int i = 0; i < localRows; ++i) {
        int globalIndex = rank * localRows + i;
        localMapMatrix(i, globalIndex) = 4;  // Diagonal dominance
        if (globalIndex > 0) localMapMatrix(i, globalIndex - 1) = -1;
        if (globalIndex < globalRows - 1) localMapMatrix(i, globalIndex + 1) = -1;
    }

    DistributedMatrix A(globalRows, localRows, rank, size);
    A.localMatrix = localMapMatrix.toCSR();  // Convert MapMatrix to CSR and set to DistributedMatrix
    A.finalize();

    Vec b(localRows);
    b.setOnes();

    double tol = 1e-6;
    Vec x_global = conjugateGradient(A, b, tol, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Solution computed successfully.\n";
    }

    MPI_Finalize();
    return 0;
}
