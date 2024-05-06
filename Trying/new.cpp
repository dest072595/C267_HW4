#include <iostream>
#include <vector>
#include <cassert>
#include <mpi.h>
#include <Eigen/Sparse>
#include <cmath>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::VectorXd Vec;
typedef Eigen::Triplet<double> T;

class DistributedMatrix {
    int globalRows, localRows, rank;
    SpMat localMatrix;
    SpMat localPreconditioner;

public:
    DistributedMatrix(int globalRows, int localRows, int rank)
        : globalRows(globalRows), localRows(localRows), rank(rank),
          localMatrix(localRows, globalRows), localPreconditioner(localRows, localRows) {}

    void insert(int localRow, int globalCol, double value) {
        assert(localRow >= 0 && localRow < localRows);
        assert(globalCol >= 0 && globalCol < globalRows);
        localMatrix.insert(localRow, globalCol) = value;
    }

    void finalize() {
        localMatrix.makeCompressed();
        std::vector<T> diagonalTriplets;

        for (int i = 0; i < localRows; ++i) {
            int globalIndex = rank * localRows + i;
            double diagonalValue = std::max(localMatrix.coeff(i, globalIndex), 1e-10);
            diagonalTriplets.emplace_back(i, i, 1.0 / diagonalValue);
        }

        localPreconditioner.setFromTriplets(diagonalTriplets.begin(), diagonalTriplets.end());
    }

    Vec multiply(const Vec& x) const {
        assert(x.size() == globalRows);
        Vec localX = x.segment(rank * localRows, localRows);
        Vec localResult = localMatrix * localX;

        Vec globalResult = Vec::Zero(globalRows);
        MPI_Allreduce(localResult.data(), globalResult.data(), globalRows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        return globalResult;
    }

    Vec applyPreconditioner(const Vec& r) const {
        assert(r.size() == localRows);
        return localPreconditioner * r;
    }

    int getGlobalRows() const {
        return globalRows;
    }

    int getLocalRows() const {
        return localRows;
    }
};

double parallelDotProduct(const Vec& u, const Vec& v, MPI_Comm comm) {
    assert(u.size() == v.size());
    double localDot = u.dot(v);
    double globalDot = 0;
    MPI_Allreduce(&localDot, &globalDot, 1, MPI_DOUBLE, MPI_SUM, comm);
    return globalDot;
}

double parallelNorm(const Vec& u, MPI_Comm comm) {
    return std::sqrt(parallelDotProduct(u, u, comm));
}

Vec conjugateGradient(const DistributedMatrix& A, const Vec& b, double tol, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int globalRows = A.getGlobalRows();
    Vec x_global = Vec::Zero(globalRows);

    Vec r = b - A.multiply(x_global);
    Vec z = A.applyPreconditioner(r);
    Vec p = z;

    double rsold = parallelDotProduct(z, r, comm);

    // Debug initial residual
    double initialResidualNorm = parallelNorm(r, comm);
    if (rank == 0) std::cout << "Initial Residual Norm: " << initialResidualNorm << std::endl;

    int iteration = 0;
    while (true) {
        Vec Ap = A.multiply(p);
        double pAp = parallelDotProduct(p, Ap, comm);
        double alpha = rsold / pAp;

        x_global += alpha * p;
        Vec r_new = r - alpha * Ap;

        double residualNorm = parallelNorm(r_new, comm);
        if (rank == 0) std::cout << "Iteration " << iteration << ": Residual Norm = " << residualNorm << std::endl;

        if (residualNorm < tol) {
            if (rank == 0) std::cout << "Converged after " << iteration << " iterations.\n";
            break;
        }

        Vec z_new = A.applyPreconditioner(r_new);
        double rsnew = parallelDotProduct(z_new, r_new, comm);

        double beta = rsnew / rsold;
        p = z_new + beta * p;

        r = r_new;
        rsold = rsnew;

        iteration++;
    }

    return x_global;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::strcmp(argv[i], option) == 0) {
            return std::stoi(argv[i + 1]);
        }
    }
    return default_value;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int globalRows = find_int_arg(argc, argv, "-N", 64);
    assert(globalRows % size == 0);

    int localRows = globalRows / size;

    DistributedMatrix A(globalRows, localRows, rank);
    for (int i = 0; i < localRows; ++i) {
        int globalIndex = rank * localRows + i;

        A.insert(i, globalIndex, 4.0);

        if (globalIndex - 1 >= 0) A.insert(i, globalIndex - 1, -1.0);
        if (globalIndex + 1 < globalRows) A.insert(i, globalIndex + 1, -1.0);
    }

    A.finalize();

    Vec b = Vec::Ones(globalRows);

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    Vec x_global = conjugateGradient(A, b, 1e-6, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "Wall time: " << MPI_Wtime() - startTime << std::endl;

    MPI_Finalize();
    return 0;
}
