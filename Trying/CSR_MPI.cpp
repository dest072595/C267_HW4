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

class DistributedMatrix {
public:
    SpMat localMatrix;
    Eigen::SimplicialLLT<SpMat> localCholesky;  // Cholesky decomposition for block preconditioner
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
                localMatrix.coeffRef(i, i) = 1e-10;  // Adjust to a small non-zero value
            }
        }

        // Compute the Cholesky decomposition of the local matrix block
        localCholesky.compute(localMatrix.block(0, rank * localRows, localRows, localRows));
    }

    Vec multiply(const Vec& x) const {
        assert(x.size() == globalRows);  // Ensure the global vector size matches expected global rows
        Vec local_x = x.segment(rank * localRows, localRows);  // Extract the local segment of the vector
        Vec local_result = localMatrix * local_x;  // Perform local multiplication

        Vec global_result(globalRows);
        MPI_Allreduce(local_result.data(), global_result.data(), localRows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        return global_result;
    }

    Vec applyPreconditioner(const Vec& r) const {
        assert(r.size() == localRows);  // Residual vector needs to be the correct size
        return localCholesky.solve(r);  // Solve using Cholesky factorization
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

    Vec x_global(A.globalRows);  // Initial guess (zero vector)
    x_global.setZero();

    Vec r = b - A.multiply(x_global);  // Initial residual
    Vec z = A.applyPreconditioner(r);  // Apply preconditioner to the initial residual
    Vec p = z;  // Initial direction vector is the preconditioned residual

    double rsold = dotProduct(z, r, comm);  // Initial dot product of preconditioned residual and residual

    int iteration = 0;
    while (true) {
        Vec Ap = A.multiply(p);  // Matrix-vector product
        double pAp = dotProduct(p, Ap, comm);  // Dot product of direction vector and A*p
        double alpha = rsold / pAp;  // Step size

        x_global += alpha * p;  // Update the solution vector
        Vec r_new = r - alpha * Ap;  // Compute the new residual

        double residualNorm = std::sqrt(dotProduct(r_new, r_new, comm));
        if (residualNorm < tol) {  // Check convergence
            if (rank == 0) {
                std::cout << "Converged after " << iteration << " iterations with residual norm " << residualNorm << ".\n";
            }
            break;
        }

        Vec z_new = A.applyPreconditioner(r_new);  // Apply preconditioner to the new residual
        double rsnew = dotProduct(z_new, r_new, comm);  // Dot product of new preconditioned residual and original residuals

        double beta = rsnew / rsold;  // Compute beta for CG update
        p = z_new + beta * p;  // Update the direction vector

        r = r_new;  // Update residual
        z = z_new;  // Update preconditioned residual
        rsold = rsnew;  // Update dot product result

        iteration++;
    }

    return x_global;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "Rank: " << rank << ", running with " << size << " total processes.\n";

    int globalRows = find_int_arg(argc, argv, "-N", -1);
    if (globalRows <= 0) {
        if (rank == 0) {
            std::cerr << "Invalid or missing matrix size provided. Use -N <size> to specify a valid matrix size." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = std::sqrt(globalRows);
    if (n * n != globalRows || globalRows % size != 0) {
        if (rank == 0) {
            std::cerr << "Global rows should be a perfect square and divisible by the number of processes!" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int localRows = globalRows / size;
    DistributedMatrix A(globalRows, localRows, rank, size);
    std::cout << "Rank: " << rank << ", handles " << localRows << " local rows of " << globalRows << " global rows.\n";

    for (int i = 0; i < localRows; ++i) {
        int globalIndex = rank * localRows + i;
        int x = globalIndex % n;
        int y = globalIndex / n;

        if (x > 0) A.insert(i, globalIndex - 1, -1);
        if (x < n - 1) A.insert(i, globalIndex + 1, -1);
        if (y > 0) A.insert(i, globalIndex - n, -1);
        if (y < n - 1) A.insert(i, globalIndex + n, -1);
        A.insert(i, globalIndex, 4);  // Ensure diagonal is correctly set
    }

    A.finalize();

    Vec b(localRows);
    b.setOnes();

    double tol = 1e-6;

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Main computation block: Run Conjugate Gradient solver
    Vec x_global = conjugateGradient(A, b, tol, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double execution_time = end_time - start_time;

    if (rank == 0) {
        std::cout << "Execution time with " << size << " MPI processes: " << execution_time << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
