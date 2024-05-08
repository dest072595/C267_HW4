#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <cstring>
#include <map>
#include <chrono>

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
        assert(row >= 0 && row < nbrow && col >= 0 && col < nbcol);
        return data[{row, col}];
    }

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
    int globalRows, localRows, rank, size;

    DistributedMatrix(int gr, int lr, int r, int s)
        : localMatrix(lr, lr), globalRows(gr), localRows(lr), rank(r), size(s) {}

    void finalize() {
        localMatrix.makeCompressed();
        for (int i = 0; i < localRows; ++i) {
            if (localMatrix.coeffRef(i, i) <= 0) {
                localMatrix.coeffRef(i, i) = 1e-10;  // Ensure positive definiteness
            }
        }
        localCholesky.compute(localMatrix);
        if (localCholesky.info() != Eigen::Success) {
            std::cerr << "Cholesky decomposition failed at rank " << rank << "." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    Vec multiply(const Vec& local_x) const {
        return localMatrix * local_x;
    }

    Vec applyPreconditioner(const Vec& r) const {
        return localCholesky.solve(r);
    }
};

double dotProduct(const Vec& u, const Vec& v, MPI_Comm comm) {
    double localDot = u.dot(v);
    double globalDot = 0;
    MPI_Allreduce(&localDot, &globalDot, 1, MPI_DOUBLE, MPI_SUM, comm);
    return globalDot;
}

double norm(const Vec& u, MPI_Comm comm) {
    return std::sqrt(dotProduct(u, u, comm));
}


Vec conjugateGradient(const DistributedMatrix& A, const Vec& global_b, double tol, MPI_Comm comm, int& iteration) {
    auto cg_start = std::chrono::high_resolution_clock::now();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Setup for distributing vectors
    std::vector<int> displacements(size), lengths(size, A.localRows);
    for (int i = 0; i < size; ++i) {
        displacements[i] = i * A.localRows;
        lengths[i] = (i == size - 1) ? A.globalRows - displacements[i] : A.localRows;
    }

    Vec local_b = Vec::Zero(lengths[rank]);
    std::cout << "Process " << rank << " - Local matrix setup completed, starting CG." << std::endl;
    MPI_Scatterv(global_b.data(), lengths.data(), displacements.data(), MPI_DOUBLE,
                 local_b.data(), lengths[rank], MPI_DOUBLE, 0, comm);

    Vec x_local = Vec::Zero(lengths[rank]);
    Vec r = local_b - A.multiply(x_local);
    double initial_residual_norm = norm(r, comm);

    if (rank == 0) {
        std::cout << "Initial residual norm: " << initial_residual_norm << std::endl;
    }

    if (initial_residual_norm < tol) {
        return Vec::Zero(A.globalRows); // Directly return a global vector if already converged
    }

    Vec z = A.applyPreconditioner(r);
    Vec p = z;
    double rsold = dotProduct(z, r, comm);
    iteration = 0;

    while (true) {
        Vec Ap = A.multiply(p);
        double pAp = dotProduct(p, Ap, comm);
        if (pAp == 0) {
            std::cout << "Division by zero encountered in pAp calculation." << std::endl;
            break; // Prevent division by zero
        }

        double alpha = rsold / pAp;
        x_local += alpha * p;
        Vec r_new = r - alpha * Ap;
        double rsnew = dotProduct(r_new, r_new, comm);

        if (sqrt(rsnew) < tol) {
            std::cout << "Process " << rank << " - Convergence achieved with residual norm: " << sqrt(rsnew) << std::endl;
            break;
        }

        Vec z_new = A.applyPreconditioner(r_new);
        double beta = rsnew / rsold;
        p = z_new + beta * p;
        r = r_new;
        z = z_new;
        rsold = rsnew;
        iteration++;
    }

    Vec global_x = Vec::Zero(A.globalRows);
    MPI_Allgatherv(x_local.data(), lengths[rank], MPI_DOUBLE,
                   global_x.data(), lengths.data(), displacements.data(), MPI_DOUBLE, comm);

    auto cg_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cg_duration = cg_end - cg_start;

    if (rank == 0) {
        Vec global_residual = global_b - global_x;
        double final_res_norm = norm(global_residual, comm);
        std::cout << "Converged after " << iteration << " iterations in " << cg_duration.count() << " seconds." << std::endl;
        std::cout << "|Ax-b|/|b| = " << final_res_norm / initial_residual_norm << std::endl;
    }

    return global_x;
}



bool isSymmetricLocal(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, double tol = 1e-10) {
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(mat, k); it; ++it) {
            if (it.row() > it.col()) {
                double symValue = mat.coeff(it.col(), it.row());
                if (std::abs(it.value() - symValue) > tol) {
                    return false;
                }
            }
        }
    }
    return true;
}


bool isSymmetricGlobal(const DistributedMatrix& A, MPI_Comm comm) {
    bool local_symmetric = isSymmetricLocal(A.localMatrix);
    int local_symmetric_int = local_symmetric ? 1 : 0;
    int global_symmetric;
    MPI_Allreduce(&local_symmetric_int, &global_symmetric, 1, MPI_INT, MPI_MIN, comm);
    return global_symmetric == 1;
}


bool isDiagonallyDominant(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat) {
    for (int i = 0; i < mat.outerSize(); ++i) {
        double sum = 0.0;
        double diag = 0.0;
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it) {
            if (it.row() == it.col())
                diag = std::abs(it.value());
            else
                sum += std::abs(it.value());
        }
        if (diag <= sum) return false; // Not diagonally dominant if diagonal is not larger than sum of non-diagonal row elements
    }
    return true;
}


bool checkPositiveDefiniteness(const SpMat& mat) {
    Eigen::SimplicialLDLT<SpMat> chol(mat);
    return chol.info() == Eigen::Success;
}








int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], option) == 0 && i + 1 < argc) {
            return std::stoi(argv[i + 1]);
        }
    }
    return default_value;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int globalRows = find_int_arg(argc, argv, "-N", 100);
    int localRows = (globalRows + size - 1) / size; // Divide rows evenly among processes

    MapMatrix localMapMatrix(localRows, localRows);
        // Ensure the matrix is diagonally dominant for positive definiteness
    for (int i = 0; i < localRows; ++i) {
        // Set a large enough value on the diagonal to ensure positive definiteness
        localMapMatrix(i, i) = 4.0;  // Increase if necessary
        if (i > 0) localMapMatrix(i, i - 1) = -1;
        if (i < localRows - 1) localMapMatrix(i, i + 1) = -1;
    }


    DistributedMatrix A(globalRows, localRows, rank, size);
    A.localMatrix = localMapMatrix.toCSR();
    A.finalize();

    if (!isSymmetricLocal(A.localMatrix)) {
        std::cerr << "Process " << rank << ": Matrix is not symmetric." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (!isSymmetricGlobal(A, MPI_COMM_WORLD)) {
        if (rank == 0) std::cerr << "Global matrix check failed: Not symmetric." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (!checkPositiveDefiniteness(A.localMatrix)) {
        std::cerr << "Local matrix check failed: Not positive definite." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    Vec b_global = Vec::Ones(globalRows);  // Initialize global vector b with ones

    double tol = 1e-6;
    int iteration;
    Vec x_global = conjugateGradient(A, b_global, tol, MPI_COMM_WORLD, iteration);

    MPI_Finalize();
    return 0;
}
