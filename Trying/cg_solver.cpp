#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <mpi.h>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Triplet;
typedef std::pair<int, int> IndexPair;

class DistributedSparseMatrix {
public:
    std::map<IndexPair, double> localData;
    int localRows, globalRows, globalCols;

    DistributedSparseMatrix(int localRows, int globalRows, int globalCols)
        : localRows(localRows), globalRows(globalRows), globalCols(globalCols) {}

    void addValue(int localRow, int col, double value) {
        if (localRow < localRows && col < globalCols) {
            localData[{localRow, col}] = value;
        }
    }

    SpMat toCSR() const {
        std::vector<Triplet> triplets;
        for (const auto& entry : localData) {
            triplets.emplace_back(entry.first.first, entry.first.second, entry.second);
        }
        SpMat mat(localRows, globalCols);
        mat.setFromTriplets(triplets.begin(), triplets.end());
        return mat;
    }

    void printLocalMatrix(int rank) const {
        std::cout << "Rank " << rank << " - Local Matrix Values:\n";
        for (const auto& entry : localData) {
            std::cout << "  (" << entry.first.first << ", " << entry.first.second << ") = " << entry.second << "\n";
        }
    }
};

void computeResidual(const SpMat& mat, const Eigen::VectorXd& x, const Eigen::VectorXd& b) {
    Eigen::VectorXd residual = mat * x - b;
    std::cout << "Residual norm: " << residual.norm() << std::endl;
}

void conjugateGradient(const DistributedSparseMatrix& A, const std::vector<double>& b, std::vector<double>& x, int maxIterations, double tol) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    SpMat csrMatrix = A.toCSR();
    Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> cg;
    cg.compute(csrMatrix);

    Eigen::VectorXd ex = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());
    Eigen::VectorXd eb = Eigen::Map<const Eigen::VectorXd>(b.data(), b.size());

    // Print initial vectors
    if (rank == 0) {
        std::cout << "Initial x vector: " << ex.transpose() << std::endl;
        std::cout << "Initial b vector: " << eb.transpose() << std::endl;
        std::cout << "Initial ";
        computeResidual(csrMatrix, ex, eb);
    }

    ex = cg.solveWithGuess(eb, ex);

    // Final residual debugging
    if (rank == 0) {
        std::cout << "Final ";
        computeResidual(csrMatrix, ex, eb);
        std::cout << "Conjugate Gradient converged in " << cg.iterations() << " iterations with an error of " << cg.error() << std::endl;
    }

    // Update the solution vector
    Eigen::Map<Eigen::VectorXd>(x.data(), x.size()) = ex;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <global matrix size>\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int globalSize = atoi(argv[1]);
    if (globalSize % size != 0) {
        if (rank == 0) {
            std::cerr << "Global matrix size must be divisible by the number of processes.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int localSize = globalSize / size;

    DistributedSparseMatrix A(localSize, globalSize, globalSize);
    std::vector<double> b(localSize, 1.0);  // Set the right-hand side to non-zero values
    std::vector<double> x(localSize, 0.0);  // Initial guess should not be trivial

    for (int i = 0; i < localSize; i++) {
        int globalRow = rank * localSize + i;
        A.addValue(i, globalRow, 2.0);
        if (globalRow > 0) A.addValue(i, globalRow - 1, -1.0);
        if (globalRow < globalSize - 1) A.addValue(i, globalRow + 1, -1.0);
    }

    // Print the local matrix for verification
    A.printLocalMatrix(rank);

    // Increase iterations and tolerance if needed
    conjugateGradient(A, b, x, 2000, 1e-6);

    MPI_Finalize();
    return 0;
}
