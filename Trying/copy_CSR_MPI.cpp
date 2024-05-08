#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <chrono>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;
typedef Eigen::VectorXd Vec;
typedef Eigen::Triplet<double> Triplet;



// Helper function to serialize a sparse matrix row
void serializeSparseRow(const SpMat &matrix, int rowIndex, std::vector<double> &values, std::vector<int> &indices) {
    int start = matrix.outerIndexPtr()[rowIndex];
    int end = matrix.outerIndexPtr()[rowIndex + 1];
    values.clear();
    indices.clear();
    for (int j = start; j < end; j++) {
        values.push_back(matrix.valuePtr()[j]);
        indices.push_back(matrix.innerIndexPtr()[j]);
    }
}

// Helper function to deserialize and insert a sparse matrix row
void deserializeSparseRow(SpMat &matrix, int rowIndex, const std::vector<double> &values, const std::vector<int> &indices) {
    std::vector<Triplet> triplets;
    for (size_t j = 0; j < values.size(); j++) {
        triplets.push_back(Triplet(rowIndex, indices[j], values[j]));
    }
    matrix.setFromTriplets(triplets.begin(), triplets.end());
}

// Function to exchange the first and last rows between neighboring MPI processes
void exchangeSparseRows(int rank, int size, SpMat &localMatrix, MPI_Comm comm) {
    MPI_Status status;
    if (rank > 0) {
        // Serialize and send the first row to the previous rank
        std::vector<double> values;
        std::vector<int> indices;
        serializeSparseRow(localMatrix, 0, values, indices);

        // Send sizes first
        int count = values.size();
        MPI_Send(&count, 1, MPI_INT, rank - 1, 0, comm);
        MPI_Send(values.data(), count, MPI_DOUBLE, rank - 1, 0, comm);
        MPI_Send(indices.data(), count, MPI_INT, rank - 1, 0, comm);

        // Receive the last row from the previous rank
        MPI_Recv(&count, 1, MPI_INT, rank - 1, 0, comm, &status);
        values.resize(count);
        indices.resize(count);
        MPI_Recv(values.data(), count, MPI_DOUBLE, rank - 1, 0, comm, &status);
        MPI_Recv(indices.data(), count, MPI_INT, rank - 1, 0, comm, &status);
        deserializeSparseRow(localMatrix, 0, values, indices);  // Note: adjust the row index as needed
    }

    if (rank < size - 1) {
        // Serialize and send the last row to the next rank
        std::vector<double> values;
        std::vector<int> indices;
        serializeSparseRow(localMatrix, localMatrix.rows() - 1, values, indices);  // Adjust index as needed

        // Send sizes first
        int count = values.size();
        MPI_Send(&count, 1, MPI_INT, rank + 1, 1, comm);
        MPI_Send(values.data(), count, MPI_DOUBLE, rank + 1, 1, comm);
        MPI_Send(indices.data(), count, MPI_INT, rank + 1, 1, comm);

        // Receive the first row from the next rank
        MPI_Recv(&count, 1, MPI_INT, rank + 1, 1, comm, &status);
        values.resize(count);
        indices.resize(count);
        MPI_Recv(values.data(), count, MPI_DOUBLE, rank + 1, 1, comm, &status);
        MPI_Recv(indices.data(), count, MPI_INT, rank + 1, 1, comm, &status);
        deserializeSparseRow(localMatrix, localMatrix.rows() - 1, values, indices);  // Adjust row index as needed
    }
}


// Partition the matrix across MPI processes
void partitionMatrix(int totalRows, int rank, int size, int& startRow, int& endRow) {
    int rowsPerProc = totalRows / size;
    int remainder = totalRows % size;
    startRow = rank * rowsPerProc + (rank < remainder ? rank : remainder);
    endRow = startRow + rowsPerProc - 1 + (rank < remainder ? 1 : 0);

    std::cout << "Rank " << rank << " handles rows from " << startRow << " to " << endRow << std::endl;
}




void exchangeBoundaryRows(int rank, int size, int* topRow, int* bottomRow, int numCols) {
    if (rank > 0) {
        MPI_Send(topRow, numCols, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
        MPI_Recv(bottomRow, numCols, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < size - 1) {
        MPI_Send(bottomRow, numCols, MPI_INT, rank + 1, 1, MPI_COMM_WORLD);
        MPI_Recv(topRow, numCols, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// Log the assigned range for each rank
void logAssignedRange(int rank, int globalRows, int startRow, int localRows) {
    std::cout << "Rank " << rank << ": Handles rows from " << startRow
              << " to " << (startRow + localRows - 1) << " out of " << globalRows << std::endl;
}

// Validate that local matrix partition aligns with global matrix
bool validatePartitioning(int rank, int startRow, int localRows, int globalRows) {
    if (startRow < 0 || startRow + localRows > globalRows) {
        std::cerr << "Rank " << rank << ": Invalid partitioning. Start row: " << startRow
                  << ", local rows: " << localRows << ", out of " << globalRows << std::endl;
        return false;
    }
    return true;
}

// Ensure all data lengths are consistent for communication
void checkMPICommunication(const std::vector<int>& lengths, const std::vector<int>& displacements, int globalRows, int rank) {
    int sum = 0;
    for (size_t i = 0; i < lengths.size(); ++i) {
        sum += lengths[i];
    }
    if (sum != globalRows) {
        std::cerr << "Rank " << rank << ": MPI communication error, total sum of lengths "
                  << "does not match global matrix rows: " << sum << " vs " << globalRows << std::endl;
    } else {
        std::cout << "Rank " << rank << ": MPI communication verified." << std::endl;
    }
}


// Helper function to calculate grid coordinates for 2D process layout
std::pair<int, int> calculateGridCoords(int rank, int numRows, int numCols) {
    return {rank / numCols, rank % numCols};
}



bool isSymmetricAndPositiveDefinite(const SpMat& mat, int rank);



// Helper function to map global index to local index
int globalToLocal(int globalIndex, int startRow) {
    return globalIndex - startRow;
}

// Modify the setupMatrix function to include bounds checking:
void setupMatrix(SpMat& matrix, int startRow, int endRow, int numCols, int rank, int size) {
    assert(matrix.cols() == numCols); // Ensure the number of columns is as expected
    std::vector<Triplet> triplets;

    for (int globalRow = startRow; globalRow <= endRow; ++globalRow) {
        int localRow = globalToLocal(globalRow, startRow);
        assert(localRow >= 0 && localRow < matrix.rows()); // Ensure local row is within bounds
        for (int col = 0; col < numCols; ++col) {
            // Example matrix initialization logic
            double value = (std::abs(globalRow - col) <= 1) ? -1.0 : 0.0;
            if (globalRow == col) value = 4.0; // Diagonal dominance
            triplets.push_back(Triplet(localRow, col, value));
        }
    }
    matrix.setFromTriplets(triplets.begin(), triplets.end());
}


//Possible fix for error:

/*
// Ensure Cholesky decomposition is called on a square matrix:
void finalizeMatrix(SpMat& matrix &A) {
    assert(A.localMatrix.rows() == A.localMatrix.cols()); // Ensure matrix is square
    A.localCholesky.compute(A.localMatrix);
    if (A.localCholesky.info() != Eigen::Success) {
        std::cerr << "Cholesky decomposition failed at rank " << A.rank << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
 */






class CSRMatrix {
public:
    SpMat localMatrix;
    Eigen::SimplicialLLT<SpMat> localCholesky; // Cholesky decomposition for preconditioning
    int globalRows, globalCols, localRows, localCols, rank, size;

    CSRMatrix(int gr, int gc, int lr, int lc, int r, int s)
        : localMatrix(lr, lc), globalRows(gr), globalCols(gc), localRows(lr), localCols(lc), rank(r), size(s) {}


    void finalize() {
        std::cout << "Matrix dimensions: " << localMatrix.rows() << "x" << localMatrix.cols() << std::endl;
        if (!isSymmetricAndPositiveDefinite(localMatrix, rank)) {
            std::cerr << "Rank " << rank << ": Matrix is not symmetric and positive-definite." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        localMatrix.makeCompressed();
        localCholesky.compute(localMatrix);
        if (localCholesky.info() != Eigen::Success) {
            std::cerr << "Cholesky decomposition failed at rank " << rank << "." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            std::cout << "Rank " << rank << ": Cholesky decomposition succeeded." << std::endl;
        }

        for (int i = 0; i < localRows; ++i) {
            double nonDiagonalSum = 0.0;
            double diagonalValue = 0.0;
            for (int j = 0; j < localCols; ++j) {
                if (i != j) {
                    nonDiagonalSum += std::abs(localMatrix.coeff(i, j));
                } else {
                    diagonalValue = localMatrix.coeff(i, j);
                }
            }
            if (diagonalValue <= nonDiagonalSum) {
                std::cout << "Rank " << rank << ": Non-positive definite condition at row " << i << ", diagonal: " << diagonalValue << ", row sum: " << nonDiagonalSum << std::endl;
            }
        }
    }

    
    Vec multiply(const Vec& local_x) const {
        assert(local_x.size() == localMatrix.cols());
        Vec result = Vec::Zero(localMatrix.rows());
        for (int k = 0; k < localMatrix.outerSize(); ++k) {
            for (SpMat::InnerIterator it(localMatrix, k); it; ++it) {
                int row = it.row();
                int col = it.col();
                assert(row >= 0 && row < localMatrix.rows());
                assert(col >= 0 && col < localMatrix.cols());
                result[row] += it.value() * local_x[col];
            }
        }
        return result;
    }

    Vec applyPreconditioner(const Vec& r) const {
        assert(r.size() == localMatrix.rows());
        return localCholesky.solve(r); // Using Cholesky factor to solve
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

// Conjugate Gradient solver using MPI for parallel processing
Vec conjugateGradient(const CSRMatrix& A, const Vec& global_b, double tol, MPI_Comm comm, int& iteration) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Calculate lengths and displacements for scattering the vectors
    std::vector<int> lengths(size), displacements(size);
    int totalLength = 0;
    for (int i = 0; i < size; ++i) {
        lengths[i] = A.globalRows / size + (i < A.globalRows % size ? 1 : 0);
        displacements[i] = totalLength;
        totalLength += lengths[i];
    }
    
    // Log and validate the assigned range
    int startRow = displacements[rank];
    logAssignedRange(rank, A.globalRows, startRow, lengths[rank]);
    if (!validatePartitioning(rank, startRow, lengths[rank], A.globalRows)) {
        MPI_Abort(MPI_COMM_WORLD, 1);  // Abort if the partition is invalid
    }

    // Check communication consistency
    checkMPICommunication(lengths, displacements, A.globalRows, rank);
    

    // Scatter the global vector b to all processes
    Vec local_b = Vec::Zero(A.localRows);
    MPI_Scatterv(global_b.data(), lengths.data(), displacements.data(), MPI_DOUBLE,
                 local_b.data(), A.localRows, MPI_DOUBLE, 0, comm);
    Vec x_local = Vec::Zero(A.localRows);


    // Begin the conjugate gradient iterations
    Vec r = local_b - A.multiply(x_local);
    Vec z = A.applyPreconditioner(r);
    Vec p = z;
    double rsold = z.dot(r);
    iteration = 0;

    while (true) {
        Vec Ap = A.multiply(p);
        double pAp = p.dot(Ap);
        if (pAp == 0) {
            std::cerr << "Process " << rank << ": Division by zero encountered in pAp calculation." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        double alpha = rsold / pAp;
        x_local += alpha * p;
        Vec r_new = r - alpha * Ap;
        double rsnew = r_new.dot(r_new);
        if (sqrt(rsnew) < tol) {
            std::cout << "Process " << rank << ": Convergence achieved with residual norm: " << sqrt(rsnew) << std::endl;
            break;
        }
        Vec z_new = A.applyPreconditioner(r_new);
        double beta = rsnew / rsold;
        p = z_new + beta * p;
        r = r_new;
        z = z_new;
        rsold = rsnew;
        iteration++;
        
        std::cout << "Iteration " << iteration << ": Residual norm = " << sqrt(rsnew) << std::endl;

    }

    // Gather all local parts of the solution vector to form the global solution vector
    Vec global_x = Vec::Zero(A.globalRows);
    MPI_Allgatherv(x_local.data(), lengths[rank], MPI_DOUBLE,
                   global_x.data(), lengths.data(), displacements.data(), MPI_DOUBLE, comm);

    return global_x;
}

// Initialize the matrix with boundary conditions
void setupBoundaryConditions(CSRMatrix& A) {
    int rowsPerGrid = static_cast<int>(std::sqrt(A.size));
    int colsPerGrid = A.size / rowsPerGrid;

    int gridRow = A.rank / colsPerGrid;
    int gridCol = A.rank % colsPerGrid;

    int startRow = gridRow * A.localRows;
    int startCol = gridCol * A.localCols;

    std::vector<Triplet> triplets;
    double diagonalBoost = 2.0; // Modify this as per matrix requirements

    std::cout << "Rank " << A.rank << ": Setting up matrix values..." << std::endl;

    for (int i = 0; i < A.localRows; ++i) {
        int globalRow = startRow + i;
        double rowSum = 0.0;

        // Assigning off-diagonal values
        for (int j = 0; j < A.localCols; ++j) {
            int globalCol = startCol + j;

            // Ensure indices are within valid boundaries
            assert(i >= 0 && i < A.localRows);
            assert(j >= 0 && j < A.localCols);

            // Only add values if within global matrix bounds
            if (globalRow < A.globalRows && globalCol < A.globalCols) {
                double value = 0.0;

                // Assign matrix values here (adjust logic as needed)
                if (std::abs(globalRow - globalCol) == 1) {
                    value = -1.0; // Example off-diagonal value
                    rowSum += std::abs(value);
                }

                // Insert values
                triplets.emplace_back(i, j, value);
                std::cout << "Rank " << A.rank << ": Setting value at (" << i << ", " << j << ") to " << value << std::endl;
            } else {
                std::cerr << "Rank " << A.rank << ": Index out of bounds at (" << globalRow << ", " << globalCol << ")" << std::endl;
            }
        }

        // Add the diagonal value with a boost
        double diagonalValue = rowSum + diagonalBoost;
        triplets.emplace_back(i, i, diagonalValue);
        std::cout << "Rank " << A.rank << ": Setting diagonal (" << i << ", " << i << ") to " << diagonalValue << " (Row sum + boost)" << std::endl;
    }

    A.localMatrix.setFromTriplets(triplets.begin(), triplets.end());
    A.localMatrix.makeCompressed();
    std::cout << "Rank " << A.rank << ": Boundary conditions setup completed. Matrix size: " << A.localRows << "x" << A.localCols << "." << std::endl;
}


// Check for symmetry and positive definiteness
bool isSymmetricAndPositiveDefinite(const SpMat &matrix, int rank) {
    bool isSymmetric = true, isPositiveDefinite = true;
    for (int i = 0; i < matrix.outerSize(); ++i) {
        for (SpMat::InnerIterator it(matrix, i); it; ++it) {
            if (it.row() == it.col() && it.value() <= 0) {
                isPositiveDefinite = false; // Diagonal element not positive
                std::cerr << "Rank " << rank << ": Non-positive diagonal element found." << std::endl;
            }
            if (it.row() != it.col() && matrix.coeff(it.col(), it.row()) != it.value()) {
                isSymmetric = false; // Non-symmetric element found
                std::cerr << "Rank " << rank << ": Matrix is not symmetric." << std::endl;
            }
        }
    }
    return isSymmetric && isPositiveDefinite;
}



int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}


int main(int argc, char* argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Global matrix dimensions
    int N = 20;
    if (N % size != 0) {
        if (rank == 0) {
            std::cerr << "N must be divisible by the number of processes." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int localCols = N; // Number of columns in the local matrix, equal to the total number of columns

    // Local rows per process
    int startRow, endRow;
    partitionMatrix(N, rank, size, startRow, endRow);
    int localRows = endRow - startRow + 1;

    // Validate the partitioning was successful and makes sense
    if (localRows < 1) {
        std::cerr << "Rank " << rank << " has an invalid number of rows: " << localRows << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    SpMat localMatrix(localRows, localCols);  // Local matrix for each process

    // Initialize the matrix with appropriate values
    setupMatrix(localMatrix, startRow, endRow, N, rank, size);

    // Set up the CSRMatrix object for the local partition
    CSRMatrix A(N, N, localRows, localCols, rank, size);

    // Initialize the matrix with boundary conditions and verify it is set up correctly
    setupBoundaryConditions(A);
    A.finalize(); // Finalize the matrix setup

    // Exchange boundary rows between neighboring ranks
    int* topRow = new int[localCols]();       // allocate with initialized zeros
    int* bottomRow = new int[localCols]();    // allocate with initialized zeros
    exchangeBoundaryRows(rank, size, topRow, bottomRow, localCols);

    // Random initial guess vector of size `localRows`
    Vec x = Vec::Random(localRows);

    // Right-hand side (RHS) vector `b` (ones) of global size
    Vec b_global = Vec::Ones(N);
    double tol = 1e-6;
    int iterations = 0;

    // Start the timer
    MPI_Barrier(MPI_COMM_WORLD);
    auto time_start = MPI_Wtime();

    // Solve using Conjugate Gradient
    Vec x_global = conjugateGradient(A, b_global, tol, MPI_COMM_WORLD, iterations);

    // Stop the timer
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_time = MPI_Wtime() - time_start;
    if (rank == 0) {
        std::cout << "Wall time for CG: " << elapsed_time << " seconds" << std::endl;
    }

    // Calculate residual error
    Vec r_global = b_global - A.multiply(x_global);
    double residual_norm = norm(r_global, MPI_COMM_WORLD);
    double b_norm = norm(b_global, MPI_COMM_WORLD);
    double err = residual_norm / b_norm;

    if (rank == 0) {
        std::cout << "|Ax-b|/|b| = " << err << std::endl;
    }

    // Clean up memory
    delete[] topRow;
    delete[] bottomRow;

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}





/*
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Smaller, manageable size for debugging
    int globalRows = 40;
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    assert(globalRows % size == 0);  // Ensure divisibility

    int localRows = globalRows / size;

    CSRMatrix A(globalRows, globalRows, localRows, localRows, rank, size);
    setupBoundaryConditions(A);  // Assuming a simple modification for debugging
    A.finalize();

    Vec b_global = Vec::Ones(globalRows);
    double tol = 1e-6;
    int iteration;
    Vec x_global = conjugateGradient(A, b_global, tol, MPI_COMM_WORLD, iteration);

    MPI_Finalize();
    return 0;
}


*/
