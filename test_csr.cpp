#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <mpi.h>
#include <Eigen/Sparse>

// Types for Eigen sparse matrices
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

// Parallel CSR Matrix class
class CSRMatrix {
public:
    int nrows, ncols;
    std::vector<int> row_ptr, col_indices;
    std::vector<double> values;

    CSRMatrix(int rows, int cols, const std::vector<T>& triplets) {
        nrows = rows;
        ncols = cols;

        // Initialize row pointers
        row_ptr.resize(nrows + 1, 0);

        // Count occurrences of rows in the triplets
        for (const auto& triplet : triplets) {
            row_ptr[triplet.row() + 1]++;
        }

        // Compute cumulative counts to get final row pointers
        for (int i = 0; i < nrows; ++i) {
            row_ptr[i + 1] += row_ptr[i];
        }

        // Initialize column indices and values with proper sizes
        col_indices.resize(triplets.size());
        values.resize(triplets.size());

        // Temporary array to track positions within each row
        std::vector<int> temp_pos(row_ptr);

        // Fill column indices and values based on row positions
        for (const auto& triplet : triplets) {
            int row = triplet.row();
            int pos = temp_pos[row]++;
            col_indices[pos] = triplet.col();
            values[pos] = triplet.value();
        }
    }

    // Matrix-vector multiplication
    std::vector<double> operator*(const std::vector<double>& vec) const {
        std::vector<double> result(nrows, 0.0);
        for (int row = 0; row < nrows; ++row) {
            for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
                result[row] += values[j] * vec[col_indices[j]];
            }
        }
        return result;
    }
};

// Parallel norm computation
double parallel_norm(const std::vector<double>& vec) {
    double local_sum = 0.0;
    for (double val : vec) {
        local_sum += val * val;
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return std::sqrt(global_sum);
}

// Parallel residual computation
double compute_residual(const CSRMatrix& A, const std::vector<double>& x, const std::vector<double>& b) {
    std::vector<double> Ax = A * x;
    std::vector<double> residual(Ax.size());
    for (size_t i = 0; i < Ax.size(); ++i) {
        residual[i] = Ax[i] - b[i];
    }
    return parallel_norm(residual);
}

// Distributed Conjugate Gradient Solver
void parallel_cg(const CSRMatrix& A, const std::vector<double>& b, std::vector<double>& x, double tol = 1e-6) {
    assert(b.size() == A.nrows);
    x.assign(b.size(), 0.0);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> r = b;
    std::vector<double> p = r;
    std::vector<double> Ap(A.nrows, 0.0);
    double rs_old = parallel_norm(r) * parallel_norm(r);

    double tol2 = tol * tol;

    for (int i = 0; i < 1000; ++i) {
        Ap = A * p;
        double pAp = 0.0;
        for (int j = 0; j < A.nrows; ++j) {
            pAp += p[j] * Ap[j];
        }

        double global_pAp = 0.0;
        MPI_Allreduce(&pAp, &global_pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double alpha = rs_old / global_pAp;

        for (int j = 0; j < x.size(); ++j) {
            x[j] += alpha * p[j];
            r[j] -= alpha * Ap[j];
        }

        double rs_new = parallel_norm(r) * parallel_norm(r);

        if (rank == 0) {
            std::cout << "Iteration: " << i + 1 << "\tResidual: " << std::sqrt(rs_new) << std::endl;
        }

        if (rs_new < tol2) {
            break;
        }

        double beta = rs_new / rs_old;
        for (int j = 0; j < p.size(); ++j) {
            p[j] = r[j] + beta * p[j];
        }

        rs_old = rs_new;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 1000; // Matrix size
    if (argc > 1) N = std::stoi(argv[1]);

    int n = N / size; // Rows per process
    std::vector<T> triplets;

    // Each process generates its portion of the 1D Laplacian matrix
    int offset = rank * n;
    for (int i = 0; i < n; ++i) {
        int global_row = offset + i;
        triplets.emplace_back(i, global_row, 2.0);
        if (global_row > 0) triplets.emplace_back(i, global_row - 1, -1.0);
        if (global_row < N - 1) triplets.emplace_back(i, global_row + 1, -1.0);
    }

    // Initialize local CSR matrix
    CSRMatrix A(n, N, triplets);

    // Initial guess and right-hand side vectors
    std::vector<double> x(n, 0.0);
    std::vector<double> b(n, 1.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    parallel_cg(A, b, x);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_time = MPI_Wtime() - start_time;
    if (rank == 0) {
        std::cout << "Conjugate Gradient completed in " << elapsed_time << " seconds." << std::endl;
    }

    // Calculate relative error |Ax - b| / |b|
    double final_residual = compute_residual(A, x, b);
    double norm_b = parallel_norm(b);
    double relative_error = final_residual / norm_b;

    if (rank == 0) {
        std::cout << "|Ax-b|/|b| = " << relative_error << std::endl;
    }

    MPI_Finalize();
    return 0;
}
