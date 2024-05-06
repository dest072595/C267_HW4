#include <iostream>
#include <vector>
#include <map>
#include <mpi.h>
#include <cmath>
#include <Eigen/Sparse>
#include <numeric>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

class MapMatrix {
public:
    std::map<std::pair<int, int>, double> data;
    int nbrow, nbcol;

    // Constructor
    MapMatrix(int nr, int nc) : nbrow(nr), nbcol(nc) {}

    // CSR components
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_pointers;

    void toCSR() {
        row_pointers.resize(nbrow + 1, 0);

        // Count the number of non-zeros in each row
        for (const auto& kv : data) {
            row_pointers[kv.first.first + 1]++;
        }

        // Accumulate counts to get indices
        for (int i = 1; i <= nbrow; ++i) {
            row_pointers[i] += row_pointers[i - 1];
        }

        int nnz = row_pointers[nbrow];
        values.resize(nnz);
        col_indices.resize(nnz);

        // Fill values and column indices
        std::vector<int> offset(nbrow, 0);
        for (const auto& kv : data) {
            int row = kv.first.first;
            int idx = row_pointers[row] + offset[row]++;
            values[idx] = kv.second;
            col_indices[idx] = kv.first.second;
        }
    }

    double operator()(int row, int col) const {
        auto it = data.find(std::make_pair(row, col));
        if (it != data.end())
            return it->second;
        return 0;
    }

    double& operator()(int row, int col) {
        return data[std::make_pair(row, col)];
    }
};

std::vector<double> matVecProduct(const MapMatrix& A, const std::vector<double>& x) {
    std::vector<double> b(A.nbrow, 0.0);
    for (int i = 0; i < A.nbrow; ++i) {
        for (int j = A.row_pointers[i]; j < A.row_pointers[i + 1]; ++j) {
            b[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return b;
}

void CG(const MapMatrix& A, const std::vector<double>& b, std::vector<double>& x, double tol = 1e-6) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = A.nbrow;
    std::vector<double> r = b, p = r, Ap(n);
    double rdot = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

    for (int iter = 0; iter < n; ++iter) {
        Ap = matVecProduct(A, p);
        double pAp = std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0);
        double alpha = rdot / pAp;

        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double new_rdot = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);
        if (sqrt(new_rdot) < tol) break;
        double beta = new_rdot / rdot;
        for (int i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        rdot = new_rdot;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 10000;
    if (N % size != 0) {
        if (rank == 0) {
            std::cerr << "Matrix size N must be divisible by the number of processes." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = N / size;

    MapMatrix A(n, N);
    int row_start = rank * n;

    for (int i = 0; i < n; ++i) {
        int global_row = row_start + i;
        if (global_row > 0) A(global_row, global_row - 1) = -1.0; // Off-diagonal left
        A(global_row, global_row) = 2.0;                           // Diagonal
        if (global_row < N - 1) A(global_row, global_row + 1) = -1.0; // Off-diagonal right
    }

    A.toCSR();

    std::vector<double> x(n, 0.0);
    std::vector<double> b(n, 1.0);

    double tol = 1e-6;
    double start_time = MPI_Wtime();

    CG(A, b, x, tol);

    double end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Conjugate Gradient completed in " << (end_time - start_time) << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
