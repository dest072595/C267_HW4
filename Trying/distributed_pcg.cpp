#include <iostream>
#include <map>
#include <vector>
#include <cassert>
#include <mpi.h>
#include <numeric>

#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

class MapMatrix{
public:
  typedef std::pair<int,int>   N2;

  std::map<N2,double>  data;
  int nbrow;
  int nbcol;

public:
  MapMatrix(const int& nr, const int& nc):
    nbrow(nr), nbcol(nc) {}; 

  MapMatrix(const MapMatrix& m): 
    nbrow(m.nbrow), nbcol(m.nbcol), data(m.data) {}; 
  
  MapMatrix& operator=(const MapMatrix& m){ 
    if(this!=&m){
      nbrow=m.nbrow;
      nbcol=m.nbcol;
      data=m.data;
    }   
    return *this; 
  }

  int NbRow() const {return nbrow;}
  int NbCol() const {return nbcol;}

  double operator()(const int& j, const int& k) const {
    auto search = data.find(std::make_pair(j,k));
    if(search!=data.end()) return search->second;
    return 0;
  }

  double& Assign(const int& j, const int& k) {
    return data[std::make_pair(j,k)];
  }

  // parallel matrix-vector product with distributed vector xi
  std::vector<double> operator*(const std::vector<double>& xi) const {



    std::vector<double> x(NbCol());
    std::copy(xi.begin(),xi.end(),x.begin());
    

    std::vector<double> b(NbRow(),0.);
    for(auto it=data.begin(); it!=data.end(); ++it){
      int j = (it->first).first;
      int k = (it->first).second; 
      double Mjk = it->second;
      b[j] += Mjk*x[k];
    }

    return b;
  }
};

#include <cmath>

// parallel scalar product (u,v) (u and v are distributed)
double operator,(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size()==v.size());
  double sp=0.;
  for(int j=0; j<u.size(); j++){sp+=u[j]*v[j];}

  return sp; 
}

// norm of a vector u
double Norm(const std::vector<double>& u) { 
  return sqrt((u,u));
}

// addition of two vectors u+v
std::vector<double> operator+(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size()==v.size());
  std::vector<double> w=u;
  for(int j=0; j<u.size(); j++){w[j]+=v[j];}
  return w;
}

// multiplication of a vector by a scalar a*u
std::vector<double> operator*(const double& a, const std::vector<double>& u){ 
  std::vector<double> w(u.size());
  for(int j=0; j<w.size(); j++){w[j]=a*u[j];}
  return w;
}

// addition assignment operator, add v to u
void operator+=(std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size()==v.size());
  for(int j=0; j<u.size(); j++){u[j]+=v[j];}
}

/* block Jacobi preconditioner: perform forward and backward substitution
   using the Cholesky factorization of the local diagonal block computed by Eigen */
std::vector<double> prec(const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>& P, const std::vector<double>& u){
  Eigen::VectorXd b(u.size());
  for (int i=0; i<u.size(); i++) b[i] = u[i];
  Eigen::VectorXd xe = P.solve(b);
  std::vector<double> x(u.size());
  for (int i=0; i<u.size(); i++) x[i] = xe[i];
  return x;
}

// distributed conjugate gradient
void CG(const MapMatrix& A,
        const std::vector<double>& b,
        std::vector<double>& x,
        double tol=1e-6) {

  assert(b.size() == A.NbRow());
  x.assign(b.size(),0.);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  int n = A.NbRow();

  // get the local diagonal block of A
  std::vector<Eigen::Triplet<double>> coefficients;
  for(auto it=A.data.begin(); it!=A.data.end(); ++it){
    int j = (it->first).first;
    int k = (it->first).second;
    if (k >= 0 && k < n) coefficients.push_back(Eigen::Triplet<double>(j,k,it->second)); 
  }

  // compute the Cholesky factorization of the diagonal block for the preconditioner
  Eigen::SparseMatrix<double> B(n,n);
  B.setFromTriplets(coefficients.begin(), coefficients.end());
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(B);

  std::vector<double> r=b, z=prec(P,r), p=z, Ap=A*p;
  double np2=(p,Ap), alpha=0.,beta=0.;
  double nr = sqrt((z,r));
  double epsilon = tol*nr;

  std::vector<double> res = A*x;
  res += (-1)*b;
  
  double rres = sqrt((res,res));

  int num_it = 0;
  while(rres>1e-5) {
    alpha = (nr*nr)/(np2);
    x += (+alpha)*p; 
    r += (-alpha)*Ap;
    z = prec(P,r);
    nr = sqrt((z,r));
    beta = (nr*nr)/(alpha*np2); 
    p = z+beta*p;    
    Ap=A*p;
    np2=(p,Ap);

    rres = sqrt((r,r));

    num_it++;
    if(rank == 0 && !(num_it%1)) {
      std::cout << "iteration: " << num_it << "\t";
      std::cout << "residual:  " << rres     << "\n";
    }
  }
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}



int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv); // Initialize the MPI environment
  
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "-N <int>: side length of the sparse matrix" << std::endl;
        return 0;
    }




  int N = find_int_arg(argc, argv, "-N", 100000); // global size

  assert(N%size == 0);
  int n = N/size; // number of local rows

  // row-distributed matrix
  MapMatrix A(n,N);

  int offset = n*rank;

  // local rows of the 1D Laplacian matrix; local column indices start at -1 for rank > 0
  for (int i=0; i<n; i++) {
    A.Assign(i,i)=2.0;
    if (offset + i - 1 >= 0) A.Assign(i,i - 1) = -1;
    if (offset + i + 1 < N)  A.Assign(i,i + 1) = -1;
    if (offset + i + N < N) A.Assign(i, i + N) = -1;
    if (offset + i - N >= 0) A.Assign(i, i - N) = -1;
  }

  // initial guess
  std::vector<double> x(n,0);

  // right-hand side
  std::vector<double> b(n,1);

  MPI_Barrier(MPI_COMM_WORLD);
  double time = MPI_Wtime();

  CG(A,b,x);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::cout << "wall time for CG: " << MPI_Wtime()-time << std::endl;

  std::vector<double> r = A*x + (-1)*b;

  double err = Norm(r)/Norm(b);
  if (rank == 0) std::cout << "|Ax-b|/|b| = " << err << std::endl;

  MPI_Finalize(); // Finalize the MPI environment

  return 0;
    
    
}
