#include <iostream>
#include <vector>
#include <fstream>
#include "armadillo"
#include "mesh.h"
#include "formulation.h"
#include "gdata.h"

#define LINEAR 1

// Prototypes
template <typename T> void integrate(const element<T>& elem, const int n_gauss_pts, const std::vector<T>& nodes, const arma::Col<T>& temperature, const T k1, const T k2, const T k3, arma::Mat<T>& K_e, arma::Col<T>& F_e);
template <typename T> void assemble(const mesh<T>& m, const int n_gauss_pts, const arma::Col<T>& temperature, const T k1, const T k2, const T k3, arma::Mat<T>& K, arma::Col<T>& F);
template <typename T> void apply_dirichlet(const mesh<T>& m, T T_left, T T_right, arma::Mat<T>& K, arma::Col<T>& F);
template <typename T> arma::Col<T> solve(const arma::Mat<T>& K, const arma::Col<T>& F);
template <typename T> void iterate(const mesh<T>& m, const int ngpts, const T tol, const int max_iter, arma::Col<T>& temperature, const T k1, const T k2, const T k3);

// Main code 
int main() {

  // Declaring variables
  int nnodes;
  int nelem = 100;
  mesh<double> grid(nelem,LINEAR,0.0,1.0);
  grid.generate();
  nnodes = grid.get_num_nodes();
  double k1,k2,k3;
  k1 = 1.0;
  k2 = 2.0;
  k3 = 6.0;
  
  // Initial guess for temperature
  arma::Col<double> T = arma::ones<arma::Col<double> >(nnodes)*50.0;

  // Iterating
  iterate<double>(grid,3,1.0e-10,10000,T,k1,k2,k3);

  // Writing final data to file
  std::ofstream outfile("results.dat");
  outfile.precision(16);
  outfile.setf(std::ios_base::scientific);
  std::vector<double> nodes = grid.get_nodes();
  for (unsigned int i=0; i<nnodes; ++i) {
    outfile << nodes[i] << " " << T[i] << std::endl;
  }
  outfile.close();

  return 0;

}

template <typename T>
void integrate(const element<T>& elem, const int n_gauss_pts, const std::vector<T>& nodes, const arma::Col<T>& temperature, const T k1, const T k2, const T k3, arma::Mat<T>& K_e, arma::Col<T>& F_e) {

  // Zero-ing out Ke and Fe
  K_e.zeros();
  F_e.zeros();

  // Declaring some variables
  arma::Mat<T> K_local(2,2);
  arma::Col<T> F_local(2);

  // Getting Gauss points
  std::vector<T> xi(n_gauss_pts);
  std::vector<T> w(n_gauss_pts);
  gdata<T>(n_gauss_pts,xi,w);

  // Sampling integrands
  for (int i=0; i<n_gauss_pts; ++i) {
    sample_integrands<T>(elem, xi[i], nodes, temperature, k1, k2, k3, K_local, F_local);
    K_e += w[i]*K_local;
    F_e += w[i]*F_local;
  }

}

template <typename T>
void assemble(const mesh<T>& m, const int n_gauss_pts, const arma::Col<T>& temperature, const T k1, const T k2, const T k3, arma::Mat<T>& K, arma::Col<T>& F) {

  // Declaring variables
  K.zeros();
  F.zeros();
  arma::Mat<T> K_e(2,2);
  arma::Col<T> F_e(2);
  std::vector<int> con;
  std::vector<T> nodes = m.get_nodes();
  std::vector<element<T> > elements = m.get_elements();

  // Looping over elements
  for (unsigned int en=0; en<m.get_num_elem(); ++en) {
    
    // Getting element stiffness matrix and load vector
    integrate(elements[en],n_gauss_pts,nodes,temperature,k1,k2,k3,K_e,F_e);
    con = elements[en].get_connectivity();

    // Assembling
    for (unsigned int i=0; i<K_e.n_rows; ++i) {
      for (unsigned int j=0; j<K_e.n_cols; ++j) {
        K(con[i],con[j]) += K_e(i,j);
      }
      F(con[i]) += F_e(i);
    }

  }
  
}

template <typename T> 
void apply_dirichlet(const mesh<T>& m, T T_left, T T_right, arma::Mat<T>& K, arma::Col<T>& F) {
 
  // Getting elements
  std::vector<element<T> > elements = m.get_elements();

  // Applying to left side of domain
  std::vector<int> con = elements[0].get_connectivity();
  for (unsigned int i=0; i<K.n_cols; ++i) {
    K(con[0],i) = (T) 0;
  }
  F(con[0]) = T_left;
  K(con[0],con[0]) = (T) 1;

  // Applying to right side of domain
  con = elements.back().get_connectivity();
  for (unsigned int i=0; i<K.n_cols; ++i) {
    K(con.back(),i) = (T) 0;
  }
  F(con.back()) = T_right;
  K(con.back(),con.back()) = (T) 1;

}

template <typename T>
arma::Col<T> solve(const arma::Mat<T>& K, const arma::Col<T>& F) {
  arma::Col<T> u(F.n_elem);
  arma::solve(u,K,F);
  return u;
}

template <typename T>
void iterate(const mesh<T>& m, const int ngpts, const T tol, const int max_iter, arma::Col<T>& temperature, const T k1, const T k2, const T k3) {

  // Declaring some variables
  unsigned int nnodes = m.get_num_nodes();
  arma::Mat<T> K(nnodes,nnodes);
  arma::Col<T> F(nnodes);
  arma::Col<T> Tn(nnodes);
  arma::Col<T> Tnp1(nnodes);
  Tn = temperature;
  int i = 0;
  T eps = 100.0;

  while ((i<max_iter)&&(eps>tol)) {

    // Assembling system, applying bcs and solving
    assemble(m,ngpts,Tn,k1,k2,k3,K,F);
    apply_dirichlet<T>(m,(T) 0.0,(T) 100.0,K,F);
    Tnp1 = solve(K,F);

    // Computing the residual
    eps = arma::norm(Tnp1-Tn)/arma::norm(Tnp1);
    std::cout << "Iteration: " << i << " Residual: " << eps << std::endl;

    // Updating
    Tn = Tnp1;
    ++i;
  }
  temperature = Tnp1;

}
