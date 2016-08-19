#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <type_traits>
#include "armadillo"
#include "mesh.h"
#include "sacvm_formulation.h"
#include "gdata.h"

#define LINEAR 1

// Function prototypes
template <typename T> void write_data(const std::string& filename, const mesh<T>& m, const T L2norm, const arma::Col<T>& Tn);
template <typename T,typename U=T> void integrate(const element<T>& elem, const int n_gauss_pts, const std::vector<T>& nodes, const arma::Col<T>& temperature, const U k1, const U k2, const U k3, arma::Mat<U>& K_e, arma::Col<U>& F_e);
template <typename T> void assemble(const mesh<T>& m, const int n_gauss_pts, const arma::Col<T>& temperature, const T k1, const T k2, const T k3, arma::Mat<T>& K, arma::Col<T>& F);
template <typename T,typename U> void assemble_perturbed(const mesh<T>& m, const int n_gauss_pts, const arma::Col<T>& temperature, const U k1, const U k2, const U k3, arma::Mat<T>& dK, arma::Col<T>& dF);
template <typename T> void apply_dirichlet(const mesh<T>& m, T T_left, T T_right, arma::Mat<T>& K, arma::Col<T>& F);
template <typename T> arma::Col<T> solve(const arma::Mat<T>& K, const arma::Col<T>& F);
template <typename T> arma::Mat<T> iterate(const mesh<T>& m, const int ngpts, const T tol, const int max_iter, arma::Col<T>& temperature, const T k1, const T k2, const T k3);
template <typename U> U T_exact(U x, U TL, U TR, U L);
template <typename U> U dTdk1_exact(U x, U TL, U TR, U L);
template <typename U> U dTdk2_exact(U x, U TL, U TR, U L);
template <typename U> U dTdk3_exact(U x, U TL, U TR, U L);
template <typename U> U L2_T(const mesh<U>& g, const arma::Col<U>& T_n, const int ngpts);
template <typename U> void compute_sensitivity(const mesh<U>& m, const int ngpts, arma::Mat<U> K, const arma::Col<U>& temperature, const U k1, const U k2, const U k3, const U dk1, const U dk2, const U dk3, arma::Col<U>& dT);

// Main code 
int main() {

  // Declaring variables
  int nnodes;
  int nelem = 100;
  double k1,k2,k3;
  double dk1,dk2,dk3;
  k1 = 1.0;
  k2 = 2.0;
  k3 = 6.0;
  //dk1 = k1/1000.0;
  //dk2 = k2/1000.0;
  //dk3 = k3/1000.0;
  dk1 = 1.0e-12;
  dk2 = 1.0e-12;
  dk3 = 1.0e-12;

  // Creating meshes
  mesh<double> grid(nelem,LINEAR,0.0,1.0);
  grid.generate();
  nnodes = grid.get_num_nodes();
  
  // Initial guess for temperature
  arma::Col<double> Tm = arma::ones<arma::Col<double> >(nnodes)*50.0;

  // Iterating
  arma::Mat<double> Kglobal = iterate<double>(grid,1,1.0e-12,1000,Tm,k1,k2,k3);

  // Computing sensitivities
  arma::Col<double> sens_k1(nnodes);
  arma::Col<double> sens_k2(nnodes);
  arma::Col<double> sens_k3(nnodes);
  sens_k1.zeros();
  sens_k2.zeros();
  sens_k3.zeros();
  compute_sensitivity<double>(grid,1,Kglobal,Tm,k1,k2,k3,dk1,0.0,0.0,sens_k1);
  compute_sensitivity<double>(grid,1,Kglobal,Tm,k1,k2,k3,0.0,dk2,0.0,sens_k2);
  compute_sensitivity<double>(grid,1,Kglobal,Tm,k1,k2,k3,0.0,0.0,dk3,sens_k3);

  // Writing sensitivities to file
  std::vector<double> nodes = grid.get_nodes();
  std::ofstream sensfile("sacvm.dat");
  sensfile.precision(16);
  sensfile.setf(std::ios_base::scientific);
  for (unsigned int i=0; i<sens_k1.n_elem; ++i) {
    sensfile << nodes[i] << " " << sens_k1[i] << " " << sens_k2[i] << " " << sens_k3[i] << "\n";
  }
  sensfile.close();

  return 0;

}

template <typename T> void write_data(const std::string& filename, const mesh<T>& m, const T L2norm, const arma::Col<T>& Tn) {
  std::ofstream outfile(filename.c_str());
  outfile.precision(16);
  outfile.setf(std::ios_base::scientific);
  std::vector<double> nodes = m.get_nodes();
  unsigned int nnodes = nodes.size();
  outfile << nnodes << " " << L2norm << std::endl;
  for (unsigned int i=0; i<nnodes; ++i) {
    outfile << nodes[i] << " " << Tn[i] << std::endl;
  }
  outfile.close();
}

template <typename T,typename U=T>
void integrate(const element<T>& elem, const int n_gauss_pts, const std::vector<T>& nodes, const arma::Col<T>& temperature, const U k1, const U k2, const U k3, arma::Mat<U>& K_e, arma::Col<U>& F_e) {

  // Zero-ing out Ke and Fe
  K_e.zeros();
  F_e.zeros();

  // Declaring some variables
  arma::Mat<U> K_local(2,2);
  arma::Col<U> F_local(2);

  // Getting Gauss points
  std::vector<T> xi(n_gauss_pts);
  std::vector<T> w(n_gauss_pts);
  gdata<T>(n_gauss_pts,xi,w);

  // Sampling integrands
  for (int i=0; i<n_gauss_pts; ++i) {
    sample_integrands<T,U>(elem, xi[i], nodes, temperature, k1, k2, k3, K_local, F_local);
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
    integrate<T>(elements[en],n_gauss_pts,nodes,temperature,k1,k2,k3,K_e,F_e);
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

template <typename T,typename U> void assemble_perturbed(const mesh<T>& m, const int n_gauss_pts, const arma::Col<T>& temperature, const U k1, const U k2, const U k3, arma::Mat<T>& dK, arma::Col<T>& dF) {

  // Declaring variables
  dK.zeros();
  dF.zeros();
  arma::Mat<U> K_e(2,2);
  arma::Col<U> F_e(2);
  std::vector<int> con;
  std::vector<T> nodes = m.get_nodes();
  std::vector<element<T> > elements = m.get_elements();

  // Looping over elements
  for (unsigned int en=0; en<m.get_num_elem(); ++en) {
    
    // Getting element stiffness matrix and load vector
    integrate<T,U>(elements[en],n_gauss_pts,nodes,temperature,k1,k2,k3,K_e,F_e);
    con = elements[en].get_connectivity();

    // Assembling
    for (unsigned int i=0; i<K_e.n_rows; ++i) {
      for (unsigned int j=0; j<K_e.n_cols; ++j) {
        dK(con[i],con[j]) += K_e(i,j).imag();
      }
      dF(con[i]) += F_e(i).imag();
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
arma::Mat<T> iterate(const mesh<T>& m, const int ngpts, const T tol, const int max_iter, arma::Col<T>& temperature, const T k1, const T k2, const T k3) {

  // Declaring some variables
  unsigned int nnodes = m.get_num_nodes();
  arma::Mat<T> K(nnodes,nnodes);
  arma::Col<T> F(nnodes);
  arma::Col<T> Tn(nnodes);
  arma::Col<T> Tnp1(nnodes);
  Tn = temperature;
  int i = 0;
  T eps = 100.0;
  std::cout << "size(K) = (" << K.n_rows << "," << K.n_cols << ")" << std::endl;

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

  return K;

}

template <typename U>
U T_exact(U x, U TL, U TR, U L) {

  // Declaring variables
  U T1 = 0.0;
  U T2 = 50.0;
  U T3 = 100.0;
  U k1 = 1.0;
  U k2 = 2.0;
  U k3 = 6.0;
  U b1,b2,C1,C2;

  b1 = (k2 - k1)/(T2 - T1);
  b2 = (k3 - k2)/(T3 - T2);
  C1 = -(b1/2.0*TL*TL+(k1 - b1*T1)*TL)*(1.0-x/L) + ((b2-b1)/2.0*T2*T2+((k2-b2*T2)-(k1-b1*T1))*T2)*x/L - (b2/2.0*TR*TR+(k2-b2*T2)*TR)*x/L;
  C2 = -(b1/2.0*TL*TL+(k1 - b1*T1)*TL)*(1.0-x/L) + ((b2-b1)/2.0*T2*T2+((k2-b2*T2)-(k1-b1*T1))*T2)*(x/L-1.0) - (b2/2.0*TR*TR+(k2-b2*T2)*TR)*x/L;

  // Calculations
  if (x<=0.27272727) {
    return (-2*k1*L + 2*b1*L*T1 + sqrt(pow(-2*k1*L + 2*b1*L*T1,2) + 4*b1*L*(2*k1*L*TL - 2*b1*L*T1*TL + b1*L*pow(TL,2) + 2*k1*T2*x - 2*k2*T2*x - 2*b1*T1*T2*x + b1*pow(T2,2)*x + b2*pow(T2,2)*x - 2*k1*TL*x + 2*b1*T1*TL*x - b1*pow(TL,2)*x + 2*k2*TR*x - 2*b2*T2*TR*x + b2*pow(TR,2)*x)))/(2.*b1*L);
  }
  else {
    return (-2*k2*L + 2*b2*L*T2 + sqrt(pow(2*k2*L - 2*b2*L*T2,2) - 4*b2*L*(2*k1*L*T2 - 2*k2*L*T2 - 2*b1*L*T1*T2 + b1*L*pow(T2,2) + b2*L*pow(T2,2) - 2*k1*L*TL + 2*b1*L*T1*TL - b1*L*pow(TL,2) - 2*k1*T2*x + 2*k2*T2*x + 2*b1*T1*T2*x - b1*pow(T2,2)*x - b2*pow(T2,2)*x + 2*k1*TL*x - 2*b1*T1*TL*x + b1*pow(TL,2)*x - 2*k2*TR*x + 2*b2*T2*TR*x - b2*pow(TR,2)*x)))/(2.*b2*L);

  }

}

template <typename U>
U dTdk1_exact(U x, U TL, U TR, U L) {

  // Declaring variables
  U T1 = 0.0;
  U T2 = 50.0;
  U T3 = 100.0;
  U k1 = 1.0;
  U k2 = 2.0;
  U k3 = 6.0;

  // Calculations
  if (x<=0.27272727) {
    return (2*L*pow(T2,2) - 2*L*T2*T3 + (2*(-2*L*pow(T2,2) + 2*L*T2*T3)*
          (2*k2*L*T1*T2 - 2*k1*L*pow(T2,2) - 2*k2*L*T1*T3 + 2*k1*L*T2*T3) - 
         4*(k1*L*T2 - k2*L*T2 - k1*L*T3 + k2*L*T3)*
          (2*L*pow(T2,2)*TL - 2*L*T2*T3*TL - L*T2*pow(TL,2) + L*T3*pow(TL,2) + pow(T2,3)*x - 
            pow(T2,2)*T3*x - 2*pow(T2,2)*TL*x + 2*T2*T3*TL*x + T2*pow(TL,2)*x - 
            T3*pow(TL,2)*x) - 4*(L*T2 - L*T3)*
          (-2*k2*L*T1*T2*TL + 2*k1*L*pow(T2,2)*TL + 2*k2*L*T1*T3*TL - 2*k1*L*T2*T3*TL - 
            k1*L*T2*pow(TL,2) + k2*L*T2*pow(TL,2) + k1*L*T3*pow(TL,2) - k2*L*T3*pow(TL,2) - 
            k2*T1*pow(T2,2)*x + k3*T1*pow(T2,2)*x + k1*pow(T2,3)*x - k3*pow(T2,3)*x - 
            k1*pow(T2,2)*T3*x + k2*pow(T2,2)*T3*x + 2*k2*T1*T2*TL*x - 2*k1*pow(T2,2)*TL*x - 
            2*k2*T1*T3*TL*x + 2*k1*T2*T3*TL*x + k1*T2*pow(TL,2)*x - k2*T2*pow(TL,2)*x - 
            k1*T3*pow(TL,2)*x + k2*T3*pow(TL,2)*x - 2*k3*T1*T2*TR*x + 2*k3*pow(T2,2)*TR*x + 
            2*k2*T1*T3*TR*x - 2*k2*T2*T3*TR*x - k2*T1*pow(TR,2)*x + k3*T1*pow(TR,2)*x + 
            k2*T2*pow(TR,2)*x - k3*T2*pow(TR,2)*x))/
       (2.*sqrt(pow(2*k2*L*T1*T2 - 2*k1*L*pow(T2,2) - 2*k2*L*T1*T3 + 2*k1*L*T2*T3,2) - 
           4*(k1*L*T2 - k2*L*T2 - k1*L*T3 + k2*L*T3)*
            (-2*k2*L*T1*T2*TL + 2*k1*L*pow(T2,2)*TL + 2*k2*L*T1*T3*TL - 2*k1*L*T2*T3*TL - 
              k1*L*T2*pow(TL,2) + k2*L*T2*pow(TL,2) + k1*L*T3*pow(TL,2) - 
              k2*L*T3*pow(TL,2) - k2*T1*pow(T2,2)*x + k3*T1*pow(T2,2)*x + k1*pow(T2,3)*x - 
              k3*pow(T2,3)*x - k1*pow(T2,2)*T3*x + k2*pow(T2,2)*T3*x + 2*k2*T1*T2*TL*x - 
              2*k1*pow(T2,2)*TL*x - 2*k2*T1*T3*TL*x + 2*k1*T2*T3*TL*x + k1*T2*pow(TL,2)*x - 
              k2*T2*pow(TL,2)*x - k1*T3*pow(TL,2)*x + k2*T3*pow(TL,2)*x - 2*k3*T1*T2*TR*x + 
              2*k3*pow(T2,2)*TR*x + 2*k2*T1*T3*TR*x - 2*k2*T2*T3*TR*x - k2*T1*pow(TR,2)*x + 
              k3*T1*pow(TR,2)*x + k2*T2*pow(TR,2)*x - k3*T2*pow(TR,2)*x))))/
    (2.*(k1*L*T2 - k2*L*T2 - k1*L*T3 + k2*L*T3)) - 
   ((L*T2 - L*T3)*(-2*k2*L*T1*T2 + 2*k1*L*pow(T2,2) + 2*k2*L*T1*T3 - 2*k1*L*T2*T3 + 
        sqrt(pow(2*k2*L*T1*T2 - 2*k1*L*pow(T2,2) - 2*k2*L*T1*T3 + 2*k1*L*T2*T3,2) - 
          4*(k1*L*T2 - k2*L*T2 - k1*L*T3 + k2*L*T3)*
           (-2*k2*L*T1*T2*TL + 2*k1*L*pow(T2,2)*TL + 2*k2*L*T1*T3*TL - 2*k1*L*T2*T3*TL - 
             k1*L*T2*pow(TL,2) + k2*L*T2*pow(TL,2) + k1*L*T3*pow(TL,2) - k2*L*T3*pow(TL,2) - 
             k2*T1*pow(T2,2)*x + k3*T1*pow(T2,2)*x + k1*pow(T2,3)*x - k3*pow(T2,3)*x - 
             k1*pow(T2,2)*T3*x + k2*pow(T2,2)*T3*x + 2*k2*T1*T2*TL*x - 2*k1*pow(T2,2)*TL*x - 
             2*k2*T1*T3*TL*x + 2*k1*T2*T3*TL*x + k1*T2*pow(TL,2)*x - k2*T2*pow(TL,2)*x - 
             k1*T3*pow(TL,2)*x + k2*T3*pow(TL,2)*x - 2*k3*T1*T2*TR*x + 2*k3*pow(T2,2)*TR*x + 
             2*k2*T1*T3*TR*x - 2*k2*T2*T3*TR*x - k2*T1*pow(TR,2)*x + k3*T1*pow(TR,2)*x + 
             k2*T2*pow(TR,2)*x - k3*T2*pow(TR,2)*x))))/
    (2.*pow(k1*L*T2 - k2*L*T2 - k1*L*T3 + k2*L*T3,2));
  }
  else {
    return -((L*pow(T2,3) - L*pow(T2,2)*T3 - 2*L*pow(T2,2)*TL + 2*L*T2*T3*TL + L*T2*pow(TL,2) - 
       L*T3*pow(TL,2) - pow(T2,3)*x + pow(T2,2)*T3*x + 2*pow(T2,2)*TL*x - 2*T2*T3*TL*x - 
       T2*pow(TL,2)*x + T3*pow(TL,2)*x)/
     sqrt(pow(-2*k3*L*T1*T2 + 2*k3*L*pow(T2,2) + 2*k2*L*T1*T3 - 2*k2*L*T2*T3,2) - 
       4*(-(k2*L*T1) + k3*L*T1 + k2*L*T2 - k3*L*T2)*
        (-(k2*L*T1*pow(T2,2)) + k3*L*T1*pow(T2,2) + k1*L*pow(T2,3) - k3*L*pow(T2,3) - 
          k1*L*pow(T2,2)*T3 + k2*L*pow(T2,2)*T3 + 2*k2*L*T1*T2*TL - 2*k1*L*pow(T2,2)*TL - 
          2*k2*L*T1*T3*TL + 2*k1*L*T2*T3*TL + k1*L*T2*pow(TL,2) - k2*L*T2*pow(TL,2) - 
          k1*L*T3*pow(TL,2) + k2*L*T3*pow(TL,2) + k2*T1*pow(T2,2)*x - k3*T1*pow(T2,2)*x - 
          k1*pow(T2,3)*x + k3*pow(T2,3)*x + k1*pow(T2,2)*T3*x - k2*pow(T2,2)*T3*x - 
          2*k2*T1*T2*TL*x + 2*k1*pow(T2,2)*TL*x + 2*k2*T1*T3*TL*x - 2*k1*T2*T3*TL*x - 
          k1*T2*pow(TL,2)*x + k2*T2*pow(TL,2)*x + k1*T3*pow(TL,2)*x - k2*T3*pow(TL,2)*x + 
          2*k3*T1*T2*TR*x - 2*k3*pow(T2,2)*TR*x - 2*k2*T1*T3*TR*x + 2*k2*T2*T3*TR*x + 
          k2*T1*pow(TR,2)*x - k3*T1*pow(TR,2)*x - k2*T2*pow(TR,2)*x + k3*T2*pow(TR,2)*x)));
  }

}

template <typename U>
U dTdk2_exact(U x, U TL, U TR, U L) {

  // Declaring variables
  U T1 = 0.0;
  U T2 = 50.0;
  U T3 = 100.0;
  U k1 = 1.0;
  U k2 = 2.0;
  U k3 = 6.0;

  if (x<=0.27272727) {
    return (-2*L*T1*T2 + 2*L*T1*T3 + (2*(2*L*T1*T2 - 2*L*T1*T3)*
          (2*k2*L*T1*T2 - 2*k1*L*pow(T2,2) - 2*k2*L*T1*T3 + 2*k1*L*T2*T3) - 
         4*(k1*L*T2 - k2*L*T2 - k1*L*T3 + k2*L*T3)*
          (-2*L*T1*T2*TL + 2*L*T1*T3*TL + L*T2*pow(TL,2) - L*T3*pow(TL,2) - T1*pow(T2,2)*x + 
            pow(T2,2)*T3*x + 2*T1*T2*TL*x - 2*T1*T3*TL*x - T2*pow(TL,2)*x + T3*pow(TL,2)*x + 
            2*T1*T3*TR*x - 2*T2*T3*TR*x - T1*pow(TR,2)*x + T2*pow(TR,2)*x) - 
         4*(-(L*T2) + L*T3)*(-2*k2*L*T1*T2*TL + 2*k1*L*pow(T2,2)*TL + 2*k2*L*T1*T3*TL - 
            2*k1*L*T2*T3*TL - k1*L*T2*pow(TL,2) + k2*L*T2*pow(TL,2) + k1*L*T3*pow(TL,2) - 
            k2*L*T3*pow(TL,2) - k2*T1*pow(T2,2)*x + k3*T1*pow(T2,2)*x + k1*pow(T2,3)*x - 
            k3*pow(T2,3)*x - k1*pow(T2,2)*T3*x + k2*pow(T2,2)*T3*x + 2*k2*T1*T2*TL*x - 
            2*k1*pow(T2,2)*TL*x - 2*k2*T1*T3*TL*x + 2*k1*T2*T3*TL*x + k1*T2*pow(TL,2)*x - 
            k2*T2*pow(TL,2)*x - k1*T3*pow(TL,2)*x + k2*T3*pow(TL,2)*x - 2*k3*T1*T2*TR*x + 
            2*k3*pow(T2,2)*TR*x + 2*k2*T1*T3*TR*x - 2*k2*T2*T3*TR*x - k2*T1*pow(TR,2)*x + 
            k3*T1*pow(TR,2)*x + k2*T2*pow(TR,2)*x - k3*T2*pow(TR,2)*x))/
       (2.*sqrt(pow(2*k2*L*T1*T2 - 2*k1*L*pow(T2,2) - 2*k2*L*T1*T3 + 2*k1*L*T2*T3,2) - 
           4*(k1*L*T2 - k2*L*T2 - k1*L*T3 + k2*L*T3)*
            (-2*k2*L*T1*T2*TL + 2*k1*L*pow(T2,2)*TL + 2*k2*L*T1*T3*TL - 2*k1*L*T2*T3*TL - 
              k1*L*T2*pow(TL,2) + k2*L*T2*pow(TL,2) + k1*L*T3*pow(TL,2) - 
              k2*L*T3*pow(TL,2) - k2*T1*pow(T2,2)*x + k3*T1*pow(T2,2)*x + k1*pow(T2,3)*x - 
              k3*pow(T2,3)*x - k1*pow(T2,2)*T3*x + k2*pow(T2,2)*T3*x + 2*k2*T1*T2*TL*x - 
              2*k1*pow(T2,2)*TL*x - 2*k2*T1*T3*TL*x + 2*k1*T2*T3*TL*x + k1*T2*pow(TL,2)*x - 
              k2*T2*pow(TL,2)*x - k1*T3*pow(TL,2)*x + k2*T3*pow(TL,2)*x - 2*k3*T1*T2*TR*x + 
              2*k3*pow(T2,2)*TR*x + 2*k2*T1*T3*TR*x - 2*k2*T2*T3*TR*x - k2*T1*pow(TR,2)*x + 
              k3*T1*pow(TR,2)*x + k2*T2*pow(TR,2)*x - k3*T2*pow(TR,2)*x))))/
    (2.*(k1*L*T2 - k2*L*T2 - k1*L*T3 + k2*L*T3)) - 
   ((-(L*T2) + L*T3)*(-2*k2*L*T1*T2 + 2*k1*L*pow(T2,2) + 2*k2*L*T1*T3 - 2*k1*L*T2*T3 + 
        sqrt(pow(2*k2*L*T1*T2 - 2*k1*L*pow(T2,2) - 2*k2*L*T1*T3 + 2*k1*L*T2*T3,2) - 
          4*(k1*L*T2 - k2*L*T2 - k1*L*T3 + k2*L*T3)*
           (-2*k2*L*T1*T2*TL + 2*k1*L*pow(T2,2)*TL + 2*k2*L*T1*T3*TL - 2*k1*L*T2*T3*TL - 
             k1*L*T2*pow(TL,2) + k2*L*T2*pow(TL,2) + k1*L*T3*pow(TL,2) - k2*L*T3*pow(TL,2) - 
             k2*T1*pow(T2,2)*x + k3*T1*pow(T2,2)*x + k1*pow(T2,3)*x - k3*pow(T2,3)*x - 
             k1*pow(T2,2)*T3*x + k2*pow(T2,2)*T3*x + 2*k2*T1*T2*TL*x - 2*k1*pow(T2,2)*TL*x - 
             2*k2*T1*T3*TL*x + 2*k1*T2*T3*TL*x + k1*T2*pow(TL,2)*x - k2*T2*pow(TL,2)*x - 
             k1*T3*pow(TL,2)*x + k2*T3*pow(TL,2)*x - 2*k3*T1*T2*TR*x + 2*k3*pow(T2,2)*TR*x + 
             2*k2*T1*T3*TR*x - 2*k2*T2*T3*TR*x - k2*T1*pow(TR,2)*x + k3*T1*pow(TR,2)*x + 
             k2*T2*pow(TR,2)*x - k3*T2*pow(TR,2)*x))))/
    (2.*pow(k1*L*T2 - k2*L*T2 - k1*L*T3 + k2*L*T3,2));
  }
  else {
    return (-2*L*T1*T3 + 2*L*T2*T3 + (2*(2*L*T1*T3 - 2*L*T2*T3)*
          (-2*k3*L*T1*T2 + 2*k3*L*pow(T2,2) + 2*k2*L*T1*T3 - 2*k2*L*T2*T3) - 
         4*(-(k2*L*T1) + k3*L*T1 + k2*L*T2 - k3*L*T2)*
          (-(L*T1*pow(T2,2)) + L*pow(T2,2)*T3 + 2*L*T1*T2*TL - 2*L*T1*T3*TL - L*T2*pow(TL,2) + 
            L*T3*pow(TL,2) + T1*pow(T2,2)*x - pow(T2,2)*T3*x - 2*T1*T2*TL*x + 2*T1*T3*TL*x + 
            T2*pow(TL,2)*x - T3*pow(TL,2)*x - 2*T1*T3*TR*x + 2*T2*T3*TR*x + T1*pow(TR,2)*x - 
            T2*pow(TR,2)*x) - 4*(-(L*T1) + L*T2)*
          (-(k2*L*T1*pow(T2,2)) + k3*L*T1*pow(T2,2) + k1*L*pow(T2,3) - k3*L*pow(T2,3) - 
            k1*L*pow(T2,2)*T3 + k2*L*pow(T2,2)*T3 + 2*k2*L*T1*T2*TL - 2*k1*L*pow(T2,2)*TL - 
            2*k2*L*T1*T3*TL + 2*k1*L*T2*T3*TL + k1*L*T2*pow(TL,2) - k2*L*T2*pow(TL,2) - 
            k1*L*T3*pow(TL,2) + k2*L*T3*pow(TL,2) + k2*T1*pow(T2,2)*x - k3*T1*pow(T2,2)*x - 
            k1*pow(T2,3)*x + k3*pow(T2,3)*x + k1*pow(T2,2)*T3*x - k2*pow(T2,2)*T3*x - 
            2*k2*T1*T2*TL*x + 2*k1*pow(T2,2)*TL*x + 2*k2*T1*T3*TL*x - 2*k1*T2*T3*TL*x - 
            k1*T2*pow(TL,2)*x + k2*T2*pow(TL,2)*x + k1*T3*pow(TL,2)*x - k2*T3*pow(TL,2)*x + 
            2*k3*T1*T2*TR*x - 2*k3*pow(T2,2)*TR*x - 2*k2*T1*T3*TR*x + 2*k2*T2*T3*TR*x + 
            k2*T1*pow(TR,2)*x - k3*T1*pow(TR,2)*x - k2*T2*pow(TR,2)*x + k3*T2*pow(TR,2)*x))/
       (2.*sqrt(pow(-2*k3*L*T1*T2 + 2*k3*L*pow(T2,2) + 2*k2*L*T1*T3 - 2*k2*L*T2*T3,2) - 
           4*(-(k2*L*T1) + k3*L*T1 + k2*L*T2 - k3*L*T2)*
            (-(k2*L*T1*pow(T2,2)) + k3*L*T1*pow(T2,2) + k1*L*pow(T2,3) - k3*L*pow(T2,3) - 
              k1*L*pow(T2,2)*T3 + k2*L*pow(T2,2)*T3 + 2*k2*L*T1*T2*TL - 2*k1*L*pow(T2,2)*TL - 
              2*k2*L*T1*T3*TL + 2*k1*L*T2*T3*TL + k1*L*T2*pow(TL,2) - k2*L*T2*pow(TL,2) - 
              k1*L*T3*pow(TL,2) + k2*L*T3*pow(TL,2) + k2*T1*pow(T2,2)*x - 
              k3*T1*pow(T2,2)*x - k1*pow(T2,3)*x + k3*pow(T2,3)*x + k1*pow(T2,2)*T3*x - 
              k2*pow(T2,2)*T3*x - 2*k2*T1*T2*TL*x + 2*k1*pow(T2,2)*TL*x + 2*k2*T1*T3*TL*x - 
              2*k1*T2*T3*TL*x - k1*T2*pow(TL,2)*x + k2*T2*pow(TL,2)*x + k1*T3*pow(TL,2)*x - 
              k2*T3*pow(TL,2)*x + 2*k3*T1*T2*TR*x - 2*k3*pow(T2,2)*TR*x - 2*k2*T1*T3*TR*x + 
              2*k2*T2*T3*TR*x + k2*T1*pow(TR,2)*x - k3*T1*pow(TR,2)*x - k2*T2*pow(TR,2)*x + 
              k3*T2*pow(TR,2)*x))))/(2.*(-(k2*L*T1) + k3*L*T1 + k2*L*T2 - k3*L*T2)) - 
   ((-(L*T1) + L*T2)*(2*k3*L*T1*T2 - 2*k3*L*pow(T2,2) - 2*k2*L*T1*T3 + 2*k2*L*T2*T3 + 
        sqrt(pow(-2*k3*L*T1*T2 + 2*k3*L*pow(T2,2) + 2*k2*L*T1*T3 - 2*k2*L*T2*T3,2) - 
          4*(-(k2*L*T1) + k3*L*T1 + k2*L*T2 - k3*L*T2)*
           (-(k2*L*T1*pow(T2,2)) + k3*L*T1*pow(T2,2) + k1*L*pow(T2,3) - k3*L*pow(T2,3) - 
             k1*L*pow(T2,2)*T3 + k2*L*pow(T2,2)*T3 + 2*k2*L*T1*T2*TL - 2*k1*L*pow(T2,2)*TL - 
             2*k2*L*T1*T3*TL + 2*k1*L*T2*T3*TL + k1*L*T2*pow(TL,2) - k2*L*T2*pow(TL,2) - 
             k1*L*T3*pow(TL,2) + k2*L*T3*pow(TL,2) + k2*T1*pow(T2,2)*x - k3*T1*pow(T2,2)*x - 
             k1*pow(T2,3)*x + k3*pow(T2,3)*x + k1*pow(T2,2)*T3*x - k2*pow(T2,2)*T3*x - 
             2*k2*T1*T2*TL*x + 2*k1*pow(T2,2)*TL*x + 2*k2*T1*T3*TL*x - 2*k1*T2*T3*TL*x - 
             k1*T2*pow(TL,2)*x + k2*T2*pow(TL,2)*x + k1*T3*pow(TL,2)*x - k2*T3*pow(TL,2)*x + 
             2*k3*T1*T2*TR*x - 2*k3*pow(T2,2)*TR*x - 2*k2*T1*T3*TR*x + 2*k2*T2*T3*TR*x + 
             k2*T1*pow(TR,2)*x - k3*T1*pow(TR,2)*x - k2*T2*pow(TR,2)*x + k3*T2*pow(TR,2)*x)))
      )/(2.*pow(-(k2*L*T1) + k3*L*T1 + k2*L*T2 - k3*L*T2,2));
  }


}

template <typename U>
U dTdk3_exact(U x, U TL, U TR, U L) {

  // Declaring variables
  U T1 = 0.0;
  U T2 = 50.0;
  U T3 = 100.0;
  U k1 = 1.0;
  U k2 = 2.0;
  U k3 = 6.0;

  if (x<=0.27272727) {
    return -((T1*pow(T2,2)*x - pow(T2,3)*x - 2*T1*T2*TR*x + 2*pow(T2,2)*TR*x + T1*pow(TR,2)*x - 
       T2*pow(TR,2)*x)/
     sqrt(pow(2*k2*L*T1*T2 - 2*k1*L*pow(T2,2) - 2*k2*L*T1*T3 + 2*k1*L*T2*T3,2) - 
       4*(k1*L*T2 - k2*L*T2 - k1*L*T3 + k2*L*T3)*
        (-2*k2*L*T1*T2*TL + 2*k1*L*pow(T2,2)*TL + 2*k2*L*T1*T3*TL - 2*k1*L*T2*T3*TL - 
          k1*L*T2*pow(TL,2) + k2*L*T2*pow(TL,2) + k1*L*T3*pow(TL,2) - k2*L*T3*pow(TL,2) - 
          k2*T1*pow(T2,2)*x + k3*T1*pow(T2,2)*x + k1*pow(T2,3)*x - k3*pow(T2,3)*x - 
          k1*pow(T2,2)*T3*x + k2*pow(T2,2)*T3*x + 2*k2*T1*T2*TL*x - 2*k1*pow(T2,2)*TL*x - 
          2*k2*T1*T3*TL*x + 2*k1*T2*T3*TL*x + k1*T2*pow(TL,2)*x - k2*T2*pow(TL,2)*x - 
          k1*T3*pow(TL,2)*x + k2*T3*pow(TL,2)*x - 2*k3*T1*T2*TR*x + 2*k3*pow(T2,2)*TR*x + 
          2*k2*T1*T3*TR*x - 2*k2*T2*T3*TR*x - k2*T1*pow(TR,2)*x + k3*T1*pow(TR,2)*x + 
          k2*T2*pow(TR,2)*x - k3*T2*pow(TR,2)*x)));
  }
  else {
    return (2*L*T1*T2 - 2*L*pow(T2,2) + (2*(-2*L*T1*T2 + 2*L*pow(T2,2))*
          (-2*k3*L*T1*T2 + 2*k3*L*pow(T2,2) + 2*k2*L*T1*T3 - 2*k2*L*T2*T3) - 
         4*(-(k2*L*T1) + k3*L*T1 + k2*L*T2 - k3*L*T2)*
          (L*T1*pow(T2,2) - L*pow(T2,3) - T1*pow(T2,2)*x + pow(T2,3)*x + 2*T1*T2*TR*x - 
            2*pow(T2,2)*TR*x - T1*pow(TR,2)*x + T2*pow(TR,2)*x) - 
         4*(L*T1 - L*T2)*(-(k2*L*T1*pow(T2,2)) + k3*L*T1*pow(T2,2) + k1*L*pow(T2,3) - 
            k3*L*pow(T2,3) - k1*L*pow(T2,2)*T3 + k2*L*pow(T2,2)*T3 + 2*k2*L*T1*T2*TL - 
            2*k1*L*pow(T2,2)*TL - 2*k2*L*T1*T3*TL + 2*k1*L*T2*T3*TL + k1*L*T2*pow(TL,2) - 
            k2*L*T2*pow(TL,2) - k1*L*T3*pow(TL,2) + k2*L*T3*pow(TL,2) + k2*T1*pow(T2,2)*x - 
            k3*T1*pow(T2,2)*x - k1*pow(T2,3)*x + k3*pow(T2,3)*x + k1*pow(T2,2)*T3*x - 
            k2*pow(T2,2)*T3*x - 2*k2*T1*T2*TL*x + 2*k1*pow(T2,2)*TL*x + 2*k2*T1*T3*TL*x - 
            2*k1*T2*T3*TL*x - k1*T2*pow(TL,2)*x + k2*T2*pow(TL,2)*x + k1*T3*pow(TL,2)*x - 
            k2*T3*pow(TL,2)*x + 2*k3*T1*T2*TR*x - 2*k3*pow(T2,2)*TR*x - 2*k2*T1*T3*TR*x + 
            2*k2*T2*T3*TR*x + k2*T1*pow(TR,2)*x - k3*T1*pow(TR,2)*x - k2*T2*pow(TR,2)*x + 
            k3*T2*pow(TR,2)*x))/
       (2.*sqrt(pow(-2*k3*L*T1*T2 + 2*k3*L*pow(T2,2) + 2*k2*L*T1*T3 - 2*k2*L*T2*T3,2) - 
           4*(-(k2*L*T1) + k3*L*T1 + k2*L*T2 - k3*L*T2)*
            (-(k2*L*T1*pow(T2,2)) + k3*L*T1*pow(T2,2) + k1*L*pow(T2,3) - k3*L*pow(T2,3) - 
              k1*L*pow(T2,2)*T3 + k2*L*pow(T2,2)*T3 + 2*k2*L*T1*T2*TL - 2*k1*L*pow(T2,2)*TL - 
              2*k2*L*T1*T3*TL + 2*k1*L*T2*T3*TL + k1*L*T2*pow(TL,2) - k2*L*T2*pow(TL,2) - 
              k1*L*T3*pow(TL,2) + k2*L*T3*pow(TL,2) + k2*T1*pow(T2,2)*x - 
              k3*T1*pow(T2,2)*x - k1*pow(T2,3)*x + k3*pow(T2,3)*x + k1*pow(T2,2)*T3*x - 
              k2*pow(T2,2)*T3*x - 2*k2*T1*T2*TL*x + 2*k1*pow(T2,2)*TL*x + 2*k2*T1*T3*TL*x - 
              2*k1*T2*T3*TL*x - k1*T2*pow(TL,2)*x + k2*T2*pow(TL,2)*x + k1*T3*pow(TL,2)*x - 
              k2*T3*pow(TL,2)*x + 2*k3*T1*T2*TR*x - 2*k3*pow(T2,2)*TR*x - 2*k2*T1*T3*TR*x + 
              2*k2*T2*T3*TR*x + k2*T1*pow(TR,2)*x - k3*T1*pow(TR,2)*x - k2*T2*pow(TR,2)*x + 
              k3*T2*pow(TR,2)*x))))/(2.*(-(k2*L*T1) + k3*L*T1 + k2*L*T2 - k3*L*T2)) - 
   ((L*T1 - L*T2)*(2*k3*L*T1*T2 - 2*k3*L*pow(T2,2) - 2*k2*L*T1*T3 + 2*k2*L*T2*T3 + 
        sqrt(pow(-2*k3*L*T1*T2 + 2*k3*L*pow(T2,2) + 2*k2*L*T1*T3 - 2*k2*L*T2*T3,2) - 
          4*(-(k2*L*T1) + k3*L*T1 + k2*L*T2 - k3*L*T2)*
           (-(k2*L*T1*pow(T2,2)) + k3*L*T1*pow(T2,2) + k1*L*pow(T2,3) - k3*L*pow(T2,3) - 
             k1*L*pow(T2,2)*T3 + k2*L*pow(T2,2)*T3 + 2*k2*L*T1*T2*TL - 2*k1*L*pow(T2,2)*TL - 
             2*k2*L*T1*T3*TL + 2*k1*L*T2*T3*TL + k1*L*T2*pow(TL,2) - k2*L*T2*pow(TL,2) - 
             k1*L*T3*pow(TL,2) + k2*L*T3*pow(TL,2) + k2*T1*pow(T2,2)*x - k3*T1*pow(T2,2)*x - 
             k1*pow(T2,3)*x + k3*pow(T2,3)*x + k1*pow(T2,2)*T3*x - k2*pow(T2,2)*T3*x - 
             2*k2*T1*T2*TL*x + 2*k1*pow(T2,2)*TL*x + 2*k2*T1*T3*TL*x - 2*k1*T2*T3*TL*x - 
             k1*T2*pow(TL,2)*x + k2*T2*pow(TL,2)*x + k1*T3*pow(TL,2)*x - k2*T3*pow(TL,2)*x + 
             2*k3*T1*T2*TR*x - 2*k3*pow(T2,2)*TR*x - 2*k2*T1*T3*TR*x + 2*k2*T2*T3*TR*x + 
             k2*T1*pow(TR,2)*x - k3*T1*pow(TR,2)*x - k2*T2*pow(TR,2)*x + k3*T2*pow(TR,2)*x)))
      )/(2.*pow(-(k2*L*T1) + k3*L*T1 + k2*L*T2 - k3*L*T2,2));
  }


}

template <typename U>
U L2_T(const mesh<U>& g, const arma::Col<U>& T_n, const int ngpts) {

  // Declaring variables
  std::vector<element<U> > elements = g.get_elements();
  std::vector<U> xi(ngpts);
  std::vector<U> w(ngpts);
  std::vector<U> nodes = g.get_nodes();
  std::vector<int> con;
  U x,Tn,detJ;
  unsigned int nelem = g.get_num_elem();
  
  // Getting Gauss points and weights
  gdata<U>(ngpts,xi,w);

  // Numerically integrating
  U sum = 0.0;
  for (unsigned int en=0; en<nelem; ++en) {

    con = elements[en].get_connectivity();
    for (int i=0; i<ngpts; ++i) {
      x = (U) 0;
      Tn = (U) 0;
      detJ = elements[en].get_detJ(xi[i],nodes);

      // Recovering x from xi and determining T_numerical(xi)
      for (int j=0; j<2; ++j) {
        x  += nodes[con[j]]*elements[en].N(j,xi[i]);
        Tn += T_n(con[j])*elements[en].N(j,xi[i]);
      }
      
      sum += w[i]*pow(Tn - T_exact<U>(x,0.0,100.0,1.0),2)*detJ;

    }
    
  }
  
  return sqrt(sum);

}

template <typename U> 
void compute_sensitivity(const mesh<U>& m, const int ngpts, arma::Mat<U> K, const arma::Col<U>& T, const U k1, const U k2, const U k3, const U dk1, const U dk2, const U dk3, arma::Col<U>& dT) {

  // Declaring variables
  int max_iter = 1000;
  int i = 0;
  U eps = (U) 100.0;
  U tol = 1e-12;
  unsigned int nnodes = m.get_num_nodes();
  arma::Mat<U> dK(nnodes,nnodes);
  arma::Col<U> dF(nnodes);
  arma::Col<std::complex<U> > Tn(T,arma::zeros<arma::Col<U> >(T.n_elem));
  arma::Col<std::complex<U> > Tnp1(T,arma::zeros<arma::Col<U> >(T.n_elem));
  arma::Col<U> dA,TpdT;
  arma::Col<U> F(nnodes);
  arma::Col<U> dT_n(dT), dT_np1;
  dT_n.zeros();

  // Use solution from finite difference to check below eqn
  // [K] {du} = {df} - [dK(k1,k2,k3,T)] {u(k1,k2,k3)}
  //             ^      ^
  //             |      |
  // [dK(u+du)]{u+du} = - [K(u+du)]{du} 

  // [dK(u+du_n)]{u+du_n} = - [K(u+du_n)]{du_np1}
  // du_np1 == du_n
  // [dK(u)]{u} = - [K(u)]{du_np1}
   
  while ((eps>tol)&&(i<max_iter)) {

    // Assembling the systems of equations
    TpdT = T + dT_n;
    assemble(m,ngpts,TpdT,k1,k2,k3,K,F);
    apply_dirichlet<U>(m,(U) 0.0,(U) 100.0,K,F);
    assemble_perturbed<U,std::complex<U> >(m,ngpts,TpdT,std::complex<U>(k1,dk1),std::complex<U>(k2,dk2),std::complex<U>(k3,dk3),dK,dF);
    apply_dirichlet<U>(m,(U) 0.0,(U) 100.0,dK,dF);

    // Setting up system of eqns
    dA = -dK*TpdT + dF;

    // Solving
    arma::solve(dT_np1,K,dA);

    // Computing residual
    eps = arma::norm(dT_np1-dT_n)/arma::norm(dT_np1);
    std::cout << "Residual = " << eps << " Iteration = " << i << "\n";
    dT_n = dT_np1;
    ++i;

  }

  dT = dT_np1/(dk1+dk2+dk3);

}

