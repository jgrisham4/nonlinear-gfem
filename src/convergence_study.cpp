#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "armadillo"
#include "mesh.h"
#include "formulation.h"
#include "gdata.h"

#define LINEAR 1

// Prototypes
template <typename T> void write_data(const std::string& filename, const mesh<T>& m, const T L2norm, const arma::Col<T>& Tn);
template <typename T> void integrate(const element<T>& elem, const int n_gauss_pts, const std::vector<T>& nodes, const arma::Col<T>& temperature, const T k1, const T k2, const T k3, arma::Mat<T>& K_e, arma::Col<T>& F_e);
template <typename T> void assemble(const mesh<T>& m, const int n_gauss_pts, const arma::Col<T>& temperature, const T k1, const T k2, const T k3, arma::Mat<T>& K, arma::Col<T>& F);
template <typename T> void apply_dirichlet(const mesh<T>& m, T T_left, T T_right, arma::Mat<T>& K, arma::Col<T>& F);
template <typename T> arma::Col<T> solve(const arma::Mat<T>& K, const arma::Col<T>& F);
template <typename T> void iterate(const mesh<T>& m, const int ngpts, const T tol, const int max_iter, arma::Col<T>& temperature, const T k1, const T k2, const T k3);
template <typename U> U T_exact(U x, U TL, U TR, U L);
template <typename U> U dTdk1(U x, U TL, U TR, U L);
template <typename U> U dTdk2(U x, U TL, U TR, U L);
template <typename U> U dTdk3(U x, U TL, U TR, U L);
template <typename U> U L2_T(const mesh<U>& g, const arma::Col<U>& T_n, const int ngpts);

// Main code 
int main() {

  // Declaring variables
  int nnodes_c,nnodes_m,nnodes_f;
  int nelem_coarse = 10;
  int nelem_medium = 100;
  int nelem_fine = 200;
  double k1,k2,k3;
  double dk1,dk2,dk3;
  k1 = 1.0;
  k2 = 2.0;
  k3 = 6.0;
  dk1 = k1/100.0;
  dk2 = k2/100.0;
  dk3 = k3/100.0;

  // Creating meshes
  mesh<double> coarse(nelem_coarse,LINEAR,0.0,1.0);
  mesh<double> medium(nelem_medium,LINEAR,0.0,1.0);
  mesh<double> fine(nelem_fine,LINEAR,0.0,1.0);
  coarse.generate();
  medium.generate();
  fine.generate();
  nnodes_c = coarse.get_num_nodes();
  nnodes_m = medium.get_num_nodes();
  nnodes_f = fine.get_num_nodes();
  
  // Initial guess for temperature
  arma::Col<double> Tc = arma::ones<arma::Col<double> >(nnodes_c)*50.0;
  arma::Col<double> Tm = arma::ones<arma::Col<double> >(nnodes_m)*50.0;
  arma::Col<double> Tf = arma::ones<arma::Col<double> >(nnodes_f)*50.0;

  // Iterating
  iterate<double>(coarse,1,1.0e-12,1000,Tc,k1,k2,k3);
  iterate<double>(medium,1,1.0e-12,1000,Tm,k1,k2,k3);
  iterate<double>(fine,  1,1.0e-12,1000,Tf,k1,k2,k3);

  // Finding L2-norm
  double L2_c,L2_m,L2_f;
  L2_c = L2_T(coarse,Tc,2);
  L2_m = L2_T(medium,Tm,2);
  L2_f = L2_T(fine,Tf,2);

  std::cout << "L2 norms: " << std::endl;
  std::cout << "coarse -> " << L2_c << std::endl;
  std::cout << "medium -> " << L2_m << std::endl;
  std::cout << "fine   -> " << L2_f << std::endl;

  // Writing final data to file
  std::string fname_c = "coarse.dat";
  std::string fname_m = "medium.dat";
  std::string fname_f = "fine.dat";
  write_data(fname_c, coarse, L2_c, Tc);
  write_data(fname_m, medium, L2_m, Tm);
  write_data(fname_f,   fine, L2_f, Tf);

  // Writing exact solution to file
  arma::Col<double> Te(500);
  mesh<double> vfine(500,LINEAR,0.0,1.0);
  vfine.generate();
  std::vector<double> xe = vfine.get_nodes();
  for (unsigned int i=0; i<xe.size(); ++i) {
    Te[i] = T_exact(xe[i],0.0,100.0,1.0);
  }
  std::string fname_e = "exact.dat";
  write_data(fname_e, vfine, 0.0, Te);

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
U dTdk1(U x, U TL, U TR, U L) {

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
U dTdk2(U x, U TL, U TR, U L) {

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
U dTdk3(U x, U TL, U TR, U L) {

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
