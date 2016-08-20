#ifndef FORMULATIONHEADER
#define FORMULATIONHEADER

/**
 * \file sacvm_formulation.h
 *
 * This header contains the formulation used in the finite element
 * method.  That is, the weak form in the sample integrands function.
 * The main difference between this header and complex_formulation.h 
 * is one small modification to the sample_integrands function.  The
 * type on the temperature vector is changed from U to T because the
 * temperature should always be real.
 *
 * \author James Grisham
 * \date 11/05/2015
 */

#include <limits>
#include <vector>
#include <complex>
#include "armadillo"

/**
 * Function for piecewise linear thermal conductivity.
 *
 * @param[in] T temperature.
 * @param[in] k1 conductivity constant.
 * @param[in] k2 conductivity constant.
 * @param[in] k3 conductivity constant.
 * @return thermal conductivity.
 *
 */
template <typename A, typename B>
B conductivity(A T, B k1, B k2, B k3) {
  A T1 = 0.0;
  A T2 = 50.0;
  A T3 = 1000.0;
  if ((std::abs(T1)<=std::abs(T))&&(std::abs(T)<=std::abs(T2))) {
    return k1*(1.0 - (T-T1)/(T2-T1)) + k2*((T-T1)/(T2-T1));
  }
  else if ((std::abs(T2)<std::abs(T))&&(std::abs(T)<=std::abs(T3))) {
    return k2*(1.0 - (T-T2)/(T3-T2)) + k3*((T-T2)/(T3-T2));
  }
  else {
    std::cout << "Warning: Temperature of " << T << " is out of range." << std::endl;
    return std::numeric_limits<A>::quiet_NaN();
  }
}

/**
 * Function for sampling integrands.
 *
 * @param[in] elem element object over which integrands are evaluated.
 * @param[in] xi coordinate in the computational domain.
 * @param[in,out] K_local contribution of point xi to the element stiffness matrix.
 * @param[in,out] F_local contribution to the element load vector.
 */

template <typename T,typename U=T>
void sample_integrands(const element<T>& elem, const T xi, const std::vector<T>& nodes, const arma::Col<U>& temperature, const U k1, const U k2, const U k3, arma::Mat<U>& K_local, arma::Col<U>& F_local) {

  // Declaring variables
  K_local.zeros();
  F_local.zeros();
  U Tval = (U) 0;
  T detJ;
  T dN_i, dN_j;
  std::vector<int> con = elem.get_connectivity();

  // Interpolating the temperature at xi by projecting it onto the 
  // basis
  for (int i=0; i<2; ++i) {
    Tval += temperature(con[i])*elem.N(i,xi);
  }

  // Assembling element stiffness
  // Note: The temperature should be evaluated at the given xi
  detJ = elem.get_detJ(xi,nodes);
  for (int i=0; i<2; ++i) {
    dN_i = elem.dN(i,xi,nodes);
    for (int j=0; j<2; ++j) {
      dN_j = elem.dN(j,xi,nodes);
      K_local(i,j) = conductivity<U,U>(Tval,k1,k2,k3)*dN_i*dN_j*detJ;
    }
    F_local(i) = (T) 0;
  }

}

#endif
