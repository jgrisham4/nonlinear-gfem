#ifndef FORMULATIONHEADER
#define FORMULATIONHEADER

/**
 * \file formulation.h
 *
 * This header contains the formulation used in the finite element
 * method.  That is, the weak form in the sample integrands function.
 *
 * \author James Grisham
 * \date 11/05/2015
 */

#include <limits>
#include <vector>
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
template <typename A>
A conductivity(A T, A k1, A k2, A k3) {
  A T1 = 0.0;
  A T2 = 50.0;
  A T3 = 100.0;
  if ((T1<=T)&&(T<=T2)) {
    return k1*(1.0 - (T-T1)/(T2-T1)) + k2*((T-T1)/(T2-T1));
  }
  else if ((T2<T)&&(T<=T3)) {
    return k2*(1.0 - (T-T2)/(T3-T2)) + k3*((T-T2)/(T3-T2));
  }
  else {
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

template <typename T>
void sample_integrands(const element<T>& elem, const T xi, const std::vector<T>& nodes, const arma::Col<T>& temperature, const T k1, const T k2, const T k3, arma::Mat<T>& K_local, arma::Col<T>& F_local) {

  // Declaring variables
  K_local.zeros();
  F_local.zeros();
  arma::Col<T> k(temperature.n_elem);
  T Tval = (T) 0;
  T Nval;
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
      K_local(i,j) = conductivity<T>(Tval,k1,k2,k3)*dN_i*dN_j*detJ;
    }
    F_local(i) = (T) 0;
  }

}


#endif
