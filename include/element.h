#ifndef ELEMENTHEADER
#define ELEMENTHEADER

#include <vector>
#include "basis_lagrange.h"
#include "armadillo"

template <typename T>
class element {

  public:
    element() {};
    element(const std::vector<int>& con) : connectivity{con} {};
    std::vector<int> get_connectivity() const;
    void compute_jacobian(const T xi, const std::vector<T>& nodes, T& Jinv, T& detJ) const;
    T get_detJ(const T xi, const std::vector<T>& nodes) const;
    T N(const int i, const T xi) const;
    T dN(const int i, const T xi, const std::vector<T>& nodes) const;

    template <typename T1> friend std::ostream& operator<<(std::ostream& os, const element<T1>& e);

  private:
    std::vector<int> connectivity;
    //const static lagrange<T,1> basis;  // linear lagrange basis
    lagrange<T,1> basis;  // linear lagrange basis

};

// Overloading ostream operator
template <typename T1>
std::ostream& operator<<(std::ostream& os, const element<T1>& e) {
  for (int i : e.connectivity) {
    os << i << " ";
  }
  return os;
}

template <typename T> 
std::vector<int> element<T>::get_connectivity() const {
  return connectivity;
}

template <typename T>
void element<T>::compute_jacobian(const T xi, const std::vector<T>& nodes, T& Jinv, T& detJ) const {

  // Computing Jacobian (in 1D it is a scalar which represents the
  // ratio of the lengths between the computational and physical 
  // domains).
  detJ = (nodes[connectivity[1]]-nodes[connectivity[0]])/2.0;
  Jinv = ((T) 1.0)/detJ;

  // Making sure there aren't negative lengths
  if (detJ < (T) 0) {
    std::cerr << "\nError: Negative cell length.\n" << std::endl;
    std::cerr << "det(J) = " << detJ << std::endl;
    exit(-1);
  }
  
}

template <typename T>
T element<T>::get_detJ(const T xi, const std::vector<T>& nodes) const {
  T Jinv, detJ;
  compute_jacobian(xi,nodes,Jinv,detJ);
  return detJ;
}

template <typename T>
T element<T>::N(const int i, const T xi) const {
  return basis.psi(i,0,xi);
}

template <typename T>
T element<T>::dN(const int i, const T xi, const std::vector<T>& nodes) const {
  T Jinv, detJ;
  compute_jacobian(xi,nodes,Jinv,detJ);
  return basis.psi(i,1,xi)*Jinv;
}

#endif 
