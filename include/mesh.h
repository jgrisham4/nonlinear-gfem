#ifndef MESHHEADERDEF
#define MESHHEADERDEF

/**
 * \file mesh.h
 * \class mesh
 *
 * This class represents the mesh.  It discretizes the domain
 * and instantiates element objects.
 *
 * \author James Grisham
 * \date 11/05/2015
 */

#include <iostream>
#include <vector>
#include "element.h"
#include "armadillo"

/***************************************************************\
 * Class definition                                            *
\***************************************************************/

template<typename T> 
class mesh {

  public:

    // Constructors
    mesh() {};
    mesh(unsigned int num_elem, unsigned int order, T x_start, T x_stop) : n_elem{num_elem}, p{order}, x_i{x_start}, x_f{x_stop} {
      n_interior = n_elem*(p-1); // number of interior nodes per element
      n_nodes = n_elem + n_interior + 1;
      nodes.resize(n_nodes);
      elements.resize(n_elem);
      dx = (x_f - x_i)/((T) n_nodes - (T) 1);
    };

    void generate();
    inline int get_num_elem() const {return elements.size(); };
    inline int get_num_nodes() const {return nodes.size(); };
    inline std::vector<T> get_nodes() const { return nodes; };
    inline std::vector<element<T> > get_elements() const { return elements; };
    template <typename T1> friend std::ostream& operator<<(std::ostream& os, const mesh<T1>& input);

  private:
    unsigned int n_interior;
    unsigned int n_elem;
    unsigned int n_nodes;
    std::vector<T> nodes;
    std::vector<element<T> > elements;
    unsigned int p;  // order of polynomial
    T x_i;  // starting point
    T x_f;  // stopping point
    T dx;

  //friend class tensor_element<T>;

};

// Overloading ostream operator
template <typename T1>
std::ostream& operator<<(std::ostream& os, const mesh<T1>& input) {
  os << "\nNode coordinates: " << std::endl;
  for (T1 n : input.nodes) {
    os << n << std::endl;
  }

  os << "Element connectivity: " << std::endl;
  for (auto e : input.elements) {
    os << e << std::endl;
  }
  return os;
}

/***************************************************************\
 * Class implementation                                        *
\***************************************************************/

template <typename T> void mesh<T>::generate() {

  // Creating nodes
  for (int i=0; i<n_nodes; i++) {
    nodes[i] = (T) i * dx;
  }

  // Creating connectivity
  unsigned int node_ctr = 0;
  std::vector<int> element_con(p+1);
  for (unsigned int en=0; en<n_elem; en++) {
    for (unsigned int i=0; i<p+1; i++) {
      element_con[i] = i + node_ctr;
    }
    node_ctr += p;
    elements[en] = element<T>(element_con);
  }

}

#endif
