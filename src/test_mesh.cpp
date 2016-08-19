#include <iostream>
#include "mesh.h"

int main() {

  mesh<double> m(10, 1, 0.0, 1.0);
  m.generate();

  std::cout << m << std::endl;

  return 0;

}
