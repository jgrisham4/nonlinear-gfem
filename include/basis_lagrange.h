#ifndef BASIS_LAGRANGE_HEADER
#define BASIS_LAGRANGE_HEADER

/**
 * \file basis_lagrange.h
 *
 * This templated set of structures is good for Lagrange basis
 * up to 5th-order.  For higher-order, more template specialization
 * must be added.  Otherwise, the compiler will complain.
 *
 * \author Ashkan Akbariyeh and James Grisham
 * \date 08/19/2015
 */

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <array>

/**
 * Definition of lagrange struct which represents Lagrange basis.
 * It has a static method named psi which returns the value at the
 * given coordinate.  There is also a typedef inside which stores
 * the value type that the basis was constructed with.  This value
 * type is carried through the code so that redundant template
 * parameters don't have to be passed.  A const static member named
 * order holds the order of the basis and a const static array holds
 * the zeros of the Lagrange polynomials.  These are used when an element
 * is converted from one order to another.
 */

template<typename T, int OR>
struct lagrange {
    T static psi(const int i, const int j, T const xi); //!< ith shape function jth derivative
    
    typedef T VT; //!< Value Type of lagrange basis like double
    const static int order = OR;
    const static std::array<T,OR+1> poly_zeros;
};

template<typename T>
struct lagrange<T, 1> {
    T static psi(const int i, const int j, T const xi);

    typedef T VT;
    const static int order = 1;
    const static std::array<T,2> poly_zeros;
};

template<typename T>
struct lagrange<T, 2> {
    T static psi(const int i, const int j, T const xi);

    typedef T VT;
    const static int order = 2;
    const static std::array<T,3> poly_zeros;
};

template<typename T>
struct lagrange<T, 3> {
    T static psi(const int i, const int j, T const xi);

    typedef T VT;
    const static int order = 3;
    const static std::array<T,4> poly_zeros;
};

template<typename T>
struct lagrange<T, 4> {
    T static psi(const int i, const int j, T const xi);

    typedef T VT;
    const static int order = 4;
    const static std::array<T,5> poly_zeros;
};

template<typename T>
struct lagrange<T, 5> {
    T static psi(const int i, const int j, T const xi);

    typedef T VT;
    const static int order = 5;
    const static std::array<T,6> poly_zeros;
};

template<typename T> const std::array<T,2> lagrange<T,1>::poly_zeros( { (T) -1.0,(T) 1.0} );
template<typename T> const std::array<T,3> lagrange<T,2>::poly_zeros( { (T) -1.0,(T) 0.0,(T) 1.0} );
template<typename T> const std::array<T,4> lagrange<T,3>::poly_zeros( { (T) -1.0,(T) -1.0/3.0 ,(T) 1.0/3.0,(T) 1.0} );
template<typename T> const std::array<T,5> lagrange<T,4>::poly_zeros( { (T) -1.0,(T) -0.5,(T) 0.0,(T) 0.5,(T) 1.0} );
template<typename T> const std::array<T,6> lagrange<T,5>::poly_zeros( { (T) -1.0,(T) -0.6,(T) -0.2,(T) 0.2,(T) 0.6,(T) 1.0} );

/**
 * Method for getting the value of the Lagrange basis.
 *
 * @param[in] i integer index which represents the i-th polynomial.
 * @param[in] j integer index which represents the j-th derivative of the i-th polynomial.
 * @param[in] xi point at which the i-th polynomial will be sampled.
 * @return the value of the polynomial.
 */
template<typename T>
T lagrange<T, 1>::psi(const int i, const int j, T const xi) {
    switch (j) {
        case 0:
            switch (i) {
                case 0:
                    return 0.5 * (1. - 1. * xi);
                case 1:
                    return 0.5 * (1. + xi);
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for first order shape functions are 0 or 1." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 1:
            switch (i) {
                case 0:
                    return -0.5;
                case 1:
                    return 0.5;
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for first order shape functions are 0 or 1." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }

        default:
            return (T) 0.0;
    }
}

template<typename T>
T lagrange<T, 2>::psi(const int i, const int j, T const xi) {
    switch (j) {
        case 0:
            switch (i) {
                case 0:
                    return 0.5 * (-1. + xi) * xi;
                case 1:
                    return 1. - 1. * xi * xi;
                case 2:
                    return 0.5 * xi * (1. + xi);
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for first order shape functions are 0,1,2." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 1:
            switch (i) {
                case 0:
                    return -0.5 + xi;
                case 1:
                    return -2. * xi;
                case 2:
                    return 0.5 + xi;
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for first order shape functions are 0,1,2." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 2:
            switch (i) {
                case 0:
                    return 1.;
                case 1:
                    return -2.;
                case 2:
                    return 1.;
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for first order shape functions are 0,1,2." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }

        default:
            return (T) 0.0;
    }
}

template<typename T>
T lagrange<T, 3>::psi(const int i, const int j, T const xi) {
    switch (j) {
        case 0:
            switch (i) {
                case 0:
                    return -0.0625 * (-1. + xi) * (-1. + 3. * xi) * (1. + 3. * xi);
                case 1:
                    return 0.5625 * (-1. + xi) * (1. + xi) * (-1. + 3. * xi);
                case 2:
                    return -0.5625 * (-1. + xi) * (1. + xi) * (1. + 3. * xi);
                case 3:
                    return 0.0625 * (1. + xi) * (-1. + 3. * xi) * (1. + 3. * xi);
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for third order shape functions are 0,1,2,3." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 1:
            switch (i) {
                case 0:
                    return 0.0625 * (1. + 18. * xi - 27. * xi * xi);
                case 1:
                    return 0.5625 * (-3. - 2. * xi + 9. * xi * xi);
                case 2:
                    return -0.5625 * (-3. + 2. * xi + 9. * xi * xi);
                case 3:
                    return 0.0625 * (-1. + 18. * xi + 27. * xi * xi);
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for third order shape functions are 0,1,2,3." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 2:
            switch (i) {
                case 0:
                    return -1.125 * (-1. + 3. * xi);
                case 1:
                    return 1.125 * (-1. + 9. * xi);
                case 2:
                    return -1.125 * (1. + 9. * xi);
                case 3:
                    return 1.125 * (1. + 3. * xi);
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for third order shape functions are 0,1,2,3." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 3:
            switch (i) {
                case 0:
                    return -3.375;
                case 1:
                    return 10.125;
                case 2:
                    return -10.125;
                case 3:
                    return 3.375;
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for third order shape functions are 0,1,2,3." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }

        default:
            return (T) 0.0;
    }
}

template<typename T>
T lagrange<T, 4>::psi(const int i, const int j, T const xi) {
    switch (j) {
        case 0:
            switch (i) {
                case 0:
                    return 0.16666666666666666 * (-1. + xi) * xi * (-1. + 2. * xi) * (1. + 2. * xi);
                case 1:
                    return -1.3333333333333333 * (-1. + xi) * xi * (1. + xi) * (-1. + 2. * xi);
                case 2:
                    return 1. - 5. * xi * xi + 4. * pow(xi, 4);
                case 3:
                    return -1.3333333333333333 * (-1. + xi) * xi * (1. + xi) * (1. + 2. * xi);
                case 4:
                    return 0.16666666666666666 * xi * (1. + xi) * (-1. + 2. * xi) * (1. + 2. * xi);
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for fourth order shape functions are 0,1,2,3,4." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 1:
            switch (i) {
                case 0:
                    return 0.16666666666666666 * (1. - 2. * xi - 12. * xi * xi + 16. * xi * xi * xi);
                case 1:
                    return -1.3333333333333333 * (1. - 4. * xi - 3. * xi * xi + 8. * xi * xi * xi);
                case 2:
                    return 2. * xi * (-5. + 8. * xi * xi);
                case 3:
                    return -1.3333333333333333 * (-1. - 4. * xi + 3. * xi * xi + 8. * xi * xi * xi);
                case 4:
                    return 0.16666666666666666 * (-1. - 2. * xi + 12. * xi * xi + 16. * xi * xi * xi);
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for fourth order shape functions are 0,1,2,3,4." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 2:
            switch (i) {
                case 0:
                    return -0.3333333333333333 - 4. * xi + 8. * xi * xi;
                case 1:
                    return 5.333333333333333 + 8. * xi - 32. * xi * xi;
                case 2:
                    return -10. + 48. * xi * xi;
                case 3:
                    return 5.333333333333333 - 8. * xi - 32. * xi * xi;
                case 4:
                    return -0.3333333333333333 + 4. * xi + 8. * xi * xi;
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for fourth order shape functions are 0,1,2,3,4." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 3:
            switch (i) {
                case 0:
                    return -4. + 16. * xi;
                case 1:
                    return 8. - 64. * xi;
                case 2:
                    return 96. * xi;
                case 3:
                    return -8. * (1. + 8. * xi);
                case 4:
                    return 4. + 16. * xi;
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for fourth order shape functions are 0,1,2,3,4." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 4:
            switch (i) {
                case 0:
                    return 16.;
                case 1:
                    return -64.;
                case 2:
                    return 96.;
                case 3:
                    return -64.;
                case 4:
                    return 16.;
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for fourth order shape functions are 0,1,2,3,4." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }

        default:
            return (T) 0.0;
    }
}

template<typename T>
T lagrange<T, 5>::psi(const int i, const int j, T const xi) {
    switch (j) {
        case 0:
            switch (i) {
                case 0:
                    return -0.0013020833333333333 * (-1. + xi) * (-3. + 5. * xi) * (-1. + 5. * xi) * (1. + 5. * xi) *
                           (3. + 5. * xi);
                case 1:
                    return 0.032552083333333336 * (-1. + xi) * (1. + xi) * (-3. + 5. * xi) * (-1. + 5. * xi) *
                           (1. + 5. * xi);
                case 2:
                    return -0.06510416666666667 * (-1. + xi) * (1. + xi) * (-3. + 5. * xi) * (-1. + 5. * xi) *
                           (3. + 5. * xi);
                case 3:
                    return 0.06510416666666667 * (-1. + xi) * (1. + xi) * (-3. + 5. * xi) * (1. + 5. * xi) *
                           (3. + 5. * xi);
                case 4:
                    return -0.032552083333333336 * (-1. + xi) * (1. + xi) * (-1. + 5. * xi) * (1. + 5. * xi) *
                           (3. + 5. * xi);
                case 5:
                    return 0.0013020833333333333 * (1. + xi) * (-3. + 5. * xi) * (-1. + 5. * xi) * (1. + 5. * xi) *
                           (3. + 5. * xi);
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for fifth order shape functions are 0,1,2,3,4,5." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 1:
            switch (i) {
                case 0:
                    return 0.0013020833333333333 *
                           (-9. - 500. * xi + 750. * xi * xi + 2500. * xi * xi * xi - 3125. * pow(xi, 4));
                case 1:
                    return 0.032552083333333336 *
                           (5. + 156. * xi - 390. * xi * xi - 300. * xi * xi * xi + 625. * pow(xi, 4));
                case 2:
                    return -0.06510416666666667 *
                           (45. + 68. * xi - 510. * xi * xi - 100. * xi * xi * xi + 625. * pow(xi, 4));
                case 3:
                    return 0.06510416666666667 *
                           (45. - 68. * xi - 510. * xi * xi + 100. * xi * xi * xi + 625. * pow(xi, 4));
                case 4:
                    return -0.032552083333333336 *
                           (5. - 156. * xi - 390. * xi * xi + 300. * xi * xi * xi + 625. * pow(xi, 4));
                case 5:
                    return 0.0013020833333333333 *
                           (9. - 500. * xi - 750. * xi * xi + 2500. * xi * xi * xi + 3125. * pow(xi, 4));
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for fifth order shape functions are 0,1,2,3,4,5." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 2:
            switch (i) {
                case 0:
                    return -0.6510416666666666 * (1. - 3. * xi - 15. * xi * xi + 25. * xi * xi * xi);
                case 1:
                    return 0.13020833333333334 * (39. - 195. * xi - 225. * xi * xi + 625. * xi * xi * xi);
                case 2:
                    return -0.2604166666666667 * (17. - 255. * xi - 75. * xi * xi + 625. * xi * xi * xi);
                case 3:
                    return 0.2604166666666667 * (-17. - 255. * xi + 75. * xi * xi + 625. * xi * xi * xi);
                case 4:
                    return -0.13020833333333334 * (-39. - 195. * xi + 225. * xi * xi + 625. * xi * xi * xi);
                case 5:
                    return 0.6510416666666666 * (-1. - 3. * xi + 15. * xi * xi + 25. * xi * xi * xi);
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for fifth order shape functions are 0,1,2,3,4,5." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 3:
            switch (i) {
                case 0:
                    return -1.953125 * (-1. - 10. * xi + 25. * xi * xi);
                case 1:
                    return 1.953125 * (-13. - 30. * xi + 125. * xi * xi);
                case 2:
                    return -3.90625 * (-17. - 10. * xi + 125. * xi * xi);
                case 3:
                    return 3.90625 * (-17. + 10. * xi + 125. * xi * xi);
                case 4:
                    return -1.953125 * (-13. + 30. * xi + 125. * xi * xi);
                case 5:
                    return 1.953125 * (-1. + 10. * xi + 25. * xi * xi);
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for fifth order shape functions are 0,1,2,3,4,5." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 4:
            switch (i) {
                case 0:
                    return -19.53125 * (-1. + 5. * xi);
                case 1:
                    return 19.53125 * (-3. + 25. * xi);
                case 2:
                    return -39.0625 * (-1. + 25. * xi);
                case 3:
                    return 39.0625 * (1. + 25. * xi);
                case 4:
                    return -19.53125 * (3. + 25. * xi);
                case 5:
                    return 19.53125 * (1. + 5. * xi);
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for fifth order shape functions are 0,1,2,3,4,5." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }
        case 5:
            switch (i) {
                case 0:
                    return -97.65625;
                case 1:
                    return 488.28125;
                case 2:
                    return -976.5625;
                case 3:
                    return 976.5625;
                case 4:
                    return -488.28125;
                case 5:
                    return 97.65625;
                default:
                    std::cerr << "\nERROR in function psi()." << std::endl;
                    std::cerr << "Indices permitted for fifth order shape functions are 0,1,2,3,4,5." << std::endl;
                    std::cerr << "i = " << i << std::endl;
                    exit(-1);
            }

        default:
            return (T) 0.0;
    }
}

#endif
