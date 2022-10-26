#include <iostream>
#include <MeshFEM/GlobalBenchmark.hh>
#include <MeshFEM/Utilities/reduce.hh>

// Positions of the polynomials nodes, assuming they are equidistant on the unit interval
template<typename Real_, size_t d>
constexpr Real_ nodePosition(size_t j) { return (d == 0) ? 0.5 : j * (1.0 / d); }

// Evaluation job for the numerator product terms of the i^th d-degree
// polynomial (to be passed to Reduce)
template<typename Real_, size_t d, size_t i>
struct ProductJob {
    template<size_t j>
    static Real_ term(Real_ x) { return x - nodePosition<Real_, d>(j); }

    using BinaryOp = std::multiplies<Real_>;
    static constexpr Real_ initializer = 1.0;
};

// Evaluation job for the outer summation terms of the i^th d-degree
// polynomial's derivative (to be passed to Reduce)
template<typename Real_, size_t d, size_t i>
struct SumJob {
    template<size_t k>
    static Real_ term(Real_ x) {
        return Reduce<d + 1, 0, ProductJob<Real_, d, i>, SkipIndices<i, k>::template Predicate>::run(x);
    }

    using BinaryOp = std::plus<Real_>;
    static constexpr Real_ initializer = 0.0;
};

/// This class implements Lagrange Polynomials, which form a basis of the polynomials suited for FEM simulations
/// \tparam d: degree of the polynomial
/// \tparam i: index of the node where the polynom is equal to 1 (index in the basis of Lagrange polynomials)
/// Let phi_i be the i-th Lagrange polynomial:
/// phi_i(x) = PROD_{j=1 to d+1, j != i}    (x - x_j) / (x_i - x_j)
/// w = PROD_{j=1 to d+1, j != i}   1 / (x_i - x_j)  can be computed at compile time, if nodes positions are known
/// Derivative: phi_i'(x) = w SUM_{k=1 to d+1, k != i} PROD_{j=1 to d+1, j != i, j != k} (x - x_j)
template<typename Real_, size_t d, size_t i>
struct LagrangeBasisPolynomial {
    // w = prod_{j = 0, j != i}^d 1 / (x_j - x_i)
    static Real_ constexpr compute_w(size_t j = 0) {
        return (j == d + 1) ? 1 : compute_w(j + 1) / ((j == i) ? 1 : nodePosition<Real_, d>(i) - nodePosition<Real_, d>(j));
    }

    // Evaluate normalization weight w at compile time and store it as a constant.
    static constexpr Real_ w = compute_w(0);

    // efficient method for evaluating the polynomial and its derivative
    static Real_ eval(Real_ x) { return w * Reduce<d + 1, 0, ProductJob<Real_, d, i>, SkipIndices<i>::template Predicate>::run(x); }
    static Real_ evalDerivative(Real_ x) {
        return w * Reduce<d + 1, 0, SumJob<Real_, d, i>, SkipIndices<i>::template Predicate>::run(x);
    }

    // for loop evaluation
    static Real_ evalStraightforward(Real_ x) {
        Real_ result = w;
        for (size_t j = 0; j < d + 1; ++j) {
            if (i == j) continue;
            result *= (x - nodePosition<Real_, d>(j));
        }
        return result;
    }

    // for loop derivative evaluation
    static Real_ evalDerivativeStraightforward(Real_ x) {
        Real_ result = 0;
        for (size_t k = 0; k < d + 1; ++k) {
            if (k == i) continue;
            Real_ prod = w;
            for (size_t j = 0; j < d + 1; ++j) {
                if ((j == k) || (j == i)) continue;
                prod *= (x - nodePosition<Real_, d>(j));
            }
            result += prod;
        }
        return result;
    }
};
