////////////////////////////////////////////////////////////////////////////////
// TensorProductBasisPolynomial
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Multivariate basis functions constructed by taking tensor products of the
//  1D Lagrange basis polynomials `phi_i`. For example, the basis function
//  associated with node {i, j, k} on a 3D element is:
//      phi_{i,j,k}(x) = phi_i(x_1) * phi_j(x_2) * phi_k(x_3)
//
//  The degree of the 1D basis functions for each dimension are specified by
//  the variadic template parameter `Degrees`. For example,
//  `TensorProductBasisPolynomial<2, 1, 3>` constructs a polynomial that is
//  degree 2 along the `x_1` axis, degree 1 along the `x_2` axis, and
//  degree 3 along the `x_3` axis.
*/
////////////////////////////////////////////////////////////////////////////////
#ifndef TENSORPRODUCTBASISPOLYNOMIAL_HH
#define TENSORPRODUCTBASISPOLYNOMIAL_HH

#include "LagrangePolynomial.hh"
#include <MeshFEM/Future.hh>
#include <MeshFEM/Utilities/NDArray.hh>

// Base case (0 dimensional)
template<typename Real_, size_t... Degrees>
struct TensorProductBasisPolynomial {
    template<size_t... idxs>
    static constexpr Real_ eval() { return 1; }
};

// Recursive case
template<typename Real_, size_t d, size_t... Degrees>
struct TensorProductBasisPolynomial<Real_, d, Degrees...> {
    // Evaluate the basis function labeled phi_<i, ...>
    // called as eval<i, ...>(ci, ...)
    template<size_t i, size_t... idxs, typename... Args>
    static Real_ eval(Real_ ci, Args... cRest) {
        static_assert(sizeof...(idxs) == sizeof...(Degrees), "Invalid number of indices");
        static_assert(sizeof...(idxs) == sizeof...(cRest),   "Invalid number of coordinates");
        return TensorProductBasisPolynomial<Real_, Degrees...>::template eval<idxs...>(cRest...) *
               LagrangeBasisPolynomial<Real_, d, i>::eval(ci);
    }

    template<size_t... idxs>
    static Real_ eval(const VecN_T<Real_, sizeof...(idxs)> &evalPt) {
        return m_eval_helper<idxs...>(evalPt, Future::make_index_sequence<sizeof...(idxs)>());
    }

private:
    // Helper function to expand the values from the "evalPoint" tuple into the arguments of eval
    template <size_t... idxs, size_t... TupleExpandSeq>
    static Real_ m_eval_helper(const VecN_T<Real_, sizeof...(idxs)> &evalPt, const Future::index_sequence<TupleExpandSeq...>) {
        return eval<idxs...>((evalPt[TupleExpandSeq])...);
    }
};

// Generate a tensor product of N Lagrange basis polynomials, all of degree "Deg".
// Replicate "Deg" N times, constructing "DegList". Then pass "DegList" as
// the template parameter for TensorProductBasisPolynomial.
template<typename Real_, size_t N, size_t Deg, size_t... DegList>
struct NDTensorProductBasisPolynomial : public NDTensorProductBasisPolynomial<Real_, N - 1, Deg, Deg, DegList...> { }; // Prepend "Deg" to DegList
template<typename Real_, size_t Deg, size_t... DegList>
struct NDTensorProductBasisPolynomial<Real_, 0, Deg, DegList...> : public TensorProductBasisPolynomial<Real_, DegList...> { };

namespace detail {
    // Helper class for evaluating a tensor product basis function at a certain fixed point.
    template<typename Real_, size_t... Degrees>
    struct BasisPolynomialEvaluator {
        static constexpr size_t N = sizeof...(Degrees);

        BasisPolynomialEvaluator(const VecN_T<Real_, N>  &evalPt) : m_evalPt(evalPt) { }
        BasisPolynomialEvaluator(      VecN_T<Real_, N> &&evalPt) : m_evalPt(std::move(evalPt)) { }

        // Agregates the value of coeff * phi_Idxs in result
        template<size_t... Idxs>
        void visit(Real_ &coeff) {
            coeff = TensorProductBasisPolynomial<Real_, Degrees...>::template eval<Idxs...>(m_evalPt);
        }

    private:
        VecN_T<Real_, N> m_evalPt;
    };

}

// Evaluate *all* basis functions at a single sample point.
template<typename Real_, size_t... Degrees>
struct TensorProductPolynomialEvaluator {
    static constexpr size_t N = sizeof...(Degrees);
    using BPS = detail::BasisPolynomialEvaluator<Real_, Degrees...>;
    static NDArray<Real_, N, (Degrees + 1)...> evaluate(const VecN_T<Real_, N> &p) {
        NDArray<Real_, N, (Degrees + 1)...> coeffs;
        auto evaluator = BPS(p);
        coeffs.visit_compile_time(evaluator);
        return coeffs;
    }
};

#endif /* end of include guard: TENSORPRODUCTBASISPOLYNOMIAL_HH */
