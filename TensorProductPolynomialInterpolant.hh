#ifndef TENSORPRODUCTPOLYNOMIALINTERPOLANT_HH
#define TENSORPRODUCTPOLYNOMIALINTERPOLANT_HH

#include <Eigen/Dense>
#include <array>
#include <utility>
#include <MeshFEM/Utilities/NDArray.hh>
#include <MeshFEM/NTuple.hh>
#include <MeshFEM/Future.hh>
#include <MeshFEM/function_traits.hh>
#include <MeshFEM/SymmetricMatrix.hh>

#include "TensorProductBasisPolynomial.hh"

namespace detail {
    // Helper class for evaluating a tensor product interpolant at a certain fixed point.
    template<typename T, typename Real_, size_t... Degrees>
    struct InterpolantEvaluator {
        static constexpr size_t N = sizeof...(Degrees);

        InterpolantEvaluator(const VecN_T<Real_, N>  &evalPt) : m_evalPt(evalPt) { }
        InterpolantEvaluator(      VecN_T<Real_, N> &&evalPt) : m_evalPt(std::move(evalPt)) { }

        // Agregates the value of coeff * phi_Idxs in result
        template<size_t... Idxs>
        void visit(const T &coeff) {
            if (m_isFirstTerm) result  = coeff * TensorProductBasisPolynomial<Real_, Degrees...>::template eval<Idxs...>(m_evalPt);
            else               result += coeff * TensorProductBasisPolynomial<Real_, Degrees...>::template eval<Idxs...>(m_evalPt);
            m_isFirstTerm = false;
        }

        T result;
    private:
        // Avoid having to zero-initialize the "result" accumulator (which would
        // need to be done specially for certain types like Eigen::Vector) by
        // initializing the value to the first term encountered.
        bool m_isFirstTerm = true;
        VecN_T<Real_, N> m_evalPt;
    };
}

/// An interpolant class for the test functions
/// These functions are in the span of the Tensor Product Basis Polynomials: f(x) = SUM f_i * phi_i(x)
/// Therefore evaluating them is just summing the terms f_i * phi_i, so we just have to store the values of the coefficients
/// and the evaluation is done by the visit_compile time method of the coefficients NDArray.
/// The evaluator passed agregates the sum f_i * phi_i(x) in its member "result"
template<typename T, typename Real_, size_t... Degrees>
struct TensorProductPolynomialInterpolant {
    static constexpr size_t N = sizeof...(Degrees);
    using IE = detail::InterpolantEvaluator<T, Real_, Degrees...>;

    template<typename... Args>
    T operator()(Args... coords) const {
        auto evaluator = IE(VecN_T<Real_, N>(Real_(coords)...));
        coeffs.visit_compile_time(evaluator);
        return evaluator.result;
    }

    T operator()(const VecN_T<Real_, N> &evalPt) const {
        auto evaluator = IE(evalPt);
        coeffs.visit_compile_time(evaluator);
        return evaluator.result;
    }

    const T &operator[](size_t i) const { return coeffs.get1D(i); }
          T &operator[](size_t i)       { return coeffs.get1D(i); }
    static constexpr size_t size() { return NDArray<T, N, (Degrees + 1)...>::size(); }

    // We need (Deg + 1) coefficients to define the polynomial along each dimension
    NDArray<T, N, (Degrees + 1)...> coeffs;
};


////////////////////////////////////////////////////////////////////////////////
// Utility methods for the conversion of 1D basis function index to multi-index
////////////////////////////////////////////////////////////////////////////////
template<size_t i, typename IS>
struct PrependToIndexSeq;

template<size_t i, size_t... Idxs>
struct PrependToIndexSeq<i, Future::index_sequence<Idxs...>> {
    using type = Future::index_sequence<i, Idxs...>;
};

// Base case
template<size_t Idx, size_t... Degrees>
struct BasisFunctionLabelFrom1DIndex {
    using label = Future::index_sequence<>;
};

template<size_t Idx, size_t D, size_t... Degrees>
struct BasisFunctionLabelFrom1DIndex<Idx, D, Degrees...> {
    constexpr static size_t i = Idx % (D + 1); // There are deg + 1 nodes along a given direction.
    using label = typename PrependToIndexSeq<i,
            typename BasisFunctionLabelFrom1DIndex<(Idx - i) / (D + 1), Degrees...>::label>::type;
};

////////////////////////////////////////////////////////////////////////////////
// Utility methods to compute the number of basis functions/interpolant nodes for a tensor-product
// polynomial with degrees "Degrees..."
////////////////////////////////////////////////////////////////////////////////
template<size_t... Degrees>
struct NumInterpolantNodes {
    constexpr static size_t value = 1;
};
template<size_t D, size_t... Degrees>
struct NumInterpolantNodes<D, Degrees...> {
    constexpr static size_t value = (D + 1) * NumInterpolantNodes<Degrees...>::value;
};
template<size_t... Degrees>
using NumBasisFunctions = NumInterpolantNodes<Degrees...>;


////////////////////////////////////////////////////////////////////////////////
// Utility methods to compute the index in the parameter pack of indices, given
// the degree of the Lagrange Basis polynomials
////////////////////////////////////////////////////////////////////////////////
template<size_t I, size_t Entry, size_t... Entries>
struct GetIthIndex {
    static constexpr size_t value = GetIthIndex<I - 1, Entries...>::value;
};

template<size_t Entry, size_t... Entries>
struct GetIthIndex<0, Entry, Entries...> {
    static constexpr size_t value = Entry;
};

// Construct the gradient of a certain basis function as an interpolant object.
template<typename Real_, size_t... Degrees>
struct Gradients {
    constexpr static size_t N = sizeof...(Degrees);
    using Grad = TensorProductPolynomialInterpolant<VecN_T<Real_, N>, Real_, Degrees...>;

    struct GHelper {
        GHelper(const VecN_T<Real_, N> &dx) : m_dx(dx) { }
        template<size_t... BasisFunctionIdx>
        void visit(Grad &g) {
            g.coeffs.visit_compile_time(GradientEvaluator<BasisFunctionIdx...>(m_dx));
        }
    private:
        const VecN_T<Real_, N> &m_dx;
    };

    // Get an array holding each basis function's gradient as an interpolant.
    static NDArray<Grad, N, (Degrees + 1)...> getGradients(const VecN_T<Real_, N> &dx) {
        NDArray<Grad, N, (Degrees + 1)...> result;
        result.visit_compile_time(GHelper(dx));
        return result;
    }
private:
    template<size_t... BasisFunction>
    struct GradientEvaluator {
        GradientEvaluator(const VecN_T<Real_, N> &dx)
            : m_dx(dx) { }
        // Evaluate the gradient of shape function with ND-label
        // "BasisFunction..." at the interpolation node with ND grid position
        // "NodeIdxs..."
        template<size_t... NodeIdxs>
        void visit(VecN_T<Real_, N> &result) {
            for (size_t c = 0; c < N; ++c)
                result[c] = eval_component<0, NodeIdxs...>(c);
        }

        // Evaluate the partial derivative with respect to parameter "c" at
        // node labeled "NodeIdxs..." (with position `n`):
        //      d / x_c prod_{i = I}^N (phi_{l[i]}(n_i)),
        // where l[i] is the label of the univariate basis function associated with
        // coordinate i, and I denotes coordinate handled by this level of recursion.
        template<size_t I, size_t NodeIdx, size_t... NodeIdxs>
        Real_ eval_component(size_t c) {
            constexpr size_t Deg_I   = GetIthIndex<I,       Degrees...>::value;
            constexpr size_t l_I     = GetIthIndex<I, BasisFunction...>::value;

            Real_ n_I = nodePosition<Real_, Deg_I>(NodeIdx);
            return (I != c ?  LagrangeBasisPolynomial<Real_, Deg_I, l_I>::eval(n_I)
                           : (LagrangeBasisPolynomial<Real_, Deg_I, l_I>::evalDerivative(n_I) / m_dx[c]))
                   * eval_component<I + 1, NodeIdxs...>(c);
        }

        template<size_t I> // Base case
        Real_ eval_component(size_t /* c */) {
            static_assert(I == N, "Out of bounds indexing");
            return 1;
        }

    private:
        const VecN_T<Real_, N> &m_dx;
    };
};

/// A struct for the strain E(f) of a test function f
/// E(f_{n+c})_{ij} = dphi_n/dx_j if i=j=c
///                   0.5 * dphi_n/dx_j if i=j and j!=c
///                   0.5 * dphi_n/dx_i if j=c and i!=c
///                   0                 otherwise
/// \tparam Degrees: degree of the tensorProductPolynomialInterpolant in each dimension
template<typename Real_, size_t... Degrees>
struct Strains {
    constexpr static size_t N = sizeof...(Degrees);
    using SMatrix = SymmetricMatrixValue<Real_, N>;
    using Strains_ = TensorProductPolynomialInterpolant<SMatrix, Real_, Degrees...>;

    // Get a vector holding each vector valued basis function's strain as an interpolant.
    static std::vector<Strains_> getStrains(const VecN_T<Real_, N> &stretchings) {
        auto gradients = Gradients<Real_, Degrees...>::getGradients(stretchings); // ND Array of each node's shape function gradient

        // Size is number of scalar-valued basis functions (nodes) times number of dimension (vector shape basis function)
        const size_t numNodes = gradients.size();
        std::vector<Strains_> strains(numNodes * N);

        // Compute strain of function Phi_{fctIdx * N + c}
        for (size_t fctIdx = 0; fctIdx < numNodes; fctIdx++) {
            auto &g = gradients[fctIdx];

            // c indexes the dimensions in the vector valued shape functions
            for (size_t c = 0; c < N; ++c) {
                auto &s = strains[fctIdx * N + c];

                // Fill in the nodal values of the strain interpolant
                for (size_t node = 0; node < numNodes; ++node) {
                    // Fill the strain matrix elements : strain(phi_{fctIdx * N + c})_{ij}
                    // Note: SymmetricMatrixValue are initialized to zero.
                    for (size_t i = 0; i < N; i++)
                        s[node](c,i) = 0.5 * g[node][i];
                    s[node](c,c) = g[node][c];
                }
            }
        }

        return strains;
    }
};




//---------------------------------------------------------------------------------------------------------------------
// Method to create tensor product interpolants, which are used to test the gauss quadrature routines
namespace detail {
    // Support both functions with N floating point arguments or a single N-tuple argument.
    template<typename F, typename... Args>
    typename std::enable_if<AcceptsNDistinctReals<F, sizeof...(Args)>::value, return_type<F>>::type
    sample_f(const F &f, Args&&... args) { return f(args...); }

    template<typename F, typename... Args>
    typename std::enable_if<AcceptsRealNTuple<F, sizeof...(Args)>::value, return_type<F>>::type
    sample_f(const F &f, Args&&... args) { return f(std::make_tuple(args...)); }

    template<typename Real_, size_t... Degrees, typename F, size_t... Idxs>
    TensorProductPolynomialInterpolant<return_type<F>, Real_, Degrees...>
    make_tp_interpolant_helper(const F &f, Future::index_sequence<Idxs...>) {
        using value_type = return_type<F>;
        TensorProductPolynomialInterpolant<value_type, Real_, Degrees...> interp;
        interp.coeffs.visit([&](value_type &c, const NDArrayIndex<sizeof...(Degrees)> &idx) {
                c = sample_f(f, nodePosition<Real_, Degrees>(idx[Idxs])...);
        });
        return interp;
    }
}

// Construct the interpolant of a function, automatically deducing the
// value type from F's return type.
template<typename Real_, size_t... Degrees, typename F>
TensorProductPolynomialInterpolant<return_type<F>, Real_, Degrees...>
make_tp_interpolant(const F &f) {
    return detail::make_tp_interpolant_helper<Real_, Degrees...>(f, Future::make_index_sequence<sizeof...(Degrees)>());
}

#endif /* end of include guard: TENSORPRODUCTPOLYNOMIALINTERPOLANT_HH */
