////////////////////////////////////////////////////////////////////////////////
// TensorProductQuadrature.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Tensor-product Gauss quadrature rules for integrating over the hypercube
//  [0,1]^N by repeated applications of 1D quadrature.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  11/07/2017 14:00:52
////////////////////////////////////////////////////////////////////////////////
#ifndef TENSORPRODUCTQUADRATURE_HH
#define TENSORPRODUCTQUADRATURE_HH
#include <MeshFEM/function_traits.hh>
#include <MeshFEM/Future.hh>
#include <MeshFEM/NTuple.hh>
#include <MeshFEM_export.h>

////////////////////////////////////////////////////////////////////////////////
// Forward declarations of univariate and tensor product quadrature rules
////////////////////////////////////////////////////////////////////////////////
template<size_t NumNodes>
struct GaussQuadratureRule;

// Number of quadrature points for a particular rule.
template<size_t Degree, size_t... Degrees>
struct NumTPQuadPts {
    static constexpr size_t value = NumTPQuadPts<Degree>::value * NumTPQuadPts<Degrees...>::value;
};

// The n point Gauss quadrature exactly integrates degree 2 * n - 1 polynomials.
// d = 2 * n - 1 ==> n = ceil((d + 1) / 2) = floor(d/2) + 1
template<size_t Degree>
struct NumTPQuadPts<Degree> { static constexpr size_t value = (Degree / 2) + 1; };

template<size_t Degree>
using UnivariateGaussQuadrature = GaussQuadratureRule<NumTPQuadPts<Degree>::value>;

// ND Gauss quadrature for varying degree along each dimension
template<typename Real_, size_t... Degrees>
struct TensorProductQuadrature;

////////////////////////////////////////////////////////////////////////////////
// Implementation of ND tensor product quadrature rules.
////////////////////////////////////////////////////////////////////////////////
// Recursive case: construct and integrate the univariate function that
// evaluates the inner (N-1)D integral (a function of the outermost integration
// variable).
// Evaluating this univariate function at x is performed by binding the leftmost
// argument of f to x and then running the (N-1)D integration.
// As the recursion proceeds, we bind the arguments of f from left to right
// by building up a std::tuple of the args which are later passed into f (see
// NDFEvaluator below).
////////////////////////////////////////////////////////////////////////////////
template<typename Real_, size_t Degree, size_t... Degrees>
struct TensorProductQuadrature<Real_, Degree, Degrees...> {
    static constexpr size_t Dims = 1 + sizeof...(Degrees);

    template<typename F>
    static return_type<F> integrate(const F &f) {
        return integrate_ndimpl(f, std::tuple<>());
    }

    template<typename F, typename... Reals>
    static return_type<F> integrate_ndimpl(const F &f, const std::tuple<Reals...> &args) {
        return UnivariateGaussQuadrature<Degree>::integrate(
            [&](Real_ x) { return TensorProductQuadrature<Real_, Degrees...>::integrate_ndimpl(f, std::tuple_cat(args, std::make_tuple(x))); });
    }
};

////////////////////////////////////////////////////////////////////////////////
// Evaluate the integrand on the full list of arguments (args..., x).
// Support two different integrand function signatures:
//   1) functions accepting N real values
//   2) functions accepting an N-tuple of real values
////////////////////////////////////////////////////////////////////////////////
template<typename Real_, typename F, typename... Reals, size_t... Idxs>
std::enable_if_t<AcceptsNDistinctReals<F, sizeof...(Reals) + 1>::value, return_type<F>>
eval_f(const F &f, const std::tuple<Reals...> &args, Real_ x, Future::index_sequence<Idxs...>) {
    return f(std::get<Idxs>(args)..., x);
}
template<typename Real_, typename F, typename... Reals, size_t... Idxs>
std::enable_if_t<AcceptsRealNTuple<F, sizeof...(Reals) + 1, Real_>::value, return_type<F>>
eval_f(const F &f, const std::tuple<Reals...> &args, Real_ x, Future::index_sequence<Idxs...>) {
    return f(std::tuple_cat(args, std::make_tuple(x)));
}
template<typename Real_, typename F, typename... Reals, size_t... Idxs>
std::enable_if_t<AcceptsVectorND<F, sizeof...(Reals) + 1, Real_>::value, return_type<F>>
eval_f(const F &f, const std::tuple<Reals...> &args, Real_ x, Future::index_sequence<Idxs...>) {
    return f(VecN_T<Real_, sizeof...(Reals) + 1>(std::get<Idxs>(args)..., x));
}

// A univariate function formed by binding N - 1 of f's arguments to the values boundArgs
template<typename Real_, typename F, typename BoundArgs>
struct NDFEvaluator {
    NDFEvaluator(const F &f, const BoundArgs &boundArgs) : boundArgs(boundArgs), f(f) { }
    return_type<F> operator()(Real_ x) const { return eval_f(f, boundArgs, x, Future::make_index_sequence<std::tuple_size<BoundArgs>::value>()); }
    const BoundArgs &boundArgs;
    const F &f;
};

template<typename Real_, typename F, typename BoundArgs>
NDFEvaluator<Real_, F, BoundArgs> make_NDFEvaluator(const F&f, const BoundArgs &boundArgs) { return NDFEvaluator<Real_, F, BoundArgs>(f, boundArgs); }

// Base case: integrate a univariate function
// (with either a 1-tuple argument or a single real argument)
template<typename Real_, size_t Degree>
struct TensorProductQuadrature<Real_, Degree> {
    template<typename F>
    static typename std::enable_if<AcceptsRealNTuple<F, 1>::value, return_type<F>>::type
    integrate(const F &f) { return UnivariateGaussQuadrature<Degree>::integrate([&](Real_ x) { return f(std::make_tuple(x)); }); }
    template<typename F>
    static typename std::enable_if<AcceptsNDistinctReals<F, 1>::value, return_type<F>>::type
    integrate(const F &f) { return UnivariateGaussQuadrature<Degree>::integrate(f); }

    template<typename F, typename... Reals>
    static typename function_traits<F>::result_type integrate_ndimpl(const F &f, const std::tuple<Reals...> &args) {
        return integrate(make_NDFEvaluator<Real_>(f, args));
    }
};

////////////////////////////////////////////////////////////////////////////////
// 1D Gauss Quadrature Implementation up to 5pt rule (degree 9)
// These only work on functions accepting a single Real_ value (not a 1-tuple).
// See Derivations/HypercubeGaussQuadrature.nb for the node/weight derivations.
// To build the final result in a single accumulator variable (thereby avoiding
// creating any temporaries), we use the ratios of successive quadrature
// weights instead of the weights themselves.
////////////////////////////////////////////////////////////////////////////////
template<> struct GaussQuadratureRule<1> {
    template<typename F>
    static return_type<F> integrate(const F &f) { return f(0.5); }
};

template<> struct GaussQuadratureRule<2> {
    template<typename F>
    static return_type<F> integrate(const F &f) {
        return_type<F> result(
                  f(0.21132486540518711775));
        result += f(0.78867513459481288225);
        result *= 0.5;
        return result;
    }
};

template<> struct GaussQuadratureRule<3> {
    template<typename F>
    static return_type<F> integrate(const F &f) {
        return_type<F> result(f(0.5));
        result *= 8 / 5.;
        result += f(0.11270166537925831148);
        result += f(0.88729833462074168852);
        result *= 5 / 18.;
        return result;
    }
};

template<> struct GaussQuadratureRule<4> {
    template<typename F>
    static return_type<F> integrate(const F &f) {
        return_type<F> result(
                  f(0.33000947820757186760));
        result += f(0.66999052179242813240);
        result *= 0.326072577431273071 / 0.1739274225687269287;
        result += f(0.93056815579702628761);
        result += f(0.06943184420297371239);
        result *= 0.1739274225687269287;
        return result;
    }
};

template<> struct GaussQuadratureRule<5> {
    template<typename F>
    static return_type<F> integrate(const F &f) {
        return_type<F> result(f(0.5));
        result *= (64 / 225.) / 0.239314335249683234;
        result += f(0.23076534494715845448);
        result += f(0.76923465505284154552);
        result *= 0.239314335249683234 / 0.1184634425280945438;
        result += f(0.04691007703066800360);
        result += f(0.95308992296933199640);
        result *= 0.1184634425280945438;
        return result;
    }
};

// NumPts-point Gaussian quadrature for the 1D interval [0, 1]
template<size_t NumPts>
struct MESHFEM_EXPORT UnivariateQuadratureTable;

template<>
struct MESHFEM_EXPORT UnivariateQuadratureTable<1> {
    static constexpr std::array<double, 1> points{{ 0.5 }};
    static constexpr std::array<double, 1> weights{{ 1.0 }};
};

template<>
struct MESHFEM_EXPORT UnivariateQuadratureTable<2> {
    static constexpr std::array<double, 2> points{{0.21132486540518711775, 0.78867513459481288225}};
    static constexpr std::array<double, 2> weights{{0.5, 0.5}};
};

template<>
struct MESHFEM_EXPORT UnivariateQuadratureTable<3> {
    static constexpr std::array<double, 3> points{{0.5, 0.11270166537925831148, 0.88729833462074168852}};
    static constexpr std::array<double, 3> weights{{0.4444444444444444444, 0.27777777777777777778, 0.27777777777777777778}};
};

template<>
struct MESHFEM_EXPORT UnivariateQuadratureTable<4> {
    static constexpr std::array<double, 4> points{{0.33000947820757186760, 0.66999052179242813240, 0.93056815579702628761, 0.06943184420297371239}};
    static constexpr std::array<double, 4> weights{{0.326072577431273071, 0.326072577431273071, 0.1739274225687269287, 0.1739274225687269287}};
};

template<>
struct MESHFEM_EXPORT UnivariateQuadratureTable<5> {
    static constexpr std::array<double, 5> points{{0.5, 0.23076534494715845448, 0.76923465505284154552, 0.04691007703066800360, 0.95308992296933199640}};
    static constexpr std::array<double, 5> weights{{0.284444444444444444, 0.239314335249683234, 0.239314335249683234, 0.1184634425280945438, 0.1184634425280945438}};
};

template<size_t... Degrees>
struct GaussQuadratureImpl;

template<size_t Deg, size_t... Degrees>
struct GaussQuadratureImpl<Deg, Degrees...> {
    static constexpr size_t NumPts = Deg / 2 + 1;
    using UQT = UnivariateQuadratureTable<NumPts>;
    template<class F, class EvalPt>
    static void run(const F &f, double w, EvalPt &p) {
        size_t c = p.size() - (1 + sizeof...(Degrees));
        for (size_t i = 0; i < NumPts; ++i) {
            p[c] = UQT::points[i];
            GaussQuadratureImpl<Degrees...>::run(f, w * UQT::weights[i], p);
        }
    }
};

template<>
struct GaussQuadratureImpl<> {
    template<class F, class EvalPt>
    static void run(const F &f, double w, const EvalPt &p) {
        f(w, p);
    }
};

// Call f(w, p) for each Gaussian quadrature point p and associated weight
// for quadrature rule "Degrees..." on interval [0, 1]^N
template<size_t... Degrees, class F>
void unitHypercubeQuadrature(const F &f) {
    constexpr size_t N = sizeof...(Degrees);
    using FT    = function_traits<F>;
    using Real_ = typename FT::template arg<0>::type;
    VecN_T<Real_, N> p;
    GaussQuadratureImpl<Degrees...>::run(f, 1, p);
}

#endif /* end of include guard: TENSORPRODUCTQUADRATURE_HH */
