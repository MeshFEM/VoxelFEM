#include "TensorProductQuadrature.hh"

// We need to provide definitions for the static constexpr `points` and
// `weights` memebers to avoid undefined reference linker errors.
constexpr std::array<double, 1> UnivariateQuadratureTable<1>::points;
constexpr std::array<double, 1> UnivariateQuadratureTable<1>::weights;
constexpr std::array<double, 2> UnivariateQuadratureTable<2>::points;
constexpr std::array<double, 2> UnivariateQuadratureTable<2>::weights;
constexpr std::array<double, 3> UnivariateQuadratureTable<3>::points;
constexpr std::array<double, 3> UnivariateQuadratureTable<3>::weights;
constexpr std::array<double, 4> UnivariateQuadratureTable<4>::points;
constexpr std::array<double, 4> UnivariateQuadratureTable<4>::weights;
constexpr std::array<double, 5> UnivariateQuadratureTable<5>::points;
constexpr std::array<double, 5> UnivariateQuadratureTable<5>::weights;
