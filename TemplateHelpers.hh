////////////////////////////////////////////////////////////////////////////////
// TemplateHelpers.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Various convenience templates/metafunctions.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  02/06/2022 11:56:35
////////////////////////////////////////////////////////////////////////////////
#ifndef TEMPLATEHELPERS_HH
#define TEMPLATEHELPERS_HH
#include <cstdlib>
#include <utility>
#include <stdexcept>

namespace detail {
    template<template<size_t...> class T, size_t N, size_t Deg, size_t... Degrees>
    struct DegreeReplicatorImpl {
        using type = typename DegreeReplicatorImpl<T, N - 1, Deg, Degrees..., 1>::type;
    };
    template<template<size_t...> class T, size_t Deg, size_t... Degrees>
    struct DegreeReplicatorImpl<T, 1, Deg, Degrees...> {
        using type = T<Deg, Degrees...>;
    };
}

// Convert a template with parameters `size_t... Degrees` into one with
// a parameters `size_t N` (that gets expanded into `N` copies of `Deg`)
template<template<size_t...> class T, size_t Deg = 1>
struct DegreeReplicator {
    template<size_t N>
    using type = typename detail::DegreeReplicatorImpl<T, N, Deg>::type;
};

// Some operations can be implemented more easily and efficiently when the
// grid dimension is known as a template parameter:
//      template<size_t N>
//      struct Impl {
//          static void run(...) {
//              // Dimension-specific implementation here.
//          }
//      };
// this function chooses the appropriate instantiation of such an
// implementation template based on the runtime dimension.
template<template<size_t /* N */> class F, typename... Args>
void dispatchDimSpecific(size_t N, Args&&... args) {
    if      (N == 1) F<1>::run(std::forward<Args>(args)...);
    else if (N == 2) F<2>::run(std::forward<Args>(args)...);
    else if (N == 3) F<3>::run(std::forward<Args>(args)...);
    else throw std::runtime_error("Unsupported dimension");
}

#endif /* end of include guard: TEMPLATEHELPERS_HH */
