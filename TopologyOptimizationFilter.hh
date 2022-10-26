#ifndef MESHFEM_TOPOLOGYOPTIMIZATIONFILTER_HH
#define MESHFEM_TOPOLOGYOPTIMIZATIONFILTER_HH

#include "NDVector.hh"
#include "TensorProductSimulator.hh"

#include <Eigen/src/Core/util/Constants.h>
#include <functional>
#include "TemplateHelpers.hh"
#include <MeshFEM/Utilities/NameMangling.hh>

using EigenNDIndex = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;
template<typename _Sim> class TopologyOptimizationProblem;

template<typename Real_>
struct Filter {
    static std::string mangledName() { return "Filter" + floatingPointTypeSuffix<Real_>(); }
    virtual std::string virtualMangledName() const { return mangledName(); }

    virtual ~Filter() {};

    /// Forward propagation.
    /// @param[in] in original variables
    /// @param[out] out fitered variables
    virtual void apply(const NDVector<Real_> &in, NDVector<Real_> &out) = 0;

    /// Backward propagation.
    /// @param[in] in chain rule partial result
    /// @param[in] vars value of variables, could be needed to evaluate current derivatives
    /// @param[out] out derivatives accounting for current filter contribution
    virtual void backprop(const NDVector<Real_> &in, const NDVector<Real_> &vars, NDVector<Real_> &out) const = 0;

    const EigenNDIndex & inputDimensions() const { return  m_inputDims; }
    const EigenNDIndex &outputDimensions() const { return m_outputDims; }

    void setInputDimensions(const EigenNDIndex &dims) {
        m_setInputDimensions(dims);
        m_gridDimsAreSet = true;
        m_dimensionsUpdated();
    }

    void setOutputDimensions(const EigenNDIndex &dims) {
        m_setOutputDimensions(dims);
        m_gridDimsAreSet = true;
        m_dimensionsUpdated();
    }

    // Throw an error if filter is used or modified without specifying the grid dimensions
    void checkGridDimensionsAreSet() const {
        if(!m_gridDimsAreSet)
            throw std::runtime_error("Filter grid dimensions not set. "
                "Initialize a TopologyOptimizationProblem object with this filter before using it.");
    }

    void validateDimensions(const NDVector<Real_> &x_in, const NDVector<Real_> &x_out) const {
        checkGridDimensionsAreSet();
        const auto & in_size = x_in .sizes();
        const auto &out_size = x_out.sizes();
        if ((in_size.size() != size_t(m_inputDims.size())) || (out_size.size() != size_t(m_outputDims.size())))
            throw std::runtime_error("Dimension count mismatch");
        for (size_t d = 0; d < in_size.size(); ++d)
            if (in_size[d] != m_inputDims[d]) throw std::runtime_error("Input dimension mismatch along axis " + std::to_string(d) + " in " + virtualMangledName());
        for (size_t d = 0; d < out_size.size(); ++d)
            if (out_size[d] != m_outputDims[d]) throw std::runtime_error("Output dimension mismatch along axis " + std::to_string(d) + " in " + virtualMangledName());
    }

protected:
    // Number of elements in each dimension of the domain
    // Needed by all the filters to allow reshaping and testing in Python
    EigenNDIndex m_inputDims, m_outputDims;

    // Set the input and output grid dimensions based on the passed input/output grid dimension.
    // Most filters preserve the grid dimension, in which case these default implementations
    // suffice. Filters that modify the grid dimension must override these methods.
    virtual void m_setInputDimensions (const EigenNDIndex &dims) { m_inputDims = m_outputDims = dims; }
    virtual void m_setOutputDimensions(const EigenNDIndex &dims) { m_inputDims = m_outputDims = dims; }

    // Action for subclass optionally to take when the grid size has updated.
    virtual void m_dimensionsUpdated() { }

    // True if a TopologyOptimizationProblem has been used to fully initialize the filter
    bool m_gridDimsAreSet = false;
};

////////////////////////////////////////////////////////////////////////////////
// Collects a chain of filters mapping from design (input) variables to output
// physical densities (output). The intermediate variables at each step
// are cached for use in `backprop`.
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
struct FilterChain {
    using VXd = VecX_T<Real_>;
    using Filters = typename std::vector<std::shared_ptr<Filter<Real_>>>;

    static std::string mangledName() { return "FilterChain" + floatingPointTypeSuffix<Real_>(); }

    // Build a filter chain that will generate a density grid of resolution `outGridDims`.
    // Note: when dimension-modifying filters (like `UpsampleFilter`) are in use,
    // this will differ from the filter chain's input grid size.
    FilterChain(const Filters &f, const EigenNDIndex &outGridDims) : m_filters(f) {
        setOutputDimensions(outGridDims);
    }

    void setInputDimensions(EigenNDIndex dims) {
        EigenNDIndex inDims = dims;
        for (auto &f : m_filters) {
            f->setInputDimensions(dims);
            dims = f->outputDimensions();
        }

        m_vars.resize(m_filters.size() + 1);
        m_vars[0].resize(inDims);
        for (size_t fi = 0; fi < numFilters(); ++fi) {
            m_vars[fi + 1].resize(m_filters[fi]->outputDimensions());
        }
    }

    void setOutputDimensions(EigenNDIndex dims) {
        const size_t nf = numFilters();
        for (size_t i = 0; i < nf; ++i) {
            const auto &f = m_filters[(nf - 1) - i];
            f->setOutputDimensions(dims);
            dims = f->inputDimensions();
        }

        // `dims` now holds the input size of the first filter (i.e., the input
        // size of the entire chain).
        m_vars.resize(m_filters.size() + 1);
        m_vars[0].resize(dims);
        for (size_t fi = 0; fi < numFilters(); ++fi) {
            m_vars[fi + 1].resize(m_filters[fi]->outputDimensions());
        }
    }

    size_t numVars()    const { return m_vars.front().size();  }
    size_t numFilters() const { return m_filters.size();       }
    auto   gridDims()   const { return m_vars.front().sizes(); }

    size_t numPhysicalVars()  const { return m_vars.back().size(); }
    auto   physicalGridDims() const { return m_vars.back().sizes(); }

    void setDesignVars(Eigen::Ref<const VXd> x) {
        if (designVars().size() != size_t(x.size()))
            throw std::runtime_error("Variable size mismatch");

        designVars().flattened() = x;

        // Pass the design variables through the filter, caching
        // the intermediate values.
        for (size_t i = 0; i < numFilters(); i++)
            m_filters[i]->apply(m_vars[i], m_vars[i+1]);
    }

    void applyInPlace(NDVector<Real_> &x, NDVector<Real_> &xscratch) const {
        for (const auto &f : m_filters) {
            xscratch.resize(f->outputDimensions());
            f->apply(x, xscratch);
            x.swap(xscratch);
        }
    }

    void backprop(NDVector<Real_> &g, NDVector<Real_> &scratch) const {
        NDVector<Real_> &dJ_dout = g; // derivative with respect to a filter output
        size_t nf = numFilters();
        for (size_t i = nf - 1; i < nf; --i) {
            scratch.resize(m_filters[i]->inputDimensions());
            m_filters[i]->backprop(dJ_dout, m_vars[i], scratch);
            dJ_dout.swap(scratch);
        }
    }

    void backprop(NDVector<Real_> &g) const {
        NDVector<Real_> scratch(g.sizes());
        backprop(g, scratch);
    }

    const NDVector<Real_> &  designVars() const { return m_vars.front(); }
          NDVector<Real_> &  designVars()       { return m_vars.front(); }
    const NDVector<Real_> &physicalVars() const { return m_vars. back(); }
          NDVector<Real_> &physicalVars()       { return m_vars. back(); }

    const Filters &filters() const { return m_filters; }

private:
    Filters m_filters;
    std::vector<NDVector<Real_>> m_vars;
};

template<typename Real_>
struct ProjectionFilter : public Filter<Real_> {
    static constexpr size_t SIMD_WIDTH = 8;

    ProjectionFilter() { }
    ProjectionFilter(Real_ beta) : m_beta(beta) { }

    static std::string mangledName() { return "ProjectionFilter" + floatingPointTypeSuffix<Real_>(); }
    virtual std::string virtualMangledName() const override { return mangledName(); }

    void apply(const NDVector<Real_> &in, NDVector<Real_> &out) override {
        this->validateDimensions(in, out);
        BENCHMARK_SCOPED_TIMER_SECTION timer("applyProjectionFilter");
        Real_ tanh_half_beta = tanh(0.5 * m_beta);
        parallel_for_range(in.size() / SIMD_WIDTH, [&](size_t i) {
            out.flattened().template segment<SIMD_WIDTH>(SIMD_WIDTH * i) = (tanh_half_beta + (m_beta*(in.flattened().template segment<SIMD_WIDTH>(SIMD_WIDTH * i).array() - 0.5)).tanh()) / (2 * tanh_half_beta);
        });
        // Remainder...
        for (size_t i = SIMD_WIDTH * (in.size() / SIMD_WIDTH); i < in.size(); ++i) {
            out[i] = (tanh_half_beta + tanh(m_beta*(in[i] - 0.5))) / (2 * tanh_half_beta);
        }
    }

    void backprop(const NDVector<Real_> &in, const NDVector<Real_> &vars, NDVector<Real_> &out) const override {
        this->validateDimensions(out, in);
        BENCHMARK_SCOPED_TIMER_SECTION timer("backpropProjectionFilter");
        Real_ scale = 1.0 / (2 * tanh(0.5*m_beta) / m_beta);
        parallel_for_range(in.size() / SIMD_WIDTH, [&](size_t i_chunk) {
            size_t i = SIMD_WIDTH * i_chunk;
            out.flattened().template segment<SIMD_WIDTH>(i) = in.flattened().template segment<SIMD_WIDTH>(i).array() *
                (1.0 - (m_beta * (vars.flattened().template segment<SIMD_WIDTH>(i).array() - 0.5)).tanh().square()) * scale;
        });
        // Remainder...
        for (size_t i = SIMD_WIDTH * (in.size() / SIMD_WIDTH); i < in.size(); ++i) {
            out[i] = in[i] * (1.0 - std::pow(tanh(m_beta * (vars[i] - 0.5)), 2)) * scale;
        }
    }

    // Determine the input density scalar that maps to scalar `filteredValue`.
    Real_ invert(Real_ filteredValue) const {
        if ((filteredValue > 1.0) || (filteredValue < 0.0))
            throw std::runtime_error("ProjectionFilter::invert domain error: target density for inversion is outside [0, 1].");
        return atanh((2 * filteredValue - 1) * tanh(0.5 * m_beta)) / m_beta + 0.5;
    }

    Real_ getBeta() const { return m_beta; }
    void setBeta(Real_ beta) {
        if (beta <= 0)
            throw std::runtime_error("Beta parameter has to be positive (received beta = " + std::to_string(beta) + ")");
        m_beta = beta;
    }

private:
    // Beta defines the steepness of the Heaviside-like projection
    // (for beta->inf, projection is the step function)
    Real_ m_beta = 1.0;
};

template<typename Real_>
struct PythonFilter : public Filter<Real_> {
    PythonFilter() { }

    using VXd = VecX_T<Real_>;

    static std::string mangledName() { return "PythonFilter" + floatingPointTypeSuffix<Real_>(); }
    virtual std::string virtualMangledName() const override { return mangledName(); }

    using ApplyCallback    = std::function<void(Eigen::Ref<const VXd> in, Eigen::Ref<      VXd> out)>;
    using BackpropCallback = std::function<void(Eigen::Ref<const VXd> in, Eigen::Ref<const VXd> vars, Eigen::Ref<VXd> out)>;

    void apply(const NDVector<Real_> &in, NDVector<Real_> &out) override {
        if (!apply_cb) throw std::runtime_error("Apply callback must be configured");
        VXd result(out.flattened().size());
        apply_cb(in.flattened(), result);
        out.flattened() = result;
    }

    void backprop(const NDVector<Real_> &in, const NDVector<Real_> &vars, NDVector<Real_> &out) const override {
        if (!backprop_cb) throw std::runtime_error("Backprop callback must be configured");
        VXd result(out.flattened().size());
        backprop_cb(in.flattened(), vars.flattened(), result);
        out.flattened() = result;
    }

    ApplyCallback    apply_cb;
    BackpropCallback backprop_cb;
};

template<typename Real_, size_t N>
struct SmoothingFilterImpl;

// Note: this smoothing filter uses reflection boundary conditions so that it
// produces the same result for optimizations with and without symmetry
// conditions.
// This has the added benefit of making the smoothing operator symmetric,
// meaning `apply` and `backprop` are the same operation!
template<typename Real_>
struct SmoothingFilter : public Filter<Real_> {
    static constexpr size_t SIMD_WIDTH = 32; // setting this large allows amortizing kernel evaluation and index bookkeeping across more entries.
    using SIMDVec = Eigen::Array<Real_, SIMD_WIDTH, 1>;

    enum class Type { Const, Linear };

    static std::string mangledName() { return "SmoothingFilter" + floatingPointTypeSuffix<Real_>(); }
    virtual std::string virtualMangledName() const override { return mangledName(); }

    SmoothingFilter(int r = 1, Type t = Type::Const) : radius(r), type(t) { }

    void apply(const NDVector<Real_> &in, NDVector<Real_> &out) override {
        this->validateDimensions(in, out);
        BENCHMARK_SCOPED_TIMER_SECTION timer("applySmoothingFilter");
        dispatchDimSpecific<ApplyImpl>(in.sizes().size(), *this, in, out);;
    }

    // Apply `S^T = S`
    // Note, the operator is only symmetric due to our reflection boundary
    // conditions (and the blur kernel symmetry).
    void backprop(const NDVector<Real_> &in, const NDVector<Real_> &/* vars */, NDVector<Real_> &out) const override {
        this->validateDimensions(out, in);
        BENCHMARK_SCOPED_TIMER_SECTION timer("backpropSmoothingFilter");
        dispatchDimSpecific<ApplyImpl>(in.sizes().size(), *this, in, out);;
    }

    struct KernelConst {
        template<typename NDIndexInt>
        static Real_ eval(const NDIndexInt &/* offset */, Real_ /* radiusPlusOne */) {
            return 1.0;
        }
    };

    struct KernelLinear {
        // Note: negative values of the kernel are discarded by ApplyImpl::run!
        // (Which implicitly does a clamp-to-zero.)
        template<typename NDIndexInt>
        static Real_ eval(const NDIndexInt &offset, Real_ radiusPlusOne) {
            return radiusPlusOne - std::sqrt(Real_((offset).matrix().squaredNorm()));
        }
    };

    template<size_t N>
    struct ApplyImpl {
        template<class Kernel, typename NDIndexInt>
        static void process_chunk(const NDVector<Real_> &in, NDVector<Real_> &out,
                                  NDIndexInt centerIndex, const NDIndexInt &sizes, const IndexRange<NDIndexInt> &neighborhood, int /* radius */, Real_ radiusPlusOne) {
            centerIndex[N - 1] *= SIMD_WIDTH;
            size_t ei = NDVector<Real_>::flatIndexConstexpr(centerIndex, sizes);
            const int chunkSize = std::min<int>(SIMD_WIDTH, sizes[N - 1] - centerIndex[N - 1]);
            SIMDVec contrib = SIMDVec::Zero();
            Real_ totalWeight = 0.0;

            auto reflectIndex = [](int i, int s) { // Reflect an out-of-grid index back into the grid (across min/max face):   -2, -1, 0, 1, 2 ==> 1, 0, 0, 1, 2
                while ((i < 0) || (i >= s)) { // for extremely narrow grids, we may reflect multiple times...
                    if (i >= s)  i = 2 * s - i - 1;
                    if (i <  0)  i =        -i - 1;
                }
                return i;
            };

            neighborhood.visit([&](const NDIndexInt &offset) {
                Real_ w = Kernel::eval(offset, radiusPlusOne);
                if (w <= 0) return;

                NDIndexInt neighbor = centerIndex + offset;
                // Reflect all but innermost (SIMD) index of the neighbor.
                for (size_t d = 0; d < N - 1; d++)
                    neighbor[d] = reflectIndex(neighbor[d], sizes[d]);

                if ((neighbor[N - 1] >= 0) && (neighbor[N - 1] + int(SIMD_WIDTH) <= sizes[N - 1])) {
                    // No reflection needed in the interior
                    int k = NDVector<Real_>::template flatIndexConstexpr(neighbor, sizes);
                    contrib += w * in.flattened().template segment<SIMD_WIDTH>(k).array();
                }
                else {
                    const size_t n = neighbor[N - 1];
                    for (int s = 0; s < chunkSize; ++s) {
                        neighbor[N - 1] = reflectIndex(n + s, sizes[N - 1]);
                        int k = NDVector<Real_>::template flatIndexConstexpr(neighbor, sizes);
                        contrib[s] += w * in[k];
                    }
                }
                totalWeight += w;
            });

            if (chunkSize == SIMD_WIDTH) {
                out.flattened().template segment<SIMD_WIDTH>(ei) = contrib.matrix() / totalWeight;
            }
            else {
                out.flattened().segment(ei, chunkSize) = contrib.head(chunkSize) / totalWeight;
            }
        }

        using SF = SmoothingFilter<Real_>;
        static void run(const SF &sf, const NDVector<Real_> &in, NDVector<Real_> &out) {
            using NDIndexInt = Eigen::Array<int, N, 1>; // indices can go negative in intermediate calculations!
            NDIndexInt sizes = sf.inputDimensions().template cast<int>();
            auto neighborhood = make_index_range<NDIndexInt>(NDIndexInt::Constant(-sf.radius),
                                                             NDIndexInt::Constant( sf.radius + 1));
            const Real_ radiusPlusOne = 1 + sf.radius; // Note, without the "+1" a linear filter with radius 1 does no smoothing.

            NDIndexInt chunks = sizes;
            chunks[N - 1] = (chunks[N - 1] + SIMD_WIDTH - 1) / SIMD_WIDTH; // ceil

            if       (sf.type == SF::Type::Linear) IndexRangeVisitor<N, /* Parallel =  */ true>::run([&](const NDIndexInt &chunkIndex) { process_chunk<typename SF::KernelLinear>(in, out, chunkIndex, sizes, neighborhood, sf.radius, radiusPlusOne); }, NDIndexInt::Zero().eval(), chunks);
            else if  (sf.type == SF::Type::Const ) IndexRangeVisitor<N, /* Parallel =  */ true>::run([&](const NDIndexInt &chunkIndex) { process_chunk<typename SF::KernelConst >(in, out, chunkIndex, sizes, neighborhood, sf.radius, radiusPlusOne); }, NDIndexInt::Zero().eval(), chunks);
            else throw std::runtime_error("Unexpected smoothing kernel type");
        }
    };

    // Filter radius in units of grid cells
    int radius = 1;
    Type type = Type::Const;
};

// Upscale an NDVector by a certain factor.
// For instance, if the input array is 2x3 and the upscaling factor is
// 2, we obtain the 3x5 grid:
//              
//  x x x  ==>  x o x o x
//  x x x       o o o o o
//              x o x o x
// where "x" indicates values that are preserved by the upsampling and "o"
// indicates interpolated values.
//
//  This operation expands an array of size s to size (s - 1) * factor + 1.
//  For a factor of 2, this means expanding an array of size
//  1 + 2^N to 1 + 2^{N + 1}.
//  Note: it therefore cannot be used to generate powers-of-two-sized
//  element density grids directly. Instead it can produce element-corner
//  values that are then interpolated to the cell centers.
template<typename Real_>
struct UpsampleFilter : public Filter<Real_> {
    UpsampleFilter(size_t factor = 2) : m_factor(factor) { }

    static std::string mangledName() { return "UpsampleFilter" + floatingPointTypeSuffix<Real_>(); }
    virtual std::string virtualMangledName() const override { return mangledName(); }

    void apply(const NDVector<Real_> &in, NDVector<Real_> &out) override {
        this->validateDimensions(in, out);
        dispatchDimSpecific<DegreeReplicator<ApplyImpl, 1>::template type>(in.sizes().size(), m_factor, in, out);
    }

    void backprop(const NDVector<Real_> &d_dout, const NDVector<Real_> &/* vars */, NDVector<Real_> &d_din) const override {
        this->validateDimensions(d_din, d_dout);
        dispatchDimSpecific<DegreeReplicator<BackpropImpl, 1>::template type>(d_dout.sizes().size(), m_factor, d_dout, d_din);
    }

private:
    size_t m_factor;

    using Filter<Real_>:: m_inputDims;
    using Filter<Real_>::m_outputDims;

    void m_setInputDimensions(const EigenNDIndex &dims) override {
        if ((dims.array() < 2).any()) throw std::runtime_error("Interpolation can only be applied to a 2^d grid or larger");
        m_inputDims = dims;
        m_outputDims = (m_inputDims.array() - 1) * m_factor + 1;
    }

    void m_setOutputDimensions(const EigenNDIndex &dims) override {
        if ((dims.array() < 2).any()) throw std::runtime_error("Interpolation can only be applied to a 2^d grid or larger");
        m_outputDims = dims;
        m_inputDims = (dims.array() - 1) / m_factor + 1;
        if (((m_inputDims.array() - 1) * m_factor + 1 != m_outputDims.array()).any())
            throw std::runtime_error("Output size is not divisible by factor");
    }

    // Upsample the coarse grid values using interpolation degrees `Degrees...`
    // along each dimension.
    template<size_t... Degrees>
    struct ApplyImpl {
        static constexpr size_t N = sizeof...(Degrees);
        using Interpolant = TensorProductPolynomialInterpolant<Real_, Real_, Degrees...>;
        using NDIndex = Eigen::Array<size_t, N, 1>;

        static void run(size_t factor, const NDVector<Real_> &in, NDVector<Real_> &out) {
            static_assert(std::max({Degrees...}) < 2, "FIXME: loop over coarse elements is incorrect for deg > 1");
            parallel_for_range(in.size(), [&](size_t i) {
                Interpolant e;
                // Construct interpolant for the coarse cell with "min" corner at coarse entry i.
                auto minCorner = in.template unflattenIndex<N>(i);
                bool valid = true; // Whether this entry actually forms the "min" corner of a cell.
                e.coeffs.visit([&](Real_ &val, const NDArrayIndex<N> &liND_c) {
                    NDIndex iND_c = minCorner + Eigen::Map<const NDIndex>(liND_c.idxs.data());
                    if (!in.NDIndexInBound(iND_c)) {
                        valid = false;
                        return;
                    }
                    val = in(iND_c);
                });
                if (!valid) return;

                // Sample the coarse interpolant at the fine nodes inside.
                minCorner *= factor; // "min" corner location in the fine grid
                for (const auto &liND_f : make_index_range<NDIndex>(NDIndex::Zero(), NDIndex::Constant(factor + 1))) {
                    out(minCorner + liND_f) = e((liND_f.template cast<Real_>() / factor).matrix().eval());
                }
            });
        }
    };

    // Upsample the coarse grid values using interpolation degrees `Degrees...`
    // along each dimension.
    template<size_t... Degrees>
    struct BackpropImpl {
        static constexpr size_t N = sizeof...(Degrees);
        using NDIndex = Eigen::Array<size_t, N, 1>;
        static void run(size_t factor, const NDVector<Real_> &d_dout, NDVector<Real_> &d_din) {

            // Loop over the coarse nodes
            parallel_for_range(d_din.size(), [&](size_t i) {
                NDIndex n_c = d_din.template unflattenIndex<N>(i);
                NDIndex n_f = factor * n_c; // fine node coinciding with coarse node
                // Determine the range of fine nodes inside the coarse shape function's support.
                // We currently assume the degree 1 case, where the coarse node is at an element corner and
                // its support region extends to the neighboring corners' coinciding fine nodes (non-inclusive).
                static_assert(std::max({Degrees...}) < 2, "FIXME: handling of interior nodes in the high-degree case is incorrect...");
                // Start with a support region containing only `n_f`
                NDIndex fineBegin = n_f,
                        fineEnd   = n_f + 1;
                for (size_t d = 0; d < N; ++d) {
                    // Expand the support region to include fine nodes of the incident elements.
                    if (n_c[d] > 0)                    fineBegin[d] -= (factor - 1); // include nodes in "left"  coarse element
                    if (n_c[d] < d_din.sizes()[d] - 1) fineEnd  [d] += (factor - 1); // include nodes in "right" coarse element
                }
                d_din[i] = 0.0;
                for (const auto &n_v : make_index_range<NDIndex>(fineBegin, fineEnd)) {
                    static_assert(std::max({Degrees...}) < 2, "FIXME: Account for different types of shape functions in Deg > 1 case.");
                    auto evalPt = ((n_v.cwiseMax(n_f) - n_v.cwiseMin(n_f)).template cast<Real_>() / factor).matrix().eval();
                    Real_ phi = TensorProductBasisPolynomial<Real_, Degrees...>::template eval<(0 * Degrees)...>(evalPt);
                    d_din[i] += phi * d_dout(n_v);
                }
            });
        }
    };
};

// Convert a piecewise (tensor product) linear vertex-valued field defined by N
// + 1 values to a piecewise constant cell-valued field defined by N values
// by sampling the interpolated field at the element center.
template<typename Real_>
struct VertexToCellFilter : public Filter<Real_> {
    VertexToCellFilter() { }

    static std::string mangledName() { return "VertexToCellFilter" + floatingPointTypeSuffix<Real_>(); }
    virtual std::string virtualMangledName() const override { return mangledName(); }

    void apply(const NDVector<Real_> &in, NDVector<Real_> &out) override {
        this->validateDimensions(in, out);
        dispatchDimSpecific<ApplyImpl>(in.sizes().size(), in, out);
    }

    void backprop(const NDVector<Real_> &d_dout, const NDVector<Real_> &/* vars */, NDVector<Real_> &d_din) const override {
        this->validateDimensions(d_din, d_dout);
        dispatchDimSpecific<BackpropImpl>(d_din.sizes().size(), d_dout, d_din);
    }

    template<size_t N>
    struct ApplyImpl {
        static void run(const NDVector<Real_> &in, NDVector<Real_> &out) {
            Real_ weight = std::pow(2.0, -int(N));
            parallel_for_range(out.size(), [&](size_t ei) {
                out[ei] = 0.0;
                // Loop over the corner vertices of this output element...
                auto minCorner = out.template unflattenIndex<N>(ei);
                auto end = (minCorner.array() + 2).eval();
                for (const auto &vtxCorner : make_index_range(minCorner, end)) {
                    out[ei] += in(vtxCorner);
                }
                out[ei] *= weight;
            });
        }
    };

    template<size_t N>
    struct BackpropImpl {
        static void run(const NDVector<Real_> &d_dout, NDVector<Real_> &d_din) {
            Real_ weight = std::pow(2.0, -int(N));
            // Compute the derivative with respect to each vertex value one at a time.
            parallel_for_range(d_din.size(), [&](size_t vi) {
                d_din[vi] = 0.0;
                // Loop over all the output elements depending on this vertex.
                // These are the elements with indices at offsets -1 and 0.
                Eigen::Array<int, N, 1> minCorner = d_din.template unflattenIndex<N>(vi).template cast<int>();
                auto end = (minCorner.array() + 1).eval();
                minCorner -= 1;
                for (const auto &e : make_index_range(minCorner, end)) {
                    if (d_dout.NDIndexInBound(e)) {
                        d_din[vi] += d_dout[d_dout.template flatIndex</* checked = */ false>(e)];
                    }
                }
                d_din[vi] *= weight;
            });
        }
    };

private:
    using Filter<Real_>:: m_inputDims;
    using Filter<Real_>::m_outputDims;

    void m_setInputDimensions(const EigenNDIndex &dims) override {
        if ((dims.array() < 2).any()) throw std::runtime_error("Input grid must be 2^d or larger.");
        m_inputDims  = dims;
        m_outputDims = dims.array() - 1;
    }

    void m_setOutputDimensions(const EigenNDIndex &dims) override {
        m_outputDims = dims;
        m_inputDims  = dims.array() + 1;
    }
};

// Note: build direction is always Y
template<typename Real_>
struct LangelaarFilter : public Filter<Real_> {
    static constexpr size_t BuildDirection = 1; // Warning: changing this requires changes to NDVector::VisitLayer...
    LangelaarFilter() { }

    static std::string mangledName() { return "LangelaarFilter" + floatingPointTypeSuffix<Real_>(); }
    virtual std::string virtualMangledName() const override { return mangledName(); }

    void apply(const NDVector<Real_> &in, NDVector<Real_> &out) override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("applyLangelaarFilter");
        this->validateDimensions(in, out);
        // Note: smax approximation overestimates max function and can lead to overshoot in Langelaar-filtered densities values, possibly > 1.
        //       The variable bounds in the optimizer will take care of keeping the density field in [0, 1]
        // Bottom layer (attached to build platform)
        visitLayer(0, [&](size_t i) { out[i] = in[i]; });
        for (size_t layer = 1; layer < m_inputDims[BuildDirection]; layer++) {
            visitLayer(layer, [&](size_t i) {
                m_cachedSmax[i] = smax(out, NDVector<Real_>::unflattenIndex(i, m_inputDims));
                out[i] = smin(in[i], m_cachedSmax[i]);
            });
        }
        m_cachedFiltered = out; // cache value of filtered variables for backpropagation
    }

    void backprop(const NDVector<Real_> &in, const NDVector<Real_> &vars, NDVector<Real_> &out) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("backpropLangelaarFilter");
        this->validateDimensions(out, in);
        NDVector<Real_> lambdas(m_inputDims);
        computeLagrangeMultipliers(in, vars, lambdas);
        visitLayer(0, [&](size_t i) { out[i] = lambdas[i]; });
        for (size_t layer = 1; layer < m_inputDims[BuildDirection]; layer++)
            visitLayer(layer, [&](size_t i) { out[i] = lambdas[i]*dsmin_dx1(vars[i], m_cachedSmax[i]); });
    }

private:
    using Filter<Real_>:: m_inputDims;
    using Filter<Real_>::m_outputDims;

    void m_dimensionsUpdated() override {
        m_cachedSmax.resize(m_inputDims);
        m_cachedFiltered.resize(m_inputDims);
    }

    void computeLagrangeMultipliers(const NDVector<Real_> &in, const NDVector<Real_> &vars, NDVector<Real_> &lambdas) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("multipliersLangelaarFilter");
        this->checkGridDimensionsAreSet();
        size_t N = m_inputDims.size();
        std::vector<int> idx(N);
        std::vector<size_t> varIdx(N), derIdx(N);
        for (int layer = int(m_inputDims[BuildDirection])-1; layer >= 0; layer--) {
            visitLayer(layer, [&](size_t i) { lambdas[i] = in[i]; });
            if (layer < int(m_inputDims[BuildDirection])-1) {           // for every layer but the upper one
                visitLayer(layer+1, [&](size_t i) {               // i is the index of the variable w.r.t. derivative is evaluated
                    std::vector<size_t> varIdx = NDVector<Real_>::unflattenIndex(i, m_inputDims);
                    visitSupportingRegion(varIdx, [&](size_t k) { // k is the index of the variable w.r.t. derivative is taken
                        std::vector<size_t> derIdx = NDVector<Real_>::unflattenIndex(k, m_inputDims);
                        lambdas[k] += lambdas[i]*sminDerivative(vars, varIdx, derIdx);
                    });
                });
            }
        }
    }

    // Visit a full layer and apply the callback to each of the voxels
    void visitLayer(size_t layerIndex, const std::function<void(size_t)> &callback) const {
        NDVector<Real_>::visitLayer(layerIndex, m_inputDims, callback);
    }

    // Visit the supporting region of an element and apply the callback to each of the supporting voxels
    void visitSupportingRegion(const std::vector<size_t> &variableIndices, const std::function<void(size_t)> &callback) const {
        NDVector<Real_>::visitSupportingRegion(variableIndices, m_inputDims, callback);
    }

    // Evaluate smax function using as input the densities in support of voxel at location defined by indices
    Real_ smax(const NDVector<Real_> &vars, const std::vector<size_t> &indices) const {
        Real_ sum = 0;
        visitSupportingRegion(indices, [&](size_t i) { sum += std::pow(vars[i], m_P); });
        return std::pow(sum, 1/m_Q);
    }

    // Evaluate smax derivative w.r.t. one of the directly supporting variables
    Real_ smaxDerivative(const NDVector<Real_> &vars, const std::vector<size_t> &indices, const std::vector<size_t> &derivativeIndices) const {
        Real_ sum = 0;
        visitSupportingRegion(indices, [&](size_t i) { sum += std::pow(vars[i], m_P); });
        return m_P*std::pow(vars(derivativeIndices), m_P-1)/m_Q * std::pow(sum, 1/m_Q-1);
    }

    // Evaluate smin derivative w.r.t. one of the directly supporting variables
    Real_ sminDerivative(const NDVector<Real_> &vars, const std::vector<size_t> &indices, const std::vector<size_t> &derivativeIndices) const {
        // Note: derivativeIndices identify a supporting voxel of the one identified by indices
        return dsmin_dx2(vars(indices), m_cachedSmax(indices)) * smaxDerivative(m_cachedFiltered, indices, derivativeIndices);
    }

    Real_ smin(Real_ x1, Real_ x2) const { return 0.5*(x1 + x2 - std::pow((x1-x2)*(x1-x2)+m_epsilon, 0.5) + std::pow(m_epsilon, 0.5)); }
    Real_ dsmin_dx1(Real_ x1, Real_ x2) const { return 0.5*(1 - (x1-x2)*std::pow((x1-x2)*(x1-x2)+m_epsilon, -0.5)); }
    Real_ dsmin_dx2(Real_ x1, Real_ x2) const { return 0.5*(1 + (x1-x2)*std::pow((x1-x2)*(x1-x2)+m_epsilon, -0.5)); }

    // Physical variables
    NDVector<Real_> m_cachedFiltered;

    // Result of smax in supporting regions
    NDVector<Real_> m_cachedSmax;

    // Coefficient used in the approximation of the min function
    Real_ m_epsilon = 1e-4;

    // Value defining the P-norm that approximates the max function
    Real_ m_P = 40;

    // Exponent used to correct the P-norm overestimation
    Real_ m_Q = 40 - 1.58;
};

#endif // MESHFEM_TOPOLOGYOPTIMIZATIONFILTER_HH
