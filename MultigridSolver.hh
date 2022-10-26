#ifndef MULTIGRIDSOLVER
#define MULTIGRIDSOLVER

#include "NDVector.hh"
#include "TensorProductPolynomialInterpolant.hh"
#include "TensorProductSimulator.hh"
#include <memory>
#include <MeshFEM/ParallelAssembly.hh>
#include "ParallelVectorOps.hh"
#include "VoxelFEMBenchmark.hh"

// Whether to use "block Gauss Seidel" as the smoothing operation rather than
// "point Gauss Seidel". In the block version a small NxN matrix inverse is
// performed to simultaneously relax all components rather than a
// forward/reverse sweep. This only influences operation performed at nodes
// without Dirichlet conditions; for nodes with partial Dirichlet conditions
// we always apply the "point" version to avoid the extra work of modifying
// the system to enforce Dirichlet conditions.
#define BLOCK_GAUSS_SEIDEL true
#define BLOCKK_ASSEMBLE_UPPERTRI false // seems slightly slower :(

/// A class for multigrid solver
template<typename Real_, size_t... Degrees>
class MultigridSolver {
public:
    constexpr static size_t N = sizeof...(Degrees);
    using TPS                 = TensorProductSimulator<Real_, Degrees...>;
    using Point               = VecN_T<Real_, N>;
    using VNd                 = Eigen::Matrix<Real_, N, 1>;
    using MNd                 = Eigen::Matrix<Real_, N, N>;
    using VField              = typename TPS::VField;
    using EigenNDIndex        = typename TPS::EigenNDIndex;
    using PerElementStiffness = typename TPS::PerElementStiffness;

    MultigridSolver(std::shared_ptr<TPS> fineSimulator, size_t numCoarseningLevels)
    {
        //if (numCoarseningLevels == 0) throw std::runtime_error("numCoarseningLevels must be > 0");
        EigenNDIndex numElemsPerDim = fineSimulator->NbElementsPerDimension();
        BBox<VectorND<N>> domain = fineSimulator->domain();

        m_x.reserve(numCoarseningLevels + 1);
        m_b.reserve(numCoarseningLevels + 1);
        m_r.reserve(numCoarseningLevels + 1);

        // Build multigrid hierarchy, l=0 corresponds to the finest level
        for (size_t l = 0; l <= numCoarseningLevels; ++l) {
            std::shared_ptr<TPS> tps = fineSimulator;
            if (l > 0) { // l = 1, 2, ... correspond to the progressively coarsened grid.
                // We assume for now that we can simply halve the grid resolution at each step.
                for (size_t d = 0; d < N; ++d) {
                    if (numElemsPerDim[d] % 2 == 1)
                        throw std::runtime_error("Grid size currently must be divisible by 2^numCoarseningLevels (nonuniform coarsening not yet implemented)");
                    numElemsPerDim[d] = numElemsPerDim[d] / 2;
                }
                tps = std::shared_ptr<TPS>(new TPS(domain, numElemsPerDim)); // make_shared doesn't call Eigen's aligned `new` overload :( :(
                tps->setETensor(fineSimulator->getETensor());

                // Coarsen the Dirichlet conditions.
                // Even if non-zero Dirichlet conditions are specified at the
                // fine level, they coarsen to zero-Dirichlet conditions at the
                // coarser levels (the coarsened levels solve for *corrections*
                // to an admissible solution).
                // If a fine Dirichlet node falls on the boundary face of a
                // coarse element, all coarse nodes on that boundary face are
                // assigned Dirichlet conditions. If a fine node falls in the
                // interior we would need to make **all** coarse nodes Dirichlet,
                // which seems undesirable--for now we throw an error in this case
                // (assuming Dirichlet conditions are applied only at the grid boundary).
                const auto &finer = *m_sims.back();
                auto &coarser = *tps;
                EigenNDIndex numElemNodes = coarser.NbNodesPerDimensionPerElement();
                typename TPS::BCBuilder coarsenedBC(coarser);
                for (size_t fni = 0; fni < finer.numNodes(); ++fni) {
                    if (!finer.hasDirichlet(fni)) continue;
                    auto e_and_p = coarser.getElementAndReferenceCoordinates(finer.nodePosition(fni));

                    // Get an Nd index representing the coarse element boundary vertex/edge/face
                    // on which `p` appears. The d^th entry of this index is -1 if `p` is not
                    // on the min/max face of the d^th dimension. Otherwise it is the d^th
                    // index entry of the index of the coinciding coarse element boundary node.
                    Eigen::Array<int, N, 1> p_on_boundary;
                    for (size_t d = 0; d < N; ++d) {
                        Real_ c = e_and_p.second[d];
                        p_on_boundary[d] = std::abs(c) < 1e-9 ? 0 : (std::abs(c - 1.0) < 1e-9 ? (numElemNodes[d] - 1) : -1);
                    }
#if 0
                    if ((p_on_boundary < 0).all()) throw std::runtime_error("Dirichlet constraints on internal nodes are not supported");
#endif
                    // Apply fine node's Dirichlet constraints to all coarse
                    // nodes on the same vertex/edge/face as `p` (necessary and
                    // sufficient to guarantee the coarsened field satisfies
                    // the Dirichlet conditions on the fine mesh).
                    coarser.visitLocalNodeNdIndices([&](const EigenNDIndex &lni) {
                        for (size_t d = 0; d < N; ++d) {
                            if (p_on_boundary[d] == -1) continue;
                            if (p_on_boundary[d] != int(lni[d])) return; // Local node `lni` is not on the same vertex/edge/face of the coarse element as `p`
                        }

                        size_t cni = coarser.elemNodeGlobalIndex(coarser.elementIndexForGridCell(e_and_p.first), lni);
                        coarsenedBC.setDirichlet(cni, VNd::Zero(), finer.dirichletComponents(fni));
                    });
                }
                coarsenedBC.apply(coarser);
            }
            m_x.emplace_back(VField::Zero(tps->numNodes(), int(N)));
            m_b.emplace_back(VField::Zero(tps->numNodes(), int(N)));
            m_r.emplace_back(VField::Zero(tps->numNodes(), int(N)));
            m_sims.push_back(tps);
        }

        // At the first (most expensive) level of coarsening, the contribution
        // of a fine element to the coarsened per-element stiffness matrix is a
        // scalar multiple of I_f^T K0 I_f. For all coarse elements, this is
        // the same for a given nested fine element `f`. Therefore, we can
        // cache these 2^N coarsened stiffness matrices.
        const auto &phis = getCompressedElementInterpolationOperator();
        for (size_t fi = 0; fi < numFineElemsPerCoarse; ++fi) {
            coarsenedFineK0s[fi].setZero();
            accumulateCoarsenedStiffnessMatrix(fi, getSimulator(0).fullDensityElementStiffnessMatrix(), phis, coarsenedFineK0s[fi]);
        }
    }

    void setSymmetricGaussSeidel(const bool symmetric) {
        m_symmetricGaussSeidel = symmetric;
    }

          TPS &getSimulator(const size_t l)       { return *m_sims.at(l); }
    const TPS &getSimulator(const size_t l) const { return *m_sims.at(l); }

    auto getCompressedNodeInterpolationOperator() const {
        // Pre-evaluate coarse shape functions on all fine node positions inside.
        // There are (Deg + 1) coarse nodes along each dimension. Packing 2
        // fine elements along one dimension of a coarse element, we obtain
        // 2 * (Deg + 1) - 1 = 2 * Deg + 1 fine nodes along each dim (one node
        // overlaps).
        // These nodes are at reference coordinates j/(2 Deg) for j in {0, ..., 2 * deg}.
        using Evaluate = TensorProductPolynomialEvaluator<Real_, Degrees...>;
        NDArray<std::array<Real_, TPS::numNodesPerElem>, N, (2 * Degrees + 1)...> phis; // phis(fine node)[coarse node]
        VNd invSpacing((2.0 * Degrees)...);

        // The single-element interpolation operator should be identical for all
        // adjacent (coarse, fine) grid pairs in the hierarchy; pick an arbitrary one.
        const auto &coarser = getSimulator(1);
        const auto &finer   = getSimulator(0);
        visitFineElementsInside(finer, /* e_c = */ EigenNDIndex::Zero(), [&](size_t /* fi */, size_t e_f) {
            finer.visitElementNodes(e_f, [&](size_t n_f) {
                VNd refCoords = (finer.nodePosition(n_f) - coarser.origin()).array() / coarser.stretchings().array();
                VNd intCoords = (refCoords.array() * invSpacing.array()).matrix();
                assert((intCoords.array().round() - intCoords.array()).matrix().norm() < 1e-10); // better be integer...
                Eigen::Array<size_t, N, 1> idx = intCoords.array().round().template cast<size_t>();
                NDArray<Real_, N, (Degrees + 1)...> coeffs = Evaluate::evaluate(refCoords); // evaluate all coarse shape functions at the point.
                for (size_t ln_c = 0; ln_c < TPS::numNodesPerElem; ++ln_c)
                    phis(idx)[ln_c] = coeffs.get1D(ln_c);
            });
        });

        return phis;
    }

    // Call visit(n_c, coeff) for each nonzero triplet (n_f, n_c, coeff) in row `n_f`
    // of the sparse interpolation operator matrix I.
    template<class Phis, class Visitor>
    static void visitInterpolationOperatorRow(const TPS &/* finer */, const TPS &coarser, const EigenNDIndex &fine_node, const Phis &phis, const Visitor &visit) {
        // Number of fine nodes "owned" by a given coarse element along each dimension.
        // This is one less than the number of fine nodes falling on the element since
        // adjacent elements overlap by one node.
        static constexpr std::array<size_t, N> fineNodesPerCoarseElement{(2 * Degrees)...};

        EigenNDIndex  e_c = (fine_node / fineNodesPerCoarseElement[0]).cwiseMin(coarser.NbElementsPerDimension());
        EigenNDIndex lfni = fine_node - e_c * fineNodesPerCoarseElement[0];

        const auto &phi = phis(lfni);
        // Visit all nodes of the coarse element containing fine_node.
        size_t ln_c = 0;
        coarser.visitElementNodes(e_c, [&](size_t n_c) { Real_ coeff = phi[ln_c++]; if (coeff != 0) visit(n_c, coeff); });
    }

    template<typename ValMatrix>
    void accum_interpolation(const TPS &finer, const TPS &coarser, const ValMatrix &values, ValMatrix &out) const {
        if (size_t(values.rows()) != coarser.numNodes()) throw std::runtime_error("Invalid input size");

        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("Interpolation");
        const auto &phis = getCompressedNodeInterpolationOperator();

        IndexRangeVisitor<N, /* Parallel = */ true>::run(
            [&](const EigenNDIndex &fineNode) {
                size_t n_f = finer.flatIndexForNodeConstexpr(fineNode);
                auto result = out.row(n_f);
                visitInterpolationOperatorRow(finer, coarser, fineNode, phis, [&](size_t n_c, Real_ coeff) {
                    result += coeff * values.row(n_c);
                });
            }, EigenNDIndex::Zero().eval(), finer.nondetachedNodesPerDim());
    }

    template<typename ValMatrix>
    void interpolation(const TPS &finer, const TPS &coarser, const ValMatrix &values, ValMatrix &out) const {
        if (size_t(values.rows()) != coarser.numNodes()) throw std::runtime_error("Invalid input size");

        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("Interpolation");
        const auto &phis = getCompressedNodeInterpolationOperator();

        IndexRangeVisitor<N, /* Parallel = */ true>::run(
            [&](const EigenNDIndex &fineNode) {
                size_t n_f = finer.flatIndexForNodeConstexpr(fineNode);
                Eigen::Matrix<typename ValMatrix::Scalar, 1, ValMatrix::ColsAtCompileTime> result;
                result.setZero(out.cols());
                visitInterpolationOperatorRow(finer, coarser, fineNode, phis, [&](size_t n_c, Real_ coeff) {
                    result += coeff * values.row(n_c);
                });
                out.row(n_f) = result;
            }, EigenNDIndex::Zero().eval(), finer.NbNodesPerDimension());
    }

    // Restrict values from fine grid to the coarse grid.
    // (Apply transposed interpolation operator I^T)
    template<typename ValMatrix>
    void restriction(const TPS &finer, const TPS &coarser, const ValMatrix &values, ValMatrix &result) const {
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("Restriction");
        if (size_t(values.rows()) != finer.numNodes()) throw std::runtime_error("Invalid input size");
#if 1
        using EigenNDIndexSigned = Eigen::Array<int, N, 1>;
        static constexpr std::array<size_t, N> elementStride = {{Degrees... }};

        result.resize(coarser.numNodes(), values.cols());

        EigenNDIndexSigned     nodeSizes = coarser.nondetachedNodesPerDim().template cast<int>();
        EigenNDIndexSigned fineNodeSizes =   finer.nondetachedNodesPerDim().template cast<int>(); // Residuals on detached nodes are ignored/considered zero
        IndexRangeVisitor<N, /* Parallel = */ true>::run([&](const EigenNDIndexSigned &coarseNode) {
                EigenNDIndex localNode;
                for (size_t d = 0; d < N; ++d)
                    localNode[d] = coarseNode[d] % (elementStride[d]);

                size_t n_c = coarser.flatIndexForNodeConstexpr(coarseNode.template cast<size_t>());

                // Accumulate contributions from the fine nodes in the coarse node basis function's support
                const auto &fns = m_fineNodesInSupport[TPS::ElementNodeIndexer::flatIndex(localNode)];
                EigenNDIndexSigned minCorner = fns.minCorner + 2 * coarseNode,
                                   endCorner = fns.endCorner + 2 * coarseNode;
                Eigen::Matrix<Real_, 1, ValMatrix::ColsAtCompileTime> val;
                val.setZero(result.cols());

                IndexRangeVisitor<N>::run([&](const EigenNDIndexSigned &i) {
                        Real_ coeff = fns.coeff(i - minCorner);
                        val += coeff * values.row(finer.flatIndexForNodeConstexpr(i.template cast<size_t>()));
                }, minCorner.cwiseMax(            0).eval(),
                   endCorner.cwiseMin(fineNodeSizes).eval());

                result.row(n_c) = val;
            }, EigenNDIndexSigned::Zero().eval(), nodeSizes);

        // Note: NaNs in the first layer of detached nodes will leak into the topmost
        // nondetached layer during `applyK` since 0 * NaN = NaN; we must zero them out.
        if (nodeSizes[TPS::BUILD_DIRECTION] != int(coarser.NbNodesPerDimension()[TPS::BUILD_DIRECTION])) {
            EigenNDIndex nbegin = EigenNDIndex::Zero(),
                         nend   = coarser.NbNodesPerDimension();
            nbegin[TPS::BUILD_DIRECTION] = nodeSizes[TPS::BUILD_DIRECTION];
            nend  [TPS::BUILD_DIRECTION] = nbegin[TPS::BUILD_DIRECTION] + 1;
            IndexRangeVisitor<N, /* Parallel = */ true>::run([&](const EigenNDIndex &coarseNode) {
                    size_t n_c = coarser.flatIndexForNodeConstexpr(coarseNode.template cast<size_t>());
                    result.row(n_c).setZero();
                }, nbegin, nend);
        }
#else
        const auto &phis = getCompressedNodeInterpolationOperator();

        auto accumulatePerNodeContrib  = [&](size_t n_f, ValMatrix &out) {
            visitInterpolationOperatorRow(finer, coarser, n_f, phis, [&](size_t n_c, Real_ coeff) {
                    out.row(n_c) += coeff * values.row(n_f);
            });
        };

        result.setZero(coarser.numNodes(), values.cols());
        assemble_parallel(accumulatePerNodeContrib, result, finer.numNodes());
#endif
    }

    struct NodeSmoothStencilFinest {
        static void apply(const MultigridSolver &/* mg */, const TPS &sim, const EigenNDIndex &globalNode, const size_t /* n */, VField &u, const VField &/* b */, VNd &bMinusSprime, MNd &M) {
            M.setZero();
            // build S and M
            const PerElementStiffness &K0 = sim.fullDensityElementStiffnessMatrix();
            sim.visitIncidentElements(globalNode, [&](size_t ek, const EigenNDIndex &/* e */, const size_t localIndex, const typename TPS::ENodesArray &enodes) {
                Real_ E = sim.elementYoungModulusScaleFactor(ek);
                Eigen::Matrix<Real_, PerElementStiffness::ColsAtCompileTime, 1> u_local;
                for (size_t m = 0; m < TPS::numNodesPerElem; ++m)
                    u_local.template segment<N>(N * m) = u.row(enodes[m]).transpose();
                // Leverage symmetry of K0 to access storage-contiguous columns instead of storage-discontiguous rows.
                bMinusSprime -= E * (K0.template middleCols<N>(localIndex * N).transpose() * u_local).eval();
                M += E * sim.fullDensityElementStiffnessMatrixDiag(localIndex);
            });
        }
    };

    struct NodeSmoothStencilSecondFinest {
        static void apply(const MultigridSolver &mg, const TPS &sim, const EigenNDIndex &globalNode, const size_t /* n */, VField &u, const VField &/* b */, VNd &bMinusSprime, MNd &M) {
            M.setZero();
            // build S and M
            sim.visitIncidentElements(globalNode, [&](size_t /* ek */, const EigenNDIndex &e, const size_t localIndex, const typename TPS::ENodesArray &enodes) {
                const auto &finest = mg.getSimulator(0);

                Eigen::Matrix<Real_, PerElementStiffness::ColsAtCompileTime, 1> u_local;
                for (size_t m = 0; m < TPS::numNodesPerElem; ++m)
                    u_local.template segment<N>(N * m) = u.row(enodes[m]).transpose();
#if 1 // This version where we first build columns of the coarsened stiffness matrix appears to perform better...
                Eigen::Matrix<Real_, PerElementStiffness::ColsAtCompileTime, N> Ke_cols;
                visitFineElementsInside(finest, e, [&](size_t fi, size_t e_f) {
                    if (fi == 0) Ke_cols  = finest.elementYoungModulusScaleFactor(e_f) * mg.coarsenedFineK0s[fi].template middleCols<N>(localIndex * N);
                    else         Ke_cols += finest.elementYoungModulusScaleFactor(e_f) * mg.coarsenedFineK0s[fi].template middleCols<N>(localIndex * N);
                });
                // Leverage symmetry of K0 to access storage-contiguous columns instead of storage-discontiguous rows.
                bMinusSprime -= Ke_cols.transpose() * u_local;
                M += Ke_cols.template middleRows<N>(localIndex * N);
#else
                visitFineElementsInside(finest, e, [&](size_t fi, size_t e_f) {
                    bMinusSprime -= finest.elementYoungModulusScaleFactor(e_f) * (mg.coarsenedFineK0s[fi].template middleCols<N>(localIndex * N).transpose() * u_local);
                    M += finest.elementYoungModulusScaleFactor(e_f) * mg.coarsenedFineK0s[fi].template  block<N, N>(localIndex * N, localIndex * N);
                });
#endif
            });
        }
    };

    struct NodeSmoothStencilBlockK {
        static void apply(const MultigridSolver &/* mg */, const TPS &sim, const EigenNDIndex &/* globalNode */, const size_t n, VField &u, const VField &/* b */, VNd &bMinusSprime, MNd &M) {
            const auto &K = sim.blockK();
            const size_t end = K.Ap[n + 1];
            // Loop over n^th row of `K` by looping over n^th col and transposing (exploiting symmetry)
            for (size_t ii = K.Ap[n]; ii < end; ++ii) {
                const size_t i = K.Ai[ii];
                bMinusSprime -= (u.row(i) * K.Ax[ii]).transpose(); // Apply transpose block `Ax[ii].transpose()`
                if (__builtin_expect(i == n, 0)) M = K.Ax[ii];     // Diagonal blocks are symmetric--no transpose needed.
            }
        }
    };

    // A less branchy, slightly faster version of the Gauss-Seidel smoothing operation proposed in
    // [Wu 2016: eqs (11)-(13)]. We define S' = S + M u_old so that we needn't exclude the diagonal block `M`
    // when accumulating `S'`.
    // Then u_new[i] = (b[i] - S[i] - (M_lower * u_old - M_upper * u_new)[i]) / M_ii
    //               = (b[i] - S'[i] + (M * u_old)[i] - (M_lower * u_old - M_upper * u_new)[i]) / M_ii
    //               = (b[i] - S'[i] - (M_upper * (u_new - u_old))[i] + M_ii u_old[i]) / M_ii
    //               = (b[i] - S'[i] - (M_upper * (u_new - u_old))[i]) / M_ii + u_old[i]
    //               = (b[i] - S'[i] - (M * (u_new - u_old))[i]) / M_ii + u_old[i]
    //              := (b[i] - S'[i] - (M * u_diff)[i]) / M_ii + u_old[i]
    // where the second-to-last equality holds because u_diff[j] := u_new[j] - u_old[j] = 0 for j in i..N-1
    // (i.e., M_lower * u_diff == 0).
    template<class NSS>
    void m_smoothNode(const TPS &sim, const EigenNDIndex &globalNode, VField &u, const VField &b, const bool forwardSweep) const {
        const size_t n = sim.flatIndexForNodeConstexpr(globalNode);
        if (sim.hasFullDirichlet(n)) return;

        VNd bMinusSprime = b.row(n); // Accumulates b - S'
        MNd M;

        NSS::apply(*this, sim, globalNode, n, u, b, bMinusSprime, M);

        // Compute displacement u
        if (__builtin_expect(sim.hasDirichlet(n) != 0, false)) {
            // Partial Dirichlet case... (The case where all components have Dirichlet constraints is already handled above.)
            VNd u_diff(VNd::Zero());
            const auto &dc = sim.dirichletComponents(n);
            if (forwardSweep) { for (size_t i =     0; i < N; ++i) u_diff[i] = (bMinusSprime[i] - M.row(i) * u_diff) * (Real_(!dc.has(i)) / M(i, i)); }
            else              { for (size_t i = N - 1; i < N; --i) u_diff[i] = (bMinusSprime[i] - M.row(i) * u_diff) * (Real_(!dc.has(i)) / M(i, i)); }
            u.row(n) += u_diff;
        }
        else {
#if BLOCK_GAUSS_SEIDEL
            // M.llt().solveInPlace(bMinusSprime); // This is slower...
            // u.row(n) += bMinusSprime;
            u.row(n) += M.inverse() * bMinusSprime;
#else
            VNd u_diff(VNd::Zero());
            if (forwardSweep) { for (size_t i = 0; i < N; ++i)     u_diff[i] = (bMinusSprime[i] - M.row(i) * u_diff) / M(i, i); }
            else {              for (size_t i = N - 1; i < N; --i) u_diff[i] = (bMinusSprime[i] - M.row(i) * u_diff) / M(i, i); }
            u.row(n) += u_diff;
#endif
        }
    }

    // smoothing at level l via Gauss-Seidel
    // Pre/postcondition: u has Dirichlet conditions applied
    void smoothing(const size_t l, VField &u, const VField &b, const bool forwardSweep = true) const {
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("Smoothing");

        auto &sim = getSimulator(l);
        const size_t nn = sim.numNodes();

        if ((size_t(u.rows()) != nn) || (size_t(b.rows()) != nn)) throw std::runtime_error("Invalid input size");

        if (l > 0) {
            for (size_t n_ = 0; n_ < nn; ++n_) {
                size_t n = (forwardSweep) ? n_ : nn - (n_ + 1);
                EigenNDIndex ni_Nd = sim.ndIndexForNode(n);
                if (sim.isNodeDetached(ni_Nd)) continue;
                m_smoothNode<NodeSmoothStencilBlockK>(sim, ni_Nd, u, b, forwardSweep);
            }
        }
        else {
            for (size_t n_ = 0; n_ < nn; ++n_) {
                size_t n = (forwardSweep) ? n_ : nn - (n_ + 1);
                EigenNDIndex ni_Nd = sim.ndIndexForNode(n);
                if (sim.isNodeDetached(ni_Nd)) continue;
                m_smoothNode<NodeSmoothStencilFinest>(sim, ni_Nd, u, b, forwardSweep);
            }
        }
    }

    template<class F>
    void visitNodesMulticolored(const size_t l, F &&visitor, bool forwardSweep = true, bool parallel = true, bool skipDetached = true) const {
        using ENI = typename TPS::ElementNodeIndexer;
        const auto &sim = getSimulator(l);
        const EigenNDIndex NbNodesPerDimensionPerElement = sim.NbNodesPerDimensionPerElement();
        const EigenNDIndex NbNodesPerDimension           = skipDetached ? sim.nondetachedNodesPerDim() : sim.NbNodesPerDimension();

        // Process one color (local node of the reference element) at a time
        for (size_t i = 0; i < ENI::size(); ++i) {
            size_t lni = forwardSweep ? i : (ENI::size() - i - 1); // Process colors in reverse order for reverse sweep
            EigenNDIndex lni_nd = TPS::eigenNDIndexWrapper(ENI::unflattenIndex(lni));

            // For nodes on element boundary, we need to advance two elements
            // in that direction to reach a node of the same color (advancing
            // by 1 reaches a different color due to element overlaps).
            // For internal nodes, we advance by one element.
            EigenNDIndex nodeIncrements;
            for (size_t d = 0; d < N; ++d) {
                bool isBoundary = (lni_nd[d] == 0) || (lni_nd[d] == (NbNodesPerDimensionPerElement[d] - 1));
                // Adding (NbNodesPerDimensionPerElement[d] - 1) to the node index advances by one element in the d direction.
                nodeIncrements[d] = (1 + isBoundary) * (NbNodesPerDimensionPerElement[d] - 1);
            }

            // Solve for largest numNodesOfColorPerDim such that:
            //      lni_nd + nodeIncrements * (numNodesOfColorPerDim - 1) <= NbNodesPerDimension - 1
            EigenNDIndex numNodesOfColorPerDim = (NbNodesPerDimension - 1 - lni_nd) / nodeIncrements + 1;

            // Visit nodes of the current color in an arbitrary order.
            auto processNode = [&](const EigenNDIndex &idxWithinColor) {
                visitor(lni_nd + (idxWithinColor * nodeIncrements));
            };
            if (parallel) IndexRangeVisitor<N, /* Parallel = */  true>::run(processNode, EigenNDIndex::Zero().eval(), numNodesOfColorPerDim);
            else          IndexRangeVisitor<N, /* Parallel = */ false>::run(processNode, EigenNDIndex::Zero().eval(), numNodesOfColorPerDim);
        }
    }

    Eigen::VectorXi debugMulticolorVisit() const {
        const auto &sim = getSimulator(0);
        Eigen::VectorXi result(sim.numNodes());
        size_t i = 0;
        visitNodesMulticolored(0, [&](const EigenNDIndex &globalNode) { result[sim.flatIndexForNodeConstexpr(globalNode)] = i++; }, /* forwardSweep */ true, /* parallel */ false);
        return result;
    }

    void smoothingMulticoloredGS(const size_t l, VField &u, const VField &b, const bool forwardSweep = true) const {
        const auto &sim = getSimulator(l);
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("smoothingMulticoloredGS " + sim.description());
        if      (l == 0) visitNodesMulticolored(l, [&](const EigenNDIndex &globalNode) { m_smoothNode<NodeSmoothStencilFinest      >(sim, globalNode, u, b, forwardSweep); }, forwardSweep, /* parallel */ true);
        else if (l == 1) visitNodesMulticolored(l, [&](const EigenNDIndex &globalNode) { m_smoothNode<NodeSmoothStencilSecondFinest>(sim, globalNode, u, b, forwardSweep); }, forwardSweep, /* parallel */ true);
        else             visitNodesMulticolored(l, [&](const EigenNDIndex &globalNode) { m_smoothNode<NodeSmoothStencilBlockK      >(sim, globalNode, u, b, forwardSweep); }, forwardSweep, /* parallel */ true);
    }

    // Compute K * u for the finest grid (ignoring Dirichlet conditions)
    VField applyK(const VField &u) { return applyK(0, u); }

    // Compute K * u for the l^th grid (ignoring Dirichlet conditions)
    VField applyK(const size_t l, const VField &u) {
        VField result;
        applyK(l, u, result);
        return result;
    }

    // Compute K * u for the l^th grid (ignoring Dirichlet conditions)
    template<bool ZeroInit = true, bool Negate = false>
    void applyK(const size_t l, const VField &u, VField &result) {
        const auto &sim = *m_sims.at(l);
        if (l == 0) { return sim.template applyK<ZeroInit, Negate>(u, result); }
        if (l == 1) {
            BENCHMARK_SCOPED_TIMER_SECTION timer("applyK firstLevel");
            if (ZeroInit) sim.maskedNodalSetZeroParallel(result);
            const auto &finest = *m_sims.at(0);
            sim.visitElementsMulticolored([&](const EigenNDIndex &eiND) {
                    auto enodes = sim.elementNodes(eiND);

                    Eigen::Matrix<Real_, PerElementStiffness::ColsAtCompileTime, 1> u_local;
                    for (size_t m = 0; m < TPS::numNodesPerElem; ++m)
                        u_local.template segment<N>(N * m) = u.row(enodes[m]).transpose();

                    Eigen::Matrix<Real_, PerElementStiffness::ColsAtCompileTime, 1> Ke_u_local; // = (Ke * u_local).eval();
                    visitFineElementsInside(finest, eiND, [&](size_t fi, size_t e_f) {
                        if (fi == 0) Ke_u_local  = finest.elementYoungModulusScaleFactor(e_f) * (coarsenedFineK0s[fi] * u_local);
                        else         Ke_u_local += finest.elementYoungModulusScaleFactor(e_f) * (coarsenedFineK0s[fi] * u_local);
                    });

                    // Loop over nodal matvec contributions
                    for (size_t m = 0; m < TPS::numNodesPerElem; ++m) {
                        if (Negate) result.row(enodes[m]) -= Ke_u_local.template segment<N>(N * m).transpose();
                        else        result.row(enodes[m]) += Ke_u_local.template segment<N>(N * m).transpose();
                    }
                }, /* parallel = */ true, /* skipMasked = */ true);
            return;
        }

        if (!sim.hasCachedElementStiffness() && !sim.hasBlockK()) updateStiffnessMatrices();
        if (sim.hasBlockK()) sim.template applyBlockK<ZeroInit, Negate>(u, result);
        else                 sim.template applyK     <ZeroInit, Negate>(u, result);
    }

    // Zero out the variables in `u` with pin constraints for the finest grid
    void zeroOutDirichletComponents(VField &u) const { zeroOutDirichletComponents(0, u); }

    // Zero out the variables in `u` with pin constraints for the l^th grid
    void zeroOutDirichletComponents(const size_t l, VField &u) const {
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("zeroOutDirichletComponents");
        m_sims[l]->zeroOutDirichletComponents(u);
    }

    // Modify `u` so that it satisfies the Dirichlet conditions for the finest grid
    void enforceDirichletConditions(VField &u) const { enforceDirichletConditions(0, u, false); }

    // Modify `u` so that it satisfies the Dirichlet conditions for the l^th grid.
    // The imposed values are either the ones stored in the simulator's Dirichlet condition
    // data structure (if `zero` is false), or zeros (if `zero` is true).
    void enforceDirichletConditions(const size_t l, VField &u, bool zero) const {
        if (zero) { zeroOutDirichletComponents(l, u); }
        else      { m_sims[l]->enforceDirichletConditions(u); }
    }

    // Compute the residual at level l from displacement u and force b
    void computeResidual(const size_t l, const VField &u, const VField &b, VField &result) {
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("Compute residual");

        // Note: NaNs in the first layer of detached nodes will leak into the topmost
        // nondetached layer during `applyK` since 0 * NaN = NaN; we therefore must copy one extra
        // layer from `b` (which should be zero since no loads are applied to detached nodes)
        getSimulator(l).maskedNodalCopyParallel(b, result, /* margin = */ VOXELFEM_SIMD_WIDTH);
        applyK</* ZeroInit = */ false, /* Negate = */ true>(l, u, result);

        // Apply Dirichlet conditions:
        // If there is a Dirichlet constraint on this displacement component
        // the residual should be zero (a single smoothing iteration will
        // enforce the Dirichlet condition exactly).
        zeroOutDirichletComponents(l, result);
    }

    // Do a complete V-Cycle to solve the multigrid system
    // if `zeroDirichlet` is true, we replace the Dirichlet constraint values with zero.
    // This is needed for solving residual systems.
    template<class Derived>
    const VField &solve(const Eigen::MatrixBase<Derived> &u, const VField &f, size_t numSteps, size_t numSmoothingSteps, bool stiffnessUpdated = false,
                        bool zeroDirichlet = false, std::function<void(size_t, const VField &)> it_callback = nullptr, bool fmg = false) {
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("MG Solver");
        if (!stiffnessUpdated)
            updateStiffnessMatrices();

        m_x[0] = u;
        if (numSteps == 0) return m_x[0];

        m_b[0] = f;

        if (fmg) {
            fullMultigrid(0, numSmoothingSteps, zeroDirichlet);
            if (it_callback) it_callback(0, m_x[0]);
            for (size_t i = 1; i < numSteps; ++i) {
                vcycle(0, numSmoothingSteps, zeroDirichlet);
                if (it_callback) it_callback(i, m_x[0]);
            }
        }
        else {
            for (size_t i = 0; i < numSteps; ++i) {
                vcycle(0, numSmoothingSteps, zeroDirichlet);
                if (it_callback) it_callback(i, m_x[0]);
            }
        }
        return m_x[0];
    }

    // Approximately solve the system K u = r.
    // We use an initial guess that is more appropriate for the preconditioner application.
    const VField &applyPreconditionerInv(const VField &r, size_t numSteps, size_t numSmoothingSteps, bool fmg = false) {
        if (numSmoothingSteps == 0) return r;
        return solve(VField::Zero(r.rows(), r.cols()), r, numSteps, numSmoothingSteps, /* stiffnessUpdated= */ true, /* zeroDirichlet= */ true, /* cb = */ nullptr, fmg);
    }

    // Run a full multigrid cycle starting at level `l` of the hierarchy using
    // `numSmoothingSteps` Gauss-Seidel iterations to solve the equation:
    //      A_l m_x[l] = m_b[l]
    // If `residualSystem` is true, we are solving a residual equation and therefore
    // the initial guess/Dirichlet condition values should be set to zero.
    void fullMultigrid(size_t l, size_t numSmoothingSteps, bool residualSystem) {
        const size_t coarsestLevel = m_sims.size() - 1;
        if (l == coarsestLevel) {
            // TODO: support nonzero Dirichlet values in `residualSystem = false` case.
            m_x[l] = getSimulator(l).solve(m_b[l]);
            return;
        }

        const auto &sim        = getSimulator(l);
        const auto &sim_coarse = getSimulator(l + 1);

        // Restrict RHS to coarser level
        restriction(sim, sim_coarse, m_b[l], m_b[l + 1]);

        // Solve on the coarser grid.
        fullMultigrid(l + 1, numSmoothingSteps, residualSystem);

        // Interpolate the coarse grid's solution to this level.
        interpolation(sim, sim_coarse, m_x[l + 1], m_x[l]);

        // Run a V-cycle to reduce the error at this level.
        vcycle(l, numSmoothingSteps, residualSystem);
    }

    // Run a V-cycle iteration starting at level `l` of the hierarchy using
    // `numSmoothingSteps` Gauss-Seidel iterations to solve the equation:
    //      A_l m_x[l] = m_b[l]
    // **starting from the initial guess already stored in m_x[l]**.
    // If `residualSystem` is true, we are solving a residual equation and
    // therefore the Dirichlet condition values should be set to zero.
    void vcycle(size_t l, size_t numSmoothingSteps, bool residualSystem) {
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("V Cycle " + m_sims[l]->description());
        const size_t coarsestLevel = m_sims.size() - 1;

        // Direct solve at the coarsest level.
        if (l == coarsestLevel) {
            m_x[l] = getSimulator(l).solve(m_b[l]);
            return;
        }

        const auto &sim        = getSimulator(l);
        const auto &sim_coarse = getSimulator(l + 1);

        enforceDirichletConditions(l, m_x[l], residualSystem);

        // Run Gauss-Seidel iterations at this level.
        for (size_t i = 0; i < numSmoothingSteps; ++i)
            smoothingMulticoloredGS(l, m_x[l], m_b[l], /* forwardSweep */ true);

        // Restrict the smoothed residual to the next coarser level to form the coarse RHS.
        computeResidual(l, m_x[l], m_b[l], m_r[l]);

        restriction(sim, sim_coarse, m_r[l], m_b[l + 1]);

        // Continue v-cycle on coarser grid, computing correction m_x[l + 1]
        // Initial guess for correction is all zeros, and we need an extra layer of detached nodes to prevent NaNs from leaking in with smoothing/applyK
        // m_x[l + 1].setConstant(std::numeric_limits<double>::quiet_NaN()); // Uninitialized values debugging
        sim_coarse.maskedNodalSetZeroParallel(m_x[l + 1], /* margin = */ VOXELFEM_SIMD_WIDTH);
        vcycle(l + 1, numSmoothingSteps, /* residualSystem = */ true);

        // Interpolate and apply the coarse grid's correction to this level.
        accum_interpolation(sim, sim_coarse, m_x[l + 1], m_x[l]);

        // // Apply Dirichlet conditions (precondition for smoother, which is preserved by smoother).
        // // Note: this technically shouldn't be necessary as the interpolated coarse solution
        // // should have zeros on this grid's Dirichlet-constrained variables.
        // enforceDirichletConditions(l, m_x[l], residualSystem);

        // Re-run Gauss-Seidel iterations at this level.
        for (size_t i = 0; i < numSmoothingSteps; ++i)
            smoothingMulticoloredGS(l, m_x[l], m_b[l], /* forwardSweep */ !m_symmetricGaussSeidel);
    }

    // Build a "compressed element form" of the interpolation operator `I`:
    // evaluate the shape functions' value on the nodes of all fine elements it contains.
    static constexpr size_t numFineElemsPerCoarse = 1 << N; // There are 2^N fine elements within each coarse element.
    using Phi = Eigen::Matrix<Real_, TPS::numNodesPerElem, TPS::numNodesPerElem>;
    std::array<Phi, numFineElemsPerCoarse> getCompressedElementInterpolationOperator() const {
        // Assumes coarse and fine elements are the same (i.e., scaled versions of each other),
        // so the level doesn't matter.
        const auto &finer = getSimulator(0); // arbitrary
        std::array<Phi, numFineElemsPerCoarse> phis; // phis[fine_e](fine_n, coarse_n) holds:
                                            //   coarse shape function `coarse_n` evaluated on node `fine_n`
                                            //   of fine element `fine_e`
        Point origin = finer.nodePosition(finer.flattenedFirstNodeOfElement1D(0));
        for (size_t fine_n = 0; fine_n < TPS::numNodesPerElem; ++fine_n) {
            // "Half-canonical coordinates" of fine_n (in [0, 0.5]^N)
            const Point fineNodeHalfCoords = ((finer.nodePosition(finer.elemNodeGlobalIndex(0, fine_n)) - origin).array() / (2.0 * finer.getStretchings().array())).matrix();
            size_t fi = 0;
            HypercubeCornerVisitor<N>::run([&](EigenNDIndex fi_Nd) {
                // Get canonical coordinates (in [0, 1]^N) within the coarse element.
                Point fineNodePosition = fineNodeHalfCoords + (fi_Nd.template cast<Real_>()).matrix() * 0.5;

                // Evaluate all coarse shape functions on this fine node.
                using Evaluator = TensorProductPolynomialEvaluator<Real_, Degrees...>;
                auto coeffs = Evaluator::evaluate(fineNodePosition);
                phis[fi++].row(fine_n) = Eigen::Map<const Eigen::Matrix<Real_, coeffs.size(), 1>>(coeffs.data());
            });
        }
        return phis;
    }

    // There are up to 2^N fine elements contained in coarse element e_c.
    // These are found by doubling e_c's ND index and adding offsets in
    // {0, 1} to each index component.
    // Call `visitor(fi, e_f)` for each, where `fi` is a local linear index
    // in 0..2^N and `e_f` is the global element index in `finer.
    template<class Visitor>
    static void visitFineElementsInside(const TPS &finer, EigenNDIndex e_c_Nd, const Visitor &visitor) {
        e_c_Nd *= 2;
        size_t e_f_mincorner = finer.elementIndexForGridCellUnchecked(e_c_Nd);
        size_t fi = 0;
        HypercubeCornerVisitor<N>::run([&](const EigenNDIndex &i) {
            visitor(fi++, e_f_mincorner + finer.elementIndexForGridCellUnchecked(i));
        });
    }

    // Compute `Ke_c += I^T Ke_f I`,
    // where I[N * i + c, N * j + d] = phi(i, j) 𝛅_cd for c, d in range(N)
    // Note for any matrix `M` of same shape as Ke_f:
    //      [M I]_{ab} = sum_k M[a, k] I[k, b]
    //      [M I][:, b] = sum_k M[:, k] phi(k / N, b / N) 𝛅_{b % N, k % N}
    //      ==> Flatten(M I) = Flatten(M) * phi
    // Likewise,  I^T Ke_f = (Ke_f I)^T ==> Flatten((Ke_f I)^T) = Flatten(Ke_f) * phi
    template<class Phis>
    void accumulateCoarsenedStiffnessMatrix(size_t fi, const PerElementStiffness &Ke_f, const Phis &phis, PerElementStiffness &Ke_c) {
        PerElementStiffness It_Ke_f;

        using FlattenKe      = Eigen::Map<      Eigen::Matrix<Real_, TPS::KeSize * N, TPS::KeSize / N>>;
        using FlattenKeConst = Eigen::Map<const Eigen::Matrix<Real_, TPS::KeSize * N, TPS::KeSize / N>>;

        const Phi &phi = phis[fi];
        FlattenKe(It_Ke_f.data()) = FlattenKeConst(Ke_f.data()) * phi;
        It_Ke_f.transposeInPlace();
        FlattenKe(Ke_c.data()) += FlattenKeConst(It_Ke_f.data()) * phi;
    }

    void m_firstLevelCoarsenedStiffnessMatrix(const EigenNDIndex &e_c_Nd, PerElementStiffness &result) const {
        // Combine coarsened full-density stiffness matrices using the
        // fine element Young's moduli as weights.
        auto &finest = getSimulator(0);
        visitFineElementsInside(finest, e_c_Nd, [&](size_t fi, size_t e_f) {
            if (fi == 0) result  = finest.elementYoungModulusScaleFactor(e_f) * coarsenedFineK0s[fi];
            else         result += finest.elementYoungModulusScaleFactor(e_f) * coarsenedFineK0s[fi];
        });
    }

    void m_firstLevelCoarsenedStiffnessMatrix(const size_t e_c, PerElementStiffness &result) const {
        // Combine coarsened full-density stiffness matrices using the
        // fine element Young's moduli as weights.
        m_firstLevelCoarsenedStiffnessMatrix(getSimulator(1).ndIndexForElement(e_c), result);
    }

    // Get the coarsened stiffness matrix for element `e_c` of the simulation
    // mesh at level `l`.
    // SIDE EFFECT: also accumulate it to the appropriate block stiffness matrix
    // or store it in a per-element-stiffness-matrix cache as appropriate.
    template<class Phis>
    PerElementStiffness m_getCoarsenedStiffnessMatrix(const size_t l, const size_t e_c, const Phis &phis) {
        PerElementStiffness result;
        assert(l != 0);
        auto &coarser = getSimulator(l);
        EigenNDIndex eiND_c = coarser.ndIndexForElement(e_c);
        if (coarser.isElementMasked(eiND_c)) {
            result.setZero();
        }
        else {
            auto &finer = getSimulator(l - 1);
            if (l == 1) {
                // Combine coarsened full-density stiffness matrices using the
                // fine element Young's moduli as weights.
                m_firstLevelCoarsenedStiffnessMatrix(e_c, result);
            }
            else {
                // Recursively compute stiffness matrices for the next-finer
                // elements contained in e_c and coarsen them.
                result.setZero();
                visitFineElementsInside(finer, eiND_c, [&, e_c](size_t fi, size_t e_f) {
                    accumulateCoarsenedStiffnessMatrix(fi, m_getCoarsenedStiffnessMatrix(l - 1, e_f, phis), phis, result);
                });
            }
        }

        // Store the result by either caching it directly or accumulating it to
        // the appropriate block stiffness matrix.
        const size_t numLevels = m_sims.size();
        const bool accumToBlockK = l < numLevels - 1;

        if ((l == 1) && accumToBlockK) {
            // We don't store the coarsened stiffness matrix at the first coarsening level:
            // the associated computational stencils can be constructed on-the-fly, and
            // storing the coarsened blockK is a memory bottleneck.
            return result;
        }

        if (accumToBlockK) {
            auto &blockK = coarser.blockK();
            constexpr size_t nn = TPS::numNodesPerElem;
#if BLOCKK_ASSEMBLE_UPPERTRI
            // Build the **upper** block triangle; this will later be reflected
            // to produce the full blockK matrix.
            const typename TPS::ENodesArray enodes = coarser.elementNodes(e_c);
            for (size_t j = 0; j < nn; ++j) {
                SuiteSparse_long hint = std::numeric_limits<SuiteSparse_long>::max();
                for (size_t i = 0; i <= j; ++i)
                    hint = blockK.addNZ(enodes[i], enodes[j], result.template block<N, N>(N * i, N * j), hint);
            }
#else
            // Look up indices of the affected blocks in the global stiffness matrix.
            Eigen::Array<size_t, nn, nn> loc;
            const typename TPS::ENodesArray enodes = coarser.elementNodes(e_c);
            for (size_t j = 0; j < nn; ++j) {
                SuiteSparse_long hint = std::numeric_limits<SuiteSparse_long>::max();
                for (size_t i = 0; i < nn; ++i) {
                    loc(i, j) = blockK.findEntry(enodes[i], enodes[j], hint);
                    hint = loc(i, j) + 1;
                }
            }
            // Accumulate into the global stiffness matrix.
            for (size_t j = 0; j < nn; ++j) {
                blockK.Ax[loc(j, j)] += result.template block<N, N>(N * j, N * j);
                for (size_t i = j + 1; i < nn; ++i) {
                    blockK.Ax[loc(i, j)] += result.template block<N, N>(N * i, N * j);
                    blockK.Ax[loc(j, i)]  = blockK.Ax[loc(i, j)].transpose();
                }
            }
#endif
        }
        else {
            coarser.getStiffnessMatrixCacheVec()[e_c] = result;
        }
        return result;
    }

    // Functionality for efficient, surgical stiffness matrix updating during
    // layer-by-layer simulation; rather than rebuilding the entire coarsening
    // hierarchy from scratch, we recompute only the matrices that changed due
    // to layer masking since the last computation.
    struct LayerMaskingStiffnessUpdateState {
        size_t cachedStiffnessLayer; // The fabrication mask height for which the stiffness matrices have been computed.
        bool  enabled = true; // Whether banded coarsened stiffness matrix updates are enabled
        LayerMaskingStiffnessUpdateState() { reset(); }
        void reset(size_t layer = TPS::LAYER_MASK_NONE) {
            cachedStiffnessLayer = layer;
        }
        bool needsFullRecomputation(size_t currentLayer) const {
            return !enabled
                || (cachedStiffnessLayer == TPS::LAYER_MASK_NONE)
                || (currentLayer         == TPS::LAYER_MASK_NONE);
        }

        bool needsPartialRecomputation(size_t currentLayer) const {
            return enabled
                    && (cachedStiffnessLayer != TPS::LAYER_MASK_NONE)
                    && (currentLayer < cachedStiffnessLayer);
        }
    };
    LayerMaskingStiffnessUpdateState m_stiffnessUpdateState;

    void updateStiffnessMatrices() {
        // Build the coarsened stiffness matrices for the current fine densities.
        // An efficient block sparse matrix is built for coarse levels
        // 1..numLevels - 2 rather than storing an array of per-element
        // coarsened stiffness matrices.
        // However, we currently need to store per-element stiffness matrices
        // in the coarsest simulator (level `numLevels - 1`) so that it can
        // build a CHOLMOD-compatible matrix for factorization.
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("updateStiffnessMatrices");

        // Do we actually need to recompute the coarsened stiffness matrices from scratch?
        // Check if only a partial update or no update is needed.
        size_t currentLayer = getSimulator(0).firstMaskedElementLayerIdx();
        if (m_stiffnessUpdateState.needsPartialRecomputation(currentLayer)) {
            // std::cout << "Doing partial update " << currentLayer << ", " << m_stiffnessUpdateState.cachedStiffnessLayer << std::endl;
            return m_partialStiffnessMatrixUpdate();
        }
        if (!m_stiffnessUpdateState.needsFullRecomputation(currentLayer)) {
            std::cout << "WARNING: entirely skipping stiffness matrix update" << std::endl;
            return;
        }

        // std::cout << "Doing full update " << currentLayer << ", " << m_stiffnessUpdateState.cachedStiffnessLayer << std::endl;
        m_stiffnessUpdateState.reset(currentLayer); // We're doing a full recomputation now!

        const size_t numLevels = m_sims.size();
        {
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("Allocate");
        for (size_t l = 1; l < numLevels; ++l) {
            const bool useBlockK = l < numLevels - 1;
            if ((l == 1) && useBlockK) continue; // don't store blockK at the first coarsened level.
            auto &sim = getSimulator(l);
            if (useBlockK) {
                sim.allocateBlockK();
                sim.blockK().setZero();
                sim.getStiffnessMatrixCacheVec().clear();
            }
            else {
                sim.getStiffnessMatrixCacheVec().resize(sim.numElements());
            }
        }
        }

        const auto &phis = getCompressedElementInterpolationOperator();

        size_t coarsestLevel = numLevels - 1;
        auto &coarsest = getSimulator(coarsestLevel);
        coarsest.visitElementsMulticolored([&](const EigenNDIndex &e_c_ND) {
                m_getCoarsenedStiffnessMatrix(coarsestLevel,
                    coarsest.elementIndexForGridCell(e_c_ND), phis);
            }, /* parallel = */ true);

#if BLOCKK_ASSEMBLE_UPPERTRI
        for (size_t l = 2; l < numLevels - 1; ++l) {
            const bool useBlockK = l < numLevels - 1;
            getSimulator(l).blockK().reflectUpperTriangleInPlaceParallel();
        }
#endif
        coarsest.invalidateNumericFactorization();
    }

    void m_partialStiffnessMatrixUpdate() {
        size_t currentLayer = getSimulator(0).firstMaskedElementLayerIdx();
        if (!m_stiffnessUpdateState.needsPartialRecomputation(currentLayer)) throw std::logic_error("No partial update to make");

        const auto &phis = getCompressedElementInterpolationOperator();

        // Coarsen the layer ranges
        size_t coarsestLevel = m_sims.size() - 1;
        std::vector<std::pair<size_t, size_t>> coarsenedBands(coarsestLevel + 1);
        // The newly masked fine-level layers since the previous update are [currentLayer, cachedStiffnessLayer).
        coarsenedBands[0] = std::make_pair(currentLayer, m_stiffnessUpdateState.cachedStiffnessLayer);
        size_t coarsenedBegin = currentLayer, coarsenedEnd = m_stiffnessUpdateState.cachedStiffnessLayer;
        for (size_t i = 1; i <= coarsestLevel; ++i) {
            coarsenedBegin /= 2;
            coarsenedEnd = (coarsenedEnd + 1) / 2;
            coarsenedBands[i] = std::make_pair(coarsenedBegin, coarsenedEnd);
        }

        auto &coarsest = getSimulator(coarsestLevel);
        coarsest.visitElementsMulticolored([&](const EigenNDIndex &e_c_ND) {
                m_computeAndSubtractCoarsenedStiffnessMatrixBand(coarsestLevel,
                    coarsest.elementIndexForGridCell(e_c_ND), phis, coarsenedBands);
                // The following is for debugging of updates.
                // PerElementStiffness K0_update = coarsest.getStiffnessMatrixCacheVec()[coarsest.elementIndexForGridCell(e_c_ND)];
                // PerElementStiffness K0_recompute =
                //     m_getCoarsenedStiffnessMatrix(coarsestLevel, coarsest.elementIndexForGridCell(e_c_ND), phis);
                // std::cout << "K update error " << (K0_update - K0_recompute).norm() << std::endl;
            }, /* parallel = */ true);

        m_stiffnessUpdateState.cachedStiffnessLayer = currentLayer;
        coarsest.invalidateNumericFactorization();
    }

    // Compute the level `l` coarsened stiffness matrices for elements
    // influenced by the densities in the band of fine elements
    // [coarsenedBands[0].first, coarsenedBands[0].second) along the build
    // direction.
    // Side effect: these are subtracted from the cached blockK/per-element
    // stiffness matrices.
    template<class Phis>
    PerElementStiffness m_computeAndSubtractCoarsenedStiffnessMatrixBand(const size_t l, const size_t e_c, const Phis &phis, const std::vector<std::pair<size_t, size_t>> &coarsenedBands) {
        assert(l != 0);
        auto &coarser = getSimulator(l);
        EigenNDIndex eiND_c = coarser.ndIndexForElement(e_c);
        // Element does not overlap the band
        if ((eiND_c[TPS::BUILD_DIRECTION] <  coarsenedBands[l].first) ||
            (eiND_c[TPS::BUILD_DIRECTION] >= coarsenedBands[l].second)) return PerElementStiffness::Zero();

        auto &finer = getSimulator(l - 1);
        PerElementStiffness result;
        if (l == 1) {
            // Combine coarsened full-density stiffness matrices using the fine element Young's moduli as weights.
            auto &finest = getSimulator(0);
            visitFineElementsInside(finest, eiND_c, [&](size_t fi, size_t e_f) {
                Real scaleFactor = 0.0;
                size_t fine_layer = finest.ndIndexForElement(e_f)[TPS::BUILD_DIRECTION];
                if ((fine_layer >= coarsenedBands[0].first) &&
                    (fine_layer <  coarsenedBands[0].second))
                    scaleFactor = finest.unmaskedYoungModulusScaleFactor(e_f);
                // std::cout << "fi " << fi << " scale factor " << scaleFactor << std::endl;
                if (fi == 0) result  = scaleFactor * coarsenedFineK0s[fi];
                else         result += scaleFactor * coarsenedFineK0s[fi];
            });
        }
        else {
            // Recursively compute stiffness matrices for the next-finer elements contained in e_c and coarsen them.
            result.setZero();
            visitFineElementsInside(finer, eiND_c, [&, e_c](size_t fi, size_t e_f) {
                accumulateCoarsenedStiffnessMatrix(fi, m_computeAndSubtractCoarsenedStiffnessMatrixBand(l - 1, e_f, phis, coarsenedBands), phis, result);
            });
        }

        // Subtract the band's stiffness from the cached stiffness matrices.
        const size_t numLevels = m_sims.size();
        const bool accumToBlockK = l < numLevels - 1;

        if ((l == 1) && accumToBlockK) {
            // We don't store the coarsened stiffness matrix at the first coarsening level:
            // the associated computational stencils can be constructed on-the-fly, and
            // storing the coarsened blockK is a memory bottleneck.
            return result;
        }

        if (accumToBlockK) {
            auto &blockK = coarser.blockK();
            constexpr size_t nn = TPS::numNodesPerElem;
            // Look up indices of the affected blocks in the global stiffness matrix.
            Eigen::Array<size_t, nn, nn> loc;
            const typename TPS::ENodesArray enodes = coarser.elementNodes(e_c);
            for (size_t j = 0; j < nn; ++j) {
                SuiteSparse_long hint = std::numeric_limits<SuiteSparse_long>::max();
                for (size_t i = 0; i < nn; ++i) {
                    loc(i, j) = blockK.findEntry(enodes[i], enodes[j], hint);
                    hint = loc(i, j) + 1;
                }
            }
            // Subtract band's stiffness from the global stiffness matrix.
            for (size_t j = 0; j < nn; ++j) {
                blockK.Ax[loc(j, j)] -= result.template block<N, N>(N * j, N * j);
                for (size_t i = j + 1; i < nn; ++i) {
                    blockK.Ax[loc(i, j)] -= result.template block<N, N>(N * i, N * j);
                    blockK.Ax[loc(j, i)]  = blockK.Ax[loc(i, j)].transpose();
                }
            }
        }
        else {
            // Subtract band's stiffness from the per-element stiffness matrix.
            coarser.getStiffnessMatrixCacheVec()[e_c] -= result;
        }
        return result;
    }


    // Set fabrication mask height for all simulators based on a layer number
    // in the fine simulator.
    void setFabricationMaskHeightByLayer(size_t fineLayerIdx) {
        getSimulator(0).setFabricationMaskHeightByLayer(fineLayerIdx);
        Real_ h = getSimulator(0).getFabricationMaskHeight();
        for (size_t l = 1; l < m_sims.size(); ++l)
            getSimulator(l).setFabricationMaskHeight(h, false);
        m_stiffnessUpdateState.reset();
    }

    void decrementFabricationMaskHeightByLayer(size_t fineLayerIncrement) {
        auto &fineSim = getSimulator(0);
        fineSim.decrementFabricationMaskHeightByLayer(fineLayerIncrement);
        Real_ h = fineSim.getFabricationMaskHeight();
        for (size_t l = 1; l < m_sims.size(); ++l)
            getSimulator(l).setFabricationMaskHeight(h, false);
    }

    // Apply PCG ***in place*** to solve K x = b starting from initial guess `x`.
    // Uses our multigrid solver as a preconditioning operator `M^{-1}`.
    // If `fmg` is true, the fullMultigrid solver is used; otherwise plain V-cycles.
    // The stopping criterion is when the force residual becomes small relative to the applied forces:
    //      ||K x - b|| / ||b|| < tol.
    // This is different from the stopping criterion in [Shewchuk 94]; it is
    // strongly preferred for its physical meaning and because it **independent
    // of the initial guess u** and of the preconditioner employed.
    using PCGCallback = std::function<void(size_t, const VField &, const VField &)>;
    void preconditionedConjugateGradient(VField &x, const VField &b, const size_t maxIter,
                                           const Real_ tol, PCGCallback it_callback = nullptr,
                                           size_t mgIterations = 1,
                                           size_t mgSmoothingIterations = 1,
                                           bool fmg = false, bool dirichletAlreadySatisified = false) {
        const auto &fineSim = getSimulator(0);
        if (x.rows() != b.rows())                throw std::runtime_error("x and b should have the same size");
        if (x.rows() != int(fineSim.numNodes())) throw std::runtime_error("size of input and number of nodes don't correspond");

        // If number of coarsening levels is set to 0 and residual is still large, then run the direct solver
        if (m_sims.size() == 1) {
            VField &r = m_b[0]; // the residual is also the RHS of the system solved by the MG preconditioner.
            computeResidual(0, x, b, r);
            if (squaredNormParallel(r) < tol * tol * squaredNormParallel(b)) return;
            x = fineSim.solve(b);
            computeResidual(0, x, b, r);
            it_callback(1, x, r);
            return;
        }

        BENCHMARK_SCOPED_TIMER_SECTION timer("CG Iterations");
        VField &r = m_b[0]; // the residual is also the RHS of the system solved by the MG preconditioner.
        VField &s = m_x[0]; // `s` holds the result of the preconditioner applied to `r`,
                            // which is computed in m_x[0]

        Real b_norm_sq, rSquaredNorm,
              r_Minv_r = 0; // after first iteration, this will hold ||r||^2 in M^{-1} norm

        bool stiffnessMatricesUpdated = false; // Be lazy about updateStiffnessMatrices call...
        {
            BENCHMARK_SCOPED_TIMER_SECTION timer("Preamble");
            if (!dirichletAlreadySatisified) enforceDirichletConditions(x);
            b_norm_sq = fineSim.maskedNodalSquaredNorm(b);
            computeResidual(0, x, b, r);
            rSquaredNorm = fineSim.maskedNodalSquaredNorm(r);
            if (std::isnan(rSquaredNorm)) {
                throw std::logic_error("NaN encountered");
            }
            // std::cout << "rSquaredNorm: " << rSquaredNorm << ", b_norm_sq " << b_norm_sq << std::endl;
        }

        // We use a slightly restructured variant of the standard PCG algorithm
        // (e.g., [Shewchuk 94]) that eliminates some duplicate code and
        // crucially avoids a needless and costly application of the
        // preconditioner at the end of the final iteration.
        //
        // Loop invariant:
        //      r holds current residual (rSquaredNorm holds its squared norm)
        //      d holds previous search direction (empty on first iteration)
        //      After the first iteration, r_Minv_r holds the squared M-inverse norm of `r`, where M is the preconditioner.
        //  To speed up the common single-CG-iteration case, we defer the copy "d = s" in the first CG iteration to
        //  immediately before the M^{-1} application of the second CG iteration
        size_t i = 0;
        VField *d = nullptr; // no previous search direction
        while ((i++ < maxIter) && (rSquaredNorm > tol * tol * b_norm_sq)) {
            // Compute new search direction by making preconditioned residual conjugate to previous direction
            if (mgIterations > 0) {
                if (!stiffnessMatricesUpdated) {
                    updateStiffnessMatrices();
                    stiffnessMatricesUpdated = true;
                }
                if (d == &s) {
                    // Previous search direction was just the previous `s`
                    // vector; execute the deferred copy operation so the
                    // preconditioner application doesn't overwrite it.
                    d = &m_d;
                    fineSim.maskedNodalCopyParallel(s, *d, /* margin = */ VOXELFEM_SIMD_WIDTH);
                }
                applyPreconditionerInv(r, mgIterations, mgSmoothingIterations, fmg); // s = M^{-1} r
            }
            else
                s = r;

            {
            FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("CG direction");
            zeroOutDirichletComponents(s);
            Real_ r_Minv_r_old = r_Minv_r;
            r_Minv_r = fineSim.maskedNodalDotProduct(r, s);
            if (d) scaleAndAddInPlace(r_Minv_r / r_Minv_r_old, *d, s); // Make s conjugate to previous direction d (d = s + (r_Minv_r / r_Minv_r_old) * d)
            else   d = &s;                                             // Previous direction doesn't exist; d = s = M^{-1} r directly.
            }

            applyK(0, *d, Ad);
            zeroOutDirichletComponents(Ad);

            {
            FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("CG update");
            Real_ alpha = r_Minv_r / fineSim.maskedNodalDotProduct(*d, Ad);   // optimal step length
            // x += alpha * d;
            fineSim.maskedNodalVisitBlocks([&](size_t nodeStart, size_t numNodes) {
                    x.middleRows(nodeStart, numNodes) += alpha * d->middleRows(nodeStart, numNodes);
                });

            rSquaredNorm = fineSim.maskedNodalReduceSum([&](size_t nodeStart, size_t numNodes) {
                        r.middleRows(nodeStart, numNodes) -= alpha * Ad.middleRows(nodeStart, numNodes);
                        return r.middleRows(nodeStart, numNodes).squaredNorm();
                    });
            }
            if (std::isnan(rSquaredNorm)) throw std::logic_error("NaN encountered at iteration" + std::to_string(i));

            if (it_callback) {
                FINE_BENCHMARK_SCOPED_TIMER_SECTION cbtimer("Callback");
                it_callback(i, x, r);
            }
        }
    }

    // Get the residual from the most recent PCG solve (which is stored in the
    // multigrid system RHS)
    const VField &pcgResidual() const { return m_b[0]; }

    const VField &debug_get_x(int l) { return m_x.at(l); }
    const VField &debug_get_b(int l) { return m_b.at(l); }

    std::array<PerElementStiffness, numFineElemsPerCoarse> coarsenedFineK0s;

private:
    std::vector<std::shared_ptr<TPS>> m_sims;

    std::vector<VField> m_x, m_b, m_r;
    VField Ad, m_d; // CG workspace

    bool m_symmetricGaussSeidel = true;

    using Stencils = TPSStencils<Real_, Degrees...>;
    const typename Stencils::FineNodesInSupport m_fineNodesInSupport = Stencils::fineNodesInSupport();

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Work around alignment issues when using SIMD
};

// Metafunction to get multigrid solver for a particular TensorProductSimulator
// instantiation.
template<typename T>
struct MGSolverForTPS;

template<typename Real_, size_t... Degrees>
struct MGSolverForTPS<TensorProductSimulator<Real_, Degrees...>> {
    using type = MultigridSolver<Real_, Degrees...>;
};

template<typename T>
using MGSolverForTPS_t = typename MGSolverForTPS<T>::type;

#endif /* MULTIGRIDSOLVER */
