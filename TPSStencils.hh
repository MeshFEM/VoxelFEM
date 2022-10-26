////////////////////////////////////////////////////////////////////////////////
// TPSStencils.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Connectivity information for accessing nodes and elements in the support of
//  a node's basis function.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  02/20/2022 18:09:19
////////////////////////////////////////////////////////////////////////////////
#ifndef TPSSTENCILS_HH
#define TPSSTENCILS_HH
#include <MeshFEM/Utilities/NDArray.hh>
#include "TensorProductBasisPolynomial.hh"
#include "NDVector.hh"
#include "ParallelVectorOps.hh"

#ifndef VOXELFEM_SIMD_WIDTH
#define VOXELFEM_SIMD_WIDTH 4
#endif

template<typename Real_, size_t... Degrees>
struct TPSStencils  {
    static constexpr size_t N = sizeof...(Degrees);
    using EigenNDIndex       = Eigen::Array<size_t, N, 1>;
    using EigenNDIndexSigned = Eigen::Array<int, N, 1>;
    using ElementNodeIndexer = NDArrayIndexer<N, (    Degrees + 1)...>;

    template<class STLNDIndex>
    static auto eigenNDIndexWrapper(const STLNDIndex &idxs) -> decltype(Eigen::Map<const EigenNDIndex>(idxs.data())) {
        assert(size_t(idxs.size()) == N);
        return Eigen::Map<const EigenNDIndex>(idxs.data());
    }

    // Information about the elements incident a node:
    //   1) The offset from the node's "primary" element Nd index to this incident element.
    //      The primary element Nd index is defined as "globalNode / elementStride"
    //      and may actually be out-of-bounds (for nodes on the upper-right grid boundaries).
    //   2) The local index of the node within this element.
    // This connectivity information is the same for all nodes of the same "type,"
    // where the type is determined by the index in (0, Degrees)... within the
    // canonical reference element.
    using ENA = std::vector<std::pair<EigenNDIndex, size_t>,
                            Eigen::aligned_allocator<std::pair<EigenNDIndex, size_t>>>;
    using ElementsAdjacentNode = NDArray<ENA, N, Degrees...>;

    struct ENAComputer {
        template<size_t... I> // I: node "type"
        static void visit(ENA &result) {
            static constexpr std::array<size_t, N> elementStride = {{ Degrees... }};
            EigenNDIndex localNode = eigenNDIndexWrapper(std::array<size_t, N>{{ I... }});

            static constexpr size_t maxNeighbors = 1 << N;
            for (size_t i = 0; i < maxNeighbors; ++i) {
                EigenNDIndex e_idx = EigenNDIndex::Zero(), localNodeInNeighbor = localNode;
                for (size_t d = 0; d < N; ++d) {
                    if ((1 << d) & i) continue;
                    // Only nodes on the min element boundary have incident elements in the "-" direction.
                    if ((localNode[d] != 0)) goto invalid;
                    else {
                        --e_idx[d];
                        localNodeInNeighbor[d] = elementStride[d];
                    }
                }

                result.push_back(std::make_pair(e_idx, ElementNodeIndexer::flatIndex(localNodeInNeighbor)));
                invalid: ;
            }
        }
    };

    static ElementsAdjacentNode elementsAdjacentNode() {
        ElementsAdjacentNode result;
        result.visit_compile_time(ENAComputer());
        return result;
    }

    // Fine nodes in the support of a coarse node's basis function.
    // This information is the same for all nodes of the same "type," where the
    // type is determined by the index in (0, Degrees)... within the canonical
    // reference element.
    struct FNS {
        EigenNDIndexSigned minCorner;
        EigenNDIndexSigned endCorner; // max + 1
        NDVector<Real_> coeff;
    };
    using FineNodesInSupport = NDArray<FNS, N, Degrees...>;

    struct FNSComputer {
        template<size_t... I> // I: node "type"
        static void visit(FNS &result) {
            EigenNDIndexSigned localNode     = eigenNDIndexWrapper(std::array<size_t, N>{{ I... }}).template cast<int>();
            EigenNDIndexSigned elementStride = eigenNDIndexWrapper(std::array<size_t, N>({{(Degrees)... }})).template cast<int>();

            EigenNDIndexSigned &minCorner = result.minCorner;
            EigenNDIndexSigned &endCorner = result.endCorner;

            // Bounding box of the support region on the coarse grid.
            for (size_t d = 0; d < N; ++d) {
                minCorner[d] = (localNode[d] == 0) ? -elementStride[d] : 0;
                endCorner[d] = elementStride[d];
            }

            // Convert to offsets from coarse node, then to fine-grid offsets,
            // then finally inset by one (basis function vanishes on the bounding box)
            minCorner = 2 * (minCorner - localNode) + 1;
            endCorner = 2 * (endCorner - localNode);

            result.coeff.resize((endCorner - minCorner).template cast<size_t>().eval());

            // std::cout << "minCorner: " << minCorner.transpose() << std::endl;
            // std::cout << "endCorner: " << endCorner.transpose() << std::endl;

            IndexRangeVisitor<N>::run([&](const EigenNDIndexSigned &i) {
                result.coeff(i - result.minCorner) =
                    TensorProductBasisPolynomial<Real_, Degrees...>::template eval<I...>(
                        (i + localNode * 2).template cast<Real_>().abs() / (2 * elementStride.template cast<Real_>())
                    );
                // std::cout << "coeff[" << (i - result.minCorner).transpose() << "] = " << result.coeff(i - result.minCorner) << std::endl;
            }, minCorner, endCorner);
        }
    };

    static FineNodesInSupport fineNodesInSupport() {
        FineNodesInSupport result;
        result.visit_compile_time(FNSComputer());
        return result;
    }
};

template<typename Real_, size_t... Degrees>
struct TensorProductSimulator;

template<typename Real_, size_t... Degrees>
struct SpecializedTPSStencils {
    static constexpr size_t N = sizeof...(Degrees);
    using EigenNDIndex        = Eigen::Array<size_t, N, 1>;
    using EigenNDIndexSigned  = Eigen::Array<int, N, 1>;
    using ElementNodeIndexer  = NDArrayIndexer<N, (    Degrees + 1)...>;
    using Sim                 = TensorProductSimulator<Real_, Degrees...>;
    using VField              = typename Sim::VField;
    using PerElementStiffness = typename Sim::PerElementStiffness;
    using LocalDisplacements  = Eigen::Matrix<Real_, PerElementStiffness::RowsAtCompileTime, 1>;

    template<class F>
    static void visitIncidentElements(const Sim &sim, const EigenNDIndex &globalNode, F &&visitor) {
        static constexpr std::array<size_t, N> elementStride = {{Degrees... }};
        EigenNDIndex primaryE, localNode;
        for (size_t d = 0; d < N; ++d) {
            primaryE [d] = globalNode[d] / (elementStride[d]);
            localNode[d] = globalNode[d] % (elementStride[d]);
        }

        for (const auto &ean : sim.m_elementsAdjacentNode[ElementNodeIndexer::flatIndex(localNode)]) {
            EigenNDIndex e = primaryE + ean.first;
            if ((e >= sim.m_NbElementsPerDimension).any()) continue; // out of bounds check (Note -1 gets wrapped to `size_t` max).
            size_t ei = sim.elementIndexForGridCellUnchecked(e);
            auto enodes = (sim.m_referenceElementNodes + sim.flattenedFirstNodeOfElement(e)).eval();
            visitor(ei, e, ean.second, enodes);
        }
    }

    template<bool ZeroInit = true, bool Negate = false>
    static void applyK(const Sim &sim, Eigen::Ref<const VField> u, VField &result) {
        if (ZeroInit) setZeroParallel(result, sim.numNodes(), N);

        const PerElementStiffness &K0 = sim.fullDensityElementStiffnessMatrix();
        const auto &enodes = sim.referenceElementNodes();

        sim.visitElementsMulticolored([&](const EigenNDIndex &eiND) {
                const size_t ei           = (eiND * sim.m_ElementIndexIncrement).sum();
                const size_t enode_offset = sim.flattenedFirstNodeOfElement(eiND);

                LocalDisplacements Ke_u_local(K0.template middleCols<N>(0) * u.row(enodes[0] + enode_offset).transpose());
                // Loop over nodal displacements
                for (size_t m = 1; m < enodes.size(); ++m)
                    Ke_u_local += K0.template middleCols<N>(N * m) * u.row(enodes[m] + enode_offset).transpose();
                Ke_u_local *= sim.elementYoungModulusScaleFactor(ei);
                // Loop over nodal matvec contributions
                for (size_t m = 0; m < enodes.size(); ++m) {
                    if (Negate) result.row(enodes[m] + enode_offset) -= Ke_u_local.template segment<N>(N * m).transpose();
                    else        result.row(enodes[m] + enode_offset) += Ke_u_local.template segment<N>(N * m).transpose();
                }
            }, /* parallel = */ true);
    }
};

#if 1
template<typename Real_>
struct SpecializedTPSStencils<Real_, 1, 1> {
    using EigenNDIndex        = Eigen::Array<size_t, 2, 1>;
    using Sim                 = TensorProductSimulator<Real_, 1, 1>;
    using VField              = typename Sim::VField;
    using VNd                 = typename Sim::VNd;
    using PerElementStiffness = typename Sim::PerElementStiffness;
    using LocalDisplacements  = Eigen::Matrix<Real_, PerElementStiffness::RowsAtCompileTime, 1>;
    static constexpr size_t N = Sim::N;

    template<class F>
    static void visitIncidentElements(const Sim &sim, const EigenNDIndex &globalNode, F &&visitor) {
        auto notLowerBorder = (globalNode > 0).eval();
        auto notUpperBorder = (globalNode < sim.NbElementsPerDimension()).eval();

        EigenNDIndex e = globalNode; // primaryE = globalNode
        size_t ei = sim.elementIndexForGridCellUnchecked(e);
        size_t firstNode = sim.flattenedFirstNodeOfElement(e);

        auto enodes = (sim.referenceElementNodes() + firstNode).eval();

        bool present[4] = {
                notUpperBorder[0] && notUpperBorder[1],
                notUpperBorder[0] && notLowerBorder[1],
                notLowerBorder[0] && notLowerBorder[1],
                notLowerBorder[0] && notUpperBorder[1] };

        // ( 0,   0), local node: 0
        if (present[0]) visitor(ei, e, 0, enodes);
        --e[1]; --ei; enodes -= 1;
        // ( 0,  -1), local node: 1
        if (present[1]) visitor(ei, e, 1, enodes);
        --e[0]; ei -= sim.m_ElementIndexIncrement[0]; enodes -= sim.m_NodeGlobalIndexIncrementPerElementIncrement[0];
        // (-1,  -1), local node: 3
        if (present[2]) visitor(ei, e, 3, enodes);
        ++e[1]; ++ei; enodes += 1;
        // (-1,   0), local node: 2
        if (present[3]) visitor(ei, e, 2, enodes);
    }

    static constexpr size_t SIMD_WIDTH = VOXELFEM_SIMD_WIDTH;
    using SIMDVec = Eigen::Array<Real_, SIMD_WIDTH, 1>;
    template<bool ZeroInit = true, bool Negate = false>
    static void applyK(const Sim &sim, Eigen::Ref<const VField> u, VField &result) {
        const size_t nn = sim.numNodes();
        result.resize(sim.numNodes(), N);
        const PerElementStiffness &K0 = sim.fullDensityElementStiffnessMatrix();
        EigenNDIndex numNodesToVisit = sim.nondetachedNodesPerDim();
        IndexRangeVisitor<N, /* Parallel = */ true>::run([&](EigenNDIndex globalNode) {
            globalNode[N - 1] *= SIMD_WIDTH;
            auto &primaryE = globalNode;
            size_t ni = sim.flatIndexForNodeConstexpr(globalNode);
            const auto &einc = sim.   m_ElementIndexIncrement;
            const auto &ninc = sim.m_NodeGlobalIndexIncrement;
            size_t ei = sim.elementIndexForGridCellUnchecked(primaryE);
            const bool interior = ((primaryE - 1) < sim.NbElementsPerDimension()).all() // wraps around!
                                && (primaryE[0]              < sim.NbElementsPerDimension()[0])
                                && (primaryE[1] + SIMD_WIDTH < sim.NbElementsPerDimension()[1]);

            SIMDVec u_n[8][N], u_i[N], f[N];

            // Accumulate an element's contribution to f using its corner displacements.
            auto accum = [&K0, &f](SIMDVec E, size_t row_offset,
                                   SIMDVec u0[N], SIMDVec u1[N], SIMDVec u2[N], SIMDVec u3[N]) {
                for (size_t c = 0; c < N; ++c) {
                    SIMDVec contrib;
                    // Leverage symmetry of K0 to scan down its columns.
                    contrib  = K0(0, row_offset) * u0[0]; contrib += K0(1, row_offset) * u0[1];
                    contrib += K0(2, row_offset) * u1[0]; contrib += K0(3, row_offset) * u1[1];
                    contrib += K0(4, row_offset) * u2[0]; contrib += K0(5, row_offset) * u2[1];
                    contrib += K0(6, row_offset) * u3[0]; contrib += K0(7, row_offset) * u3[1];
                    if (Negate) { f[c] -= E * contrib; }
                    else        { f[c] += E * contrib; }
                    ++row_offset;
                }
            };

            // Initialize f_i and load u_i (force and displacement at globalNode)
            if (ZeroInit) { f[0].setZero(); f[1].setZero(); }

            if (interior) {
                // Interior (skip bounds checks)

                // 2---------4---------7
                // |         |         |
                // | (-1, 0) | (0, 0)  |
                // |         |   ei    |
                // 1---------i---------6
                // |         |         |
                // | (-1,-1) | (0, -1) |
                // |         |         |
                // 0---------3---------5
                for (size_t c = 0; c < N; ++c) {
                    u_n[0][c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0] - 1, c);
                    u_n[1][c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0]    , c);
                    u_n[2][c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0] + 1, c);

                    u_n[3][c] = u.template block<SIMD_WIDTH, 1>(ni - 1, c);
                    u_i   [c] = u.template block<SIMD_WIDTH, 1>(ni    , c);
                    u_n[4][c] = u.template block<SIMD_WIDTH, 1>(ni + 1, c);

                    u_n[5][c] = u.template block<SIMD_WIDTH, 1>(ni + ninc[0] - 1, c);
                    u_n[6][c] = u.template block<SIMD_WIDTH, 1>(ni + ninc[0]    , c);
                    u_n[7][c] = u.template block<SIMD_WIDTH, 1>(ni + ninc[0] + 1, c);
                }

                if (!ZeroInit) {
                    f[0] = result.template block<SIMD_WIDTH, 1>(ni, 0);
                    f[1] = result.template block<SIMD_WIDTH, 1>(ni, 1);
                }

                using SV = Eigen::Map<const SIMDVec>;
                auto &Y = sim.m_youngModulusScaleFactor;
                accum(SV(&Y[ei - einc[0] - 1]), 6, u_n[0], u_n[1], u_n[3], u_i   ); // Element (-1, -1)
                accum(SV(&Y[ei - einc[0]    ]), 4, u_n[1], u_n[2], u_i   , u_n[4]); // Element (-1,  0)
                accum(SV(&Y[ei              ]), 0, u_i   , u_n[4], u_n[6], u_n[7]); // Element ( 0,  0)
                accum(SV(&Y[ei           - 1]), 2, u_n[3], u_i   , u_n[5], u_n[6]); // Element ( 0, -1)

                result.template block<SIMD_WIDTH, 1>(ni, 0) = f[0];
                result.template block<SIMD_WIDTH, 1>(ni, 1) = f[1];
            }
            else {
                // Avoid buffer overrun when accessing values. Previously the code simply
                // clamped indices within the bounds since out-of-bounds values will get multiplied
                // by a zero entry of `E` anyway. However, in the layer-by-layer simulator,
                // the detached values might be `NaN`, which we don't want to propagate.
                const auto &guarded_u = [&](size_t i, int c) { if (i < nn) return u(i, c); return 0.0; };
                for (size_t c = 0; c < N; ++c) {
                    for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                        u_n[0][c][s] = guarded_u(ni - ninc[0] - 1 + s, c);
                        u_n[1][c][s] = guarded_u(ni - ninc[0]     + s, c);
                        u_n[2][c][s] = guarded_u(ni - ninc[0] + 1 + s, c);
                    }

                    for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                        u_n[3][c][s] = guarded_u(ni - 1 + s, c);
                        u_i   [c][s] = guarded_u(ni     + s, c);
                        u_n[4][c][s] = guarded_u(ni + 1 + s, c);
                    }

                    for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                        u_n[5][c][s] = guarded_u(ni + ninc[0] - 1 + s, c);
                        u_n[6][c][s] = guarded_u(ni + ninc[0]     + s, c);
                        u_n[7][c][s] = guarded_u(ni + ninc[0] + 1 + s, c);
                    }
                }

                if (!ZeroInit) {
                    for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                        size_t sni = ni + s;
                        if (sni >= nn) break;
                        f[0][s] = result(sni, 0); f[1][s] = result(sni, 1);
                    }
                }

                SIMDVec E;
                // Element (0, 0)
                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    bool elementOutOfBounds = (globalNode[0]     >= sim.NbElementsPerDimension()[0])
                                           || (globalNode[1] + s >= sim.NbElementsPerDimension()[1]);
                    E[s] = elementOutOfBounds ? 0 : sim.elementYoungModulusScaleFactor(ei + s);
                }
                accum(E, 0, u_i, u_n[4], u_n[6], u_n[7]);

                // Element (0, -1)
                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    bool elementOutOfBounds = (globalNode[0]         >= sim.NbElementsPerDimension()[0])
                                           || (globalNode[1] - 1 + s >= sim.NbElementsPerDimension()[1]); // wraps around!
                    E[s] = elementOutOfBounds ? 0 : sim.elementYoungModulusScaleFactor(ei - 1 + s);
                }
                accum(E, 2, u_n[3], u_i , u_n[5], u_n[6]);

                // Element (-1, -1)
                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    bool elementOutOfBounds = (globalNode[0] - 1     >= sim.NbElementsPerDimension()[0])  // wraps around!
                                           || (globalNode[1] - 1 + s >= sim.NbElementsPerDimension()[1]); // wraps around!
                    E[s] = elementOutOfBounds ? 0 : sim.elementYoungModulusScaleFactor(ei - 1 - einc[0] + s);
                }
                accum(E, 6, u_n[0], u_n[1], u_n[3], u_i);

                // Element (-1, 0)
                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    bool elementOutOfBounds = (globalNode[0] - 1 >= sim.NbElementsPerDimension()[0])  // wraps around!
                                           || (globalNode[1] + s >= sim.NbElementsPerDimension()[1]);
                    E[s] = elementOutOfBounds ? 0 : sim.elementYoungModulusScaleFactor(ei - einc[0] + s);
                }
                accum(E, 4, u_n[1], u_n[2], u_i, u_n[4]);

                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    if (globalNode[N - 1] + s >= sim.NbNodesPerDimension()[N - 1]) break;
                    result(ni + s, 0) = f[0][s]; result(ni + s, 1) = f[1][s];
                }

            }
        }, EigenNDIndex::Zero().eval(), EigenNDIndex{numNodesToVisit[0], (numNodesToVisit[1] + SIMD_WIDTH - 1) / SIMD_WIDTH});

        // When a fabrication mask is active, we skipped the top margin of
        // detached nodes; these must be zeroed out now if `ZeroInit` is true.
        if (!ZeroInit) return;
        const EigenNDIndex &nn_nD = sim.NbNodesPerDimension();
        const size_t detachedMargin = nn_nD[Sim::BUILD_DIRECTION] - numNodesToVisit[Sim::BUILD_DIRECTION];
        if (detachedMargin > 0) {
            static_assert(Sim::BUILD_DIRECTION == 1, "Alternate build direction case not implemented");
            parallel_for_range(nn_nD[0], [&](size_t i) {
                result.middleRows(sim.flatIndexForNodeConstexpr(EigenNDIndex{i, numNodesToVisit[1]}), detachedMargin).setZero();
            });
        }
    }
};
#endif

#if 1
template<typename Real_>
struct SpecializedTPSStencils<Real_, 1, 1, 1> {
    using EigenNDIndex        = Eigen::Array<size_t, 3, 1>;
    using Sim                 = TensorProductSimulator<Real_, 1, 1, 1>;
    using VField              = typename Sim::VField;
    using VNd                 = typename Sim::VNd;
    using PerElementStiffness = typename Sim::PerElementStiffness;
    using LocalDisplacements  = Eigen::Matrix<Real_, PerElementStiffness::RowsAtCompileTime, 1>;
    static constexpr size_t N = Sim::N;

    template<class F>
    static void visitIncidentElements(const Sim &sim, const EigenNDIndex &globalNode, F &&visitor) {
        static constexpr std::array<size_t, N> elementStride = {{1, 1, 1}};
        EigenNDIndex primaryE, localNode;
        for (size_t d = 0; d < N; ++d) {
            primaryE [d] = globalNode[d] / (elementStride[d]);
            localNode[d] = globalNode[d] % (elementStride[d]);
        }

        for (const auto &ean : sim.m_elementsAdjacentNode[0]) {
            EigenNDIndex e = primaryE + ean.first;
            if ((e >= sim.m_NbElementsPerDimension).any()) continue; // out of bounds check (Note -1 gets wrapped to `size_t` max).
            size_t ei = sim.elementIndexForGridCellUnchecked(e);
            auto enodes = (sim.m_referenceElementNodes + sim.flattenedFirstNodeOfElement(e)).eval();
            visitor(ei, e, ean.second, enodes);
        }
    }

    static constexpr size_t SIMD_WIDTH = VOXELFEM_SIMD_WIDTH;
    using SIMDVec = Eigen::Array<Real_, SIMD_WIDTH, 1>;
    template<bool ZeroInit = true, bool Negate = false>
    static void applyK(const Sim &sim, Eigen::Ref<const VField> u, VField &result) {
        const size_t nn = sim.numNodes();
        result.resize(sim.numNodes(), N);
        const PerElementStiffness &K0 = sim.fullDensityElementStiffnessMatrix();
        EigenNDIndex numNodesToVisit = sim.nondetachedNodesPerDim();
        const size_t nn_z = sim.NbNodesPerDimension()[N - 1];
        IndexRangeVisitor<N, /* Parallel = */ true>::run([&](EigenNDIndex globalNode) {
            globalNode[N - 1] *= SIMD_WIDTH;
            const auto &primaryE = globalNode;
            size_t ni = sim.flatIndexForNodeConstexpr(globalNode);
            const auto &einc = sim.   m_ElementIndexIncrement;
            const auto &ninc = sim.m_NodeGlobalIndexIncrement;
            size_t ei = sim.elementIndexForGridCellUnchecked(primaryE);
            const bool interior = ((primaryE - 1) < sim.NbElementsPerDimension()).all() // wraps around!
                                && (primaryE[0]              < sim.NbElementsPerDimension()[0])
                                && (primaryE[1]              < sim.NbElementsPerDimension()[1])
                                && (primaryE[2] + SIMD_WIDTH < sim.NbElementsPerDimension()[2]);

            SIMDVec f[N];

            // Accumulate an element's contribution to f using its corner displacements.
            auto accum = [&K0, &f](SIMDVec E, size_t row_offset,
                                   SIMDVec u0[N], SIMDVec u1[N], SIMDVec u2[N], SIMDVec u3[N],
                                   SIMDVec u4[N], SIMDVec u5[N], SIMDVec u6[N], SIMDVec u7[N]) {
                for (size_t c = 0; c < N; ++c) {
                    SIMDVec contrib;
                    // Leverage symmetry of K0 to scan down its columns.
                    contrib  = K0( 0, row_offset) * u0[0]; contrib += K0( 1, row_offset) * u0[1]; contrib += K0( 2, row_offset) * u0[2];
                    contrib += K0( 3, row_offset) * u1[0]; contrib += K0( 4, row_offset) * u1[1]; contrib += K0( 5, row_offset) * u1[2];
                    contrib += K0( 6, row_offset) * u2[0]; contrib += K0( 7, row_offset) * u2[1]; contrib += K0( 8, row_offset) * u2[2];
                    contrib += K0( 9, row_offset) * u3[0]; contrib += K0(10, row_offset) * u3[1]; contrib += K0(11, row_offset) * u3[2];
                    contrib += K0(12, row_offset) * u4[0]; contrib += K0(13, row_offset) * u4[1]; contrib += K0(14, row_offset) * u4[2];
                    contrib += K0(15, row_offset) * u5[0]; contrib += K0(16, row_offset) * u5[1]; contrib += K0(17, row_offset) * u5[2];
                    contrib += K0(18, row_offset) * u6[0]; contrib += K0(19, row_offset) * u6[1]; contrib += K0(20, row_offset) * u6[2];
                    contrib += K0(21, row_offset) * u7[0]; contrib += K0(22, row_offset) * u7[1]; contrib += K0(23, row_offset) * u7[2];
                    if (Negate) { f[c] -= E * contrib; }
                    else        { f[c] += E * contrib; }
                    ++row_offset;
                }
            };

            // Initialize f_i and load u_i (force and displacement at globalNode)
            if (ZeroInit) { f[0].setZero(); f[1].setZero(); f[2].setZero(); }

            if (interior) {
                // Interior (skip bounds checks)

                //         z = -1                       z = 0                        z = 1
                // 6---------14--------23       7---------15--------24       8---------16--------25
                // |         |         |        |         |         |        |         |         |
                // | -1  0 -1| 0  0  -1|        |         |         |        | -1  0 0 | 0  0  0 |
                // |         |         |        |         |         |        |         |         |
                // 3---------12--------20       4---------i---------21       5---------13--------22
                // |         |         |        |         |         |        |         |         |
                // | -1 -1 -1| 0 -1  -1|        |         |         |        | -1 -1 0 | 0 -1  0 |
                // |         |         |        |         |         |        |         |         |
                // 0---------9---------17       1---------10--------18       2---------11--------19
                if (!ZeroInit) {
                    f[0] = result.template block<SIMD_WIDTH, 1>(ni, 0);
                    f[1] = result.template block<SIMD_WIDTH, 1>(ni, 1);
                    f[2] = result.template block<SIMD_WIDTH, 1>(ni, 2);
                }

                SIMDVec u_a[N], u_b[N], u_c[N], u_d[N], u_e[N], u_f[N], u_g[N], u_i[N];
                using SV = Eigen::Map<const SIMDVec>;
                auto &Y = sim.m_youngModulusScaleFactor;

                for (size_t c = 0; c < N; ++c) {
                    u_a[c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0] - ninc[1] - 1, c); // 0
                    u_b[c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0] - ninc[1]    , c); // 1
                    u_c[c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0]           - 1, c); // 3
                    u_d[c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0]              , c); // 4
                    u_e[c] = u.template block<SIMD_WIDTH, 1>(ni           - ninc[1] - 1, c); // 9
                    u_f[c] = u.template block<SIMD_WIDTH, 1>(ni           - ninc[1]    , c); // 10
                    u_g[c] = u.template block<SIMD_WIDTH, 1>(ni                     - 1, c); // 12
                    u_i[c] = u.template block<SIMD_WIDTH, 1>(ni                        , c); // i
                }

                accum(SV(&Y[ei - einc[0] - einc[1] - 1]), 21, u_a, u_b, u_c, u_d, u_e, u_f, u_g, u_i); // Element (-1,  -1, -1): 0, 1, 3, 4, 9, 10, 12, i

                for (size_t c = 0; c < N; ++c) {
                    u_a[c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0] - ninc[1] + 1, c); // 2
                    u_c[c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0]           + 1, c); // 5
                    u_e[c] = u.template block<SIMD_WIDTH, 1>(ni           - ninc[1] + 1, c); // 11
                    u_g[c] = u.template block<SIMD_WIDTH, 1>(ni                     + 1, c); // 13
                }

                accum(SV(&Y[ei - einc[0] - einc[1]]), 18, u_b, u_a, u_d, u_c, u_f, u_e, u_i, u_g); // Element (-1,  -1,  0): 1, 2, 4, 5, 10 ,11, i, 13

                for (size_t c = 0; c < N; ++c) {
                    u_a[c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0] + ninc[1]    , c); //  7
                    u_b[c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0] + ninc[1] + 1, c); //  8
                    u_e[c] = u.template block<SIMD_WIDTH, 1>(ni           + ninc[1]    , c); // 15
                    u_f[c] = u.template block<SIMD_WIDTH, 1>(ni           + ninc[1] + 1, c); // 16
                }

                accum(SV(&Y[ei - einc[0] ]), 12, u_d, u_c, u_a, u_b, u_i, u_g, u_e, u_f); // Element (-1,   0,  0): 4, 5, 7, 8, i, 13, 15, 16

                for (size_t c = 0; c < N; ++c) {
                    u_a[c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0] + ninc[1]    , c); //  7
                    u_b[c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0] + ninc[1] + 1, c); //  8
                    u_e[c] = u.template block<SIMD_WIDTH, 1>(ni           + ninc[1]    , c); // 15
                    u_f[c] = u.template block<SIMD_WIDTH, 1>(ni           + ninc[1] + 1, c); // 16
                }

                for (size_t c = 0; c < N; ++c) {
                    u_b[c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0]           - 1, c); // 3
                    u_c[c] = u.template block<SIMD_WIDTH, 1>(ni - ninc[0] + ninc[1] - 1, c); // 6
                    u_f[c] = u.template block<SIMD_WIDTH, 1>(ni                     - 1, c); // 12
                    u_g[c] = u.template block<SIMD_WIDTH, 1>(ni           + ninc[1] - 1, c); // 14
                }

                accum(SV(&Y[ei - einc[0] - 1]), 15, u_b, u_d, u_c, u_a, u_f, u_i, u_g, u_e); // Element (-1,   0, -1): 3, 4, 6, 7, 12, i, 14, 15

                for (size_t c = 0; c < N; ++c) {
                    u_a[c] = u.template block<SIMD_WIDTH, 1>(ni + ninc[0]           - 1, c); // 20
                    u_b[c] = u.template block<SIMD_WIDTH, 1>(ni + ninc[0]              , c); // 21
                    u_c[c] = u.template block<SIMD_WIDTH, 1>(ni + ninc[0] + ninc[1] - 1, c); // 23
                    u_d[c] = u.template block<SIMD_WIDTH, 1>(ni + ninc[0] + ninc[1]    , c); // 24
                }

                accum(SV(&Y[ei - 1]),  3, u_f, u_i, u_g, u_e, u_a, u_b, u_c, u_d); // Element ( 0,   0, -1): 12, i, 14, 15, 20, 21, 23, 24

                for (size_t c = 0; c < N; ++c) {
                    u_c[c] = u.template block<SIMD_WIDTH, 1>(ni           - ninc[1] - 1, c); // 9
                    u_d[c] = u.template block<SIMD_WIDTH, 1>(ni           - ninc[1]    , c); // 10
                    u_e[c] = u.template block<SIMD_WIDTH, 1>(ni + ninc[0] - ninc[1] - 1, c); // 17
                    u_g[c] = u.template block<SIMD_WIDTH, 1>(ni + ninc[0] - ninc[1]    , c); // 18
                }

                accum(SV(&Y[ei - einc[1] - 1]),  9, u_c, u_d, u_f, u_i, u_e, u_g, u_a, u_b); // Element ( 0,  -1, -1): 9, 10, 12, i, 17, 18, 20, 21

                for (size_t c = 0; c < N; ++c) {
                    u_a[c] = u.template block<SIMD_WIDTH, 1>(ni           - ninc[1] + 1, c); // 11
                    u_c[c] = u.template block<SIMD_WIDTH, 1>(ni                     + 1, c); // 13
                    u_e[c] = u.template block<SIMD_WIDTH, 1>(ni + ninc[0] - ninc[1] + 1, c); // 19
                    u_f[c] = u.template block<SIMD_WIDTH, 1>(ni + ninc[0]           + 1, c); // 22
                }

                accum(SV(&Y[ei - einc[1]]),  6, u_d, u_a, u_i, u_c, u_g, u_e, u_b, u_f); // Element ( 0,  -1,  0): 10, 11, i, 13, 18, 19, 21, 22

                for (size_t c = 0; c < N; ++c) {
                    u_a[c] = u.template block<SIMD_WIDTH, 1>(ni           + ninc[1]    , c); // 15
                    u_d[c] = u.template block<SIMD_WIDTH, 1>(ni           + ninc[1] + 1, c); // 16
                    u_e[c] = u.template block<SIMD_WIDTH, 1>(ni + ninc[0] + ninc[1]    , c); // 24
                    u_g[c] = u.template block<SIMD_WIDTH, 1>(ni + ninc[0] + ninc[1] + 1, c); // 25
                }

                accum(SV(&Y[ei]),  0, u_i, u_c, u_a, u_d, u_b, u_f, u_e, u_g); // Element ( 0,   0,  0): i, 13, 15, 16, 21, 22, 24, 25

                result.template block<SIMD_WIDTH, 1>(ni, 0) = f[0];
                result.template block<SIMD_WIDTH, 1>(ni, 1) = f[1];
                result.template block<SIMD_WIDTH, 1>(ni, 2) = f[2];
            }
            else {
                SIMDVec u_n[26][N], u_i[N];

                // Avoid buffer overrun when accessing values. Previously the code simply
                // clamped indices within the bounds since out-of-bounds values will get multiplied
                // by a zero entry of `E` anyway. However, in the layer-by-layer simulator,
                // the detached values might be `NaN`, which we don't want to propagate.
                const auto &guarded_u = [&](size_t i, int c) { if (i < nn) return u(i, c); return 0.0; };
                for (size_t c = 0; c < N; ++c) {
                    for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                        u_n[ 0][c][s] = guarded_u(ni - ninc[0] - ninc[1] - 1 + s, c);
                        u_n[ 1][c][s] = guarded_u(ni - ninc[0] - ninc[1]     + s, c);
                        u_n[ 2][c][s] = guarded_u(ni - ninc[0] - ninc[1] + 1 + s, c);
                        u_n[ 3][c][s] = guarded_u(ni - ninc[0]           - 1 + s, c);
                        u_n[ 4][c][s] = guarded_u(ni - ninc[0]               + s, c);
                        u_n[ 5][c][s] = guarded_u(ni - ninc[0]           + 1 + s, c);
                        u_n[ 6][c][s] = guarded_u(ni - ninc[0] + ninc[1] - 1 + s, c);
                        u_n[ 7][c][s] = guarded_u(ni - ninc[0] + ninc[1]     + s, c);
                        u_n[ 8][c][s] = guarded_u(ni - ninc[0] + ninc[1] + 1 + s, c);
                        u_n[ 9][c][s] = guarded_u(ni           - ninc[1] - 1 + s, c);
                        u_n[10][c][s] = guarded_u(ni           - ninc[1]     + s, c);
                        u_n[11][c][s] = guarded_u(ni           - ninc[1] + 1 + s, c);
                        u_n[12][c][s] = guarded_u(ni                     - 1 + s, c);
                        u_i    [c][s] = guarded_u(ni                         + s, c);
                        u_n[13][c][s] = guarded_u(ni                     + 1 + s, c);
                        u_n[14][c][s] = guarded_u(ni           + ninc[1] - 1 + s, c);
                        u_n[15][c][s] = guarded_u(ni           + ninc[1]     + s, c);
                        u_n[16][c][s] = guarded_u(ni           + ninc[1] + 1 + s, c);
                        u_n[17][c][s] = guarded_u(ni + ninc[0] - ninc[1] - 1 + s, c);
                        u_n[18][c][s] = guarded_u(ni + ninc[0] - ninc[1]     + s, c);
                        u_n[19][c][s] = guarded_u(ni + ninc[0] - ninc[1] + 1 + s, c);
                        u_n[20][c][s] = guarded_u(ni + ninc[0]           - 1 + s, c);
                        u_n[21][c][s] = guarded_u(ni + ninc[0]               + s, c);
                        u_n[22][c][s] = guarded_u(ni + ninc[0]           + 1 + s, c);
                        u_n[23][c][s] = guarded_u(ni + ninc[0] + ninc[1] - 1 + s, c);
                        u_n[24][c][s] = guarded_u(ni + ninc[0] + ninc[1]     + s, c);
                        u_n[25][c][s] = guarded_u(ni + ninc[0] + ninc[1] + 1 + s, c);
                    }
                }

                if (!ZeroInit) {
                    for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                        size_t sni = ni + s;
                        if (sni >= nn) break;
                        f[0][s] = result(sni, 0);
                        f[1][s] = result(sni, 1);
                        f[2][s] = result(sni, 2);
                    }
                }

                SIMDVec E;

                // Element ( 0,   0,  0)
                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    bool elementOutOfBounds = (globalNode[0]     >= sim.NbElementsPerDimension()[0])
                                           || (globalNode[1]     >= sim.NbElementsPerDimension()[1])
                                           || (globalNode[2] + s >= sim.NbElementsPerDimension()[2]);
                    E[s] = elementOutOfBounds ? 0 : sim.elementYoungModulusScaleFactor(ei + s);
                }
                accum(E, 0, u_i, u_n[13], u_n[15], u_n[16], u_n[21], u_n[22], u_n[24], u_n[25]);

                // Element ( 0,   0, -1)
                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    bool elementOutOfBounds = (globalNode[0]         >= sim.NbElementsPerDimension()[0])
                                           || (globalNode[1]         >= sim.NbElementsPerDimension()[1])
                                           || (globalNode[2] - 1 + s >= sim.NbElementsPerDimension()[2]);
                    E[s] = elementOutOfBounds ? 0 : sim.elementYoungModulusScaleFactor(ei - 1 + s);
                }
                accum(E, 3, u_n[12], u_i, u_n[14], u_n[15], u_n[20], u_n[21], u_n[23], u_n[24]);

                // Element ( 0,  -1, 0)
                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    bool elementOutOfBounds = (globalNode[0]         >= sim.NbElementsPerDimension()[0])
                                           || (globalNode[1] - 1     >= sim.NbElementsPerDimension()[1])
                                           || (globalNode[2]     + s >= sim.NbElementsPerDimension()[2]);
                    E[s] = elementOutOfBounds ? 0 : sim.elementYoungModulusScaleFactor(ei - einc[1] + s);
                }
                accum(E, 6, u_n[10], u_n[11], u_i, u_n[13], u_n[18], u_n[19], u_n[21], u_n[22]);

                // Element ( 0,  -1, -1)
                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    bool elementOutOfBounds = (globalNode[0]         >= sim.NbElementsPerDimension()[0])
                                           || (globalNode[1] - 1     >= sim.NbElementsPerDimension()[1])
                                           || (globalNode[2] - 1 + s >= sim.NbElementsPerDimension()[2]);
                    E[s] = elementOutOfBounds ? 0 : sim.elementYoungModulusScaleFactor(ei - einc[1] - 1 + s);
                }
                accum(E, 9, u_n[9], u_n[10], u_n[12], u_i, u_n[17], u_n[18], u_n[20], u_n[21]);

                // Element (-1,   0,  0)
                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    bool elementOutOfBounds = (globalNode[0] - 1     >= sim.NbElementsPerDimension()[0])
                                           || (globalNode[1]         >= sim.NbElementsPerDimension()[1])
                                           || (globalNode[2]     + s >= sim.NbElementsPerDimension()[2]);
                    E[s] = elementOutOfBounds ? 0 : sim.elementYoungModulusScaleFactor(ei - einc[0] + s);
                }
                accum(E, 12, u_n[4], u_n[5], u_n[7], u_n[ 8], u_i, u_n[13], u_n[15], u_n[16]);

                // Element (-1,   0, -1)
                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    bool elementOutOfBounds = (globalNode[0] - 1     >= sim.NbElementsPerDimension()[0])
                                           || (globalNode[1]         >= sim.NbElementsPerDimension()[1])
                                           || (globalNode[2] - 1 + s >= sim.NbElementsPerDimension()[2]);
                    E[s] = elementOutOfBounds ? 0 : sim.elementYoungModulusScaleFactor(ei - einc[0] - 1 + s);
                }
                accum(E, 15, u_n[3], u_n[4], u_n[6], u_n[ 7], u_n[12], u_i, u_n[14], u_n[15]);

                // Element (-1,  -1,  0)
                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    bool elementOutOfBounds = (globalNode[0] - 1     >= sim.NbElementsPerDimension()[0])
                                           || (globalNode[1] - 1     >= sim.NbElementsPerDimension()[1])
                                           || (globalNode[2]     + s >= sim.NbElementsPerDimension()[2]);
                    E[s] = elementOutOfBounds ? 0 : sim.elementYoungModulusScaleFactor(ei - einc[0] - einc[1] + s);
                }
                accum(E, 18, u_n[1], u_n[2], u_n[ 4], u_n[5], u_n[10], u_n[11], u_i, u_n[13]);

                // Element (-1,  -1, -1)
                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    bool elementOutOfBounds = (globalNode[0] - 1     >= sim.NbElementsPerDimension()[0])
                                           || (globalNode[1] - 1     >= sim.NbElementsPerDimension()[1])
                                           || (globalNode[2] - 1 + s >= sim.NbElementsPerDimension()[2]);
                    E[s] = elementOutOfBounds ? 0 : sim.elementYoungModulusScaleFactor(ei - einc[0] - einc[1] - 1 + s);
                }
                accum(E, 21, u_n[0], u_n[1], u_n[3], u_n[4], u_n[9], u_n[10], u_n[12], u_i);

                for (size_t s = 0; s < SIMD_WIDTH; ++s) {
                    if (globalNode[N - 1] + s >= nn_z) break;
                    result.row(ni + s) << f[0][s], f[1][s], f[2][s];
                }
            }
        }, EigenNDIndex::Zero().eval(), EigenNDIndex{numNodesToVisit[0], numNodesToVisit[1], (numNodesToVisit[2] + SIMD_WIDTH - 1) / SIMD_WIDTH});

        // When a fabrication mask is active, we skipped the top margin of
        // detached nodes; these must be zeroed out now if `ZeroInit` is true.
        if (!ZeroInit) return;
        const EigenNDIndex &nn_nD = sim.NbNodesPerDimension();
        static_assert(Sim::BUILD_DIRECTION == 1, "Alternate build direction case not implemented");
        const size_t detachedNodes_per_x = nn_nD[2] * (nn_nD[Sim::BUILD_DIRECTION] - numNodesToVisit[Sim::BUILD_DIRECTION]);
        if (detachedNodes_per_x > 0) {
            parallel_for_range(nn_nD[0], [&](size_t i) {
                result.middleRows(sim.flatIndexForNodeConstexpr(EigenNDIndex{i, numNodesToVisit[1], 0}), detachedNodes_per_x).setZero();
            });
        }
    }
};
#endif

#endif /* end of include guard: TPSSTENCILS_HH */
