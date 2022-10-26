#ifndef TENSORPRODUCTSIMULATOR
#define TENSORPRODUCTSIMULATOR

#include "NDVector.hh"
#include "TensorProductPolynomialInterpolant.hh"
#include "TensorProductQuadrature.hh"
#include "TPSStencils.hh"
#include "ParallelVectorOps.hh"
#include "VoxelFEMBenchmark.hh"

#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Materials.hh>
#include <MeshFEM/Parallelism.hh>
#include <MeshFEM/ParallelAssembly.hh>
#include <MeshFEM/Fields.hh>
#include <MeshFEM/Geometry.hh>

#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/MSHFieldParser.hh>
#include <MeshFEM/BoundaryConditions.hh>
#include <MeshFEM/util.h>

#include <string>
#include <cmath>
#include <bitset>
#include <unordered_map>
#include <typeinfo>

/// A struct for a mesh element
/// \tparam Degrees: The degrees of the Lagrange Polynomials of the FEM basis, in each dimension
template<typename Real_, size_t... Degrees>
struct Element_T {
    constexpr static size_t N = sizeof...(Degrees);
    // Number of nodes in the element
    static constexpr size_t nNodes = NumBasisFunctions<Degrees...>::value;
    // Number of vectorial basis function
    static constexpr size_t nVecPhi = N * NumBasisFunctions<Degrees...>::value;

    using ETensor         = ElasticityTensor<Real_, N>;
    using StiffnessMatrix = Eigen::Matrix<Real_, nVecPhi, nVecPhi>;
    using SMatrix         = SymmetricMatrixValue<Real_, N>;
    using ElementLoad     = Eigen::Matrix<Real_, N,  nNodes>;
    using Strain          = TensorProductPolynomialInterpolant<SMatrix, Real_, Degrees...>;
    using VField          = Eigen::Matrix<Real_, Eigen::Dynamic, N, Eigen::ColMajor>;
    using VNd             = VecN_T<Real_, N>;
    using VXd             = VecX_T<Real_>;
    using MXd             = Eigen::Matrix<Real_, Eigen::Dynamic, Eigen::Dynamic>;

    using StiffnessMatrixQuadrature = TensorProductQuadrature<Real_, (2 * Degrees)...>;

    // Set the stretching of the element: constant along the axis - no shear or weird deformation)
    // Also caches the corresponding strains
    void setStretchings(const VNd &Stretchings) {
        m_element_stretch = Stretchings;
        m_strains = Strains<Real_, Degrees...>::getStrains(m_element_stretch);
        m_volume = m_element_stretch.prod();
    }

    Real_ stretching(size_t dimension) const { return m_element_stretch.at(dimension); }

    // Compute the Element stiffness matrix Ke = int_elem [Strain : E : Strain] dV
    // Assumes the element stretchings have been set
    // Ke: Element stiffness matrix
    // E_tensor: Element Elasticity tensor
    // density: Element density
    void Stiffness(StiffnessMatrix &Ke, const ETensor &E_tensor, const Real_ density) const {
        Real_ volume = Volume();
        for (size_t i = 0; i < m_strains.size(); ++i) {
            for (size_t j = i; j < m_strains.size(); ++j) {
                Ke(i, j) = StiffnessMatrixQuadrature::integrate(
                        [&](const VNd &p) {
                            return m_strains[i](p).doubleContract(E_tensor.doubleContract(m_strains[j](p)));
                        }
                );
            }
        }

        Ke *= density * volume; // volume is the norm of the Jacobian of the transformation to the reference element
    }

    // Computes the element stress load
    // l: element stress, load, assigned in the function
    // cstress:  element stress "C : e^{ij}"
    void constantStressLoad(ElementLoad &l, const SMatrix &cstress) const {
        Real_ volume = Volume();
        // Loop over the nodes
        for (size_t j = 0; j < nNodes; ++j) {
            // Loop over the strain components for the node
            for (size_t i = 0; i < N; ++i) {
                l(i,j) = TensorProductQuadrature<Real_, Degrees...>::integrate(
                        [&](const VNd &p) {
                            return m_strains[N*j + i](p).doubleContract(cstress);
                        }
                );
            }
        }

        l *= volume;
    }

    // Computes the element strainload under contant unit strain e
    // l: Element load, assigned during computation
    // E_tensor: Element elasticity tensor
    // density: element density
    // cstrain: constant unit strain e^{ij}
    void constantStrainLoad(ElementLoad &l,
                            const ETensor &E_tensor,
                            const Real_ density,
                            const SMatrix &cstrain) const {
        constantStressLoad(l, E_tensor.doubleContract(cstrain));
        l *= density;
    }

    // assigns to e the strain of the element "ni" of the simulator,
    // NodeIndexGetter returns the index of the nodes of element ni
    // in the indexing of the simulator
    template<class NodeIndexGetter>
    void strain(const NodeIndexGetter &ni, const VField &u, Strain &e) const {
        e.coeffs.fill(SMatrix()); // SMatrix zero-initializes
        for (size_t j = 0; j < nNodes; ++j) {
            for (size_t c = 0; c < N; ++c) {
                Real_ u_comp = u(ni(j), c);
                for (size_t n = 0; n < Strain::size(); ++n)
                    e[n] += u_comp * m_strains[N * j + c][n];
            }
        }
    }

    template<class NodeIndexGetter>
    void averageStrain(const NodeIndexGetter &ni, const VField &u, SMatrix &e) const {
        e.clear();
        for (size_t j = 0; j < nNodes; ++j) {
            for (size_t c = 0; c < N; ++c) {
                // Note: we could optimize this with an "average" or
                // "integrate" method on our TensorProductPolynomialInterpolant
                e += u(ni(j), c) *
                    TensorProductQuadrature<Real_, Degrees...>::integrate(
                        [&](const VNd &p) {
                            return m_strains[N*j + c](p);
                        });
            }
        }
    }

    template<class NodeIndexGetter>
    static VXd interpolate(const NodeIndexGetter &ni, const MXd &u, const VNd &p) {
        TensorProductPolynomialInterpolant<VXd, Real_, Degrees...> interpolant;
        for (size_t n = 0; n < nNodes; ++n)
            interpolant[n] = u.row(ni(n));
        return interpolant(p);
    }

    Real_ Volume() const {
        return m_volume;
    }

private:
    VNd m_element_stretch;
    std::vector<typename Strains<Real_, Degrees...>::Strains_> m_strains;
    Real_ m_volume;
};

// Select interpolation law for densities: SIMP(default) or
// RAMP (https://link.springer.com/content/pdf/10.1007/s001580100129.pdf)
enum class InterpolationLaw {SIMP, RAMP};

/// A class for the finite element simulator
/// \tparam Degrees: The degrees of the Lagrange Polynomials of the FEM basis, in each dimension
template<typename Real_, size_t... Degrees>
struct TensorProductSimulator  {
    static constexpr size_t N = sizeof...(Degrees);
    using Element = Element_T<Real_, Degrees...>;

    static constexpr size_t numNodesPerElem = Element::nNodes;

    using Scalar  = Real_;
    using TPS     = TensorProductSimulator;
    using TMatrix = TripletMatrix<Triplet<Real_>>;
    using VField  = Eigen::Matrix<Real_, Eigen::Dynamic, N, Eigen::ColMajor>; // Column-major is better for SIMD
    using SMatrix = typename Element::SMatrix;
    using Strain  = typename Element::Strain;
    using MNd     = Eigen::Matrix<Real_, N, N>;
    using VNd     = VecN_T<Real_, N>;
    using Point   = VNd;
    using VXd     = VecX_T<Real_>;
    using MXd     = Eigen::Matrix<Real_, Eigen::Dynamic, Eigen::Dynamic>;
    using PerElementStiffness    = typename Element::StiffnessMatrix;
    using BlockSuiteSparseMatrix = CSCMatrix<SuiteSparse_long, MNd>;
    using Stencils = TPSStencils<Real_, Degrees...>;
    static constexpr size_t KeSize = PerElementStiffness::RowsAtCompileTime;

    using StiffnessMatrixQuadrature = typename Element::StiffnessMatrixQuadrature;

    using ETensor = ElasticityTensor<Real_, N>;

    using EigenNDIndex = Eigen::Array<size_t, N, 1>;
    template<class STLNDIndex>
    static auto eigenNDIndexWrapper(const STLNDIndex &idxs) -> decltype(Eigen::Map<const EigenNDIndex>(idxs.data())) {
        assert(size_t(idxs.size()) == N);
        return Eigen::Map<const EigenNDIndex>(idxs.data());
    }

    using ElementNodeIndexer   = NDArrayIndexer<N, (    Degrees + 1)...>;
    using ElementVertexIndexer = NDArrayIndexer<N, (0 * Degrees + 2)...>;

    // Construct a TPS given the domain (left-lower and right-upper points)
    // and the number of elements per dimension
    TensorProductSimulator(const BBox<VNd> &domain,
                           const EigenNDIndex &numElemsPerDim) {
        m_NbElementsPerDimension = eigenNDIndexWrapper(numElemsPerDim);

        // Compute number of nodes belonging to an element, in each dimension
        m_NbNodesPerDimensionPerElement = eigenNDIndexWrapper(std::array<size_t, N>({{(Degrees+1)... }}));

        // Initialize domain bounding box
        m_domain = domain;

        // Compute total number of nodes along each dimension:
        // For each element, count all nodes except those it shares with its "positive neighbors"
        // Then add in the "positive border" nodes
        m_NbNodesPerDimension = m_NbElementsPerDimension.array() * (m_NbNodesPerDimensionPerElement.array() - 1) + 1;

        // Set NDVector of Nodes
        m_numNodes = m_NbNodesPerDimension.prod();
        NDVector<Real_>::getFlatIndexIncrements(m_NbNodesPerDimension, m_NodeGlobalIndexIncrement);

        m_hasDirichlet    .assign(m_numNodes, false);
        m_hasFullDirichlet.assign(m_numNodes, false);

        // Determine node spacing (used to compute node positions)
        VNd boxDimensions = m_domain.dimensions();
        m_nodeSpacing = boxDimensions.array() / (m_NbNodesPerDimension.template cast<Real_>().array() - 1.0); // distance (in the coordinate directions) of adjacent nodes

        // Set NDVector of elements and densities
        m_densities.resize(m_NbElementsPerDimension);
        m_numElements = m_densities.size();

        m_updateYoungModuli();

        // Get flattened index increments that corresponding to changing Nd element indices.
        m_densities.getFlatIndexIncrements(m_ElementIndexIncrement);
        // Get increments used to move a node in one element to the corresponding node in another element.
        for (size_t d = 0; d < N; ++d) {
            // First term converts from element Nd index increment to node Nd index increment; second
            // converts to flattened index.
            m_NodeGlobalIndexIncrementPerElementIncrement[d] = (m_NbNodesPerDimensionPerElement[d] - 1) * m_NodeGlobalIndexIncrement[d];
        }

        // Initialize the element aspect ratio (uniform grid) using the distance of adjacent nodes
        // (stretchings are used for mapping to reference element when intergrating)
        setStretchings(boxDimensions.array() / m_NbElementsPerDimension.template cast<Real_>().array());

        // Generate the array of global node indices for the 0th element's nodes;
        // the nth element's nodes are just this array offset by the first node's index.
        {
            EigenNDIndex local_nNd = EigenNDIndex::Zero();

            // Update the globalNodeIndex associated with each element node as we enumerate them.
            size_t globalNodeIndex = 0;
            size_t back = 0;

            while (true) {
                m_referenceElementNodes[back++] = globalNodeIndex;

                // Increment N-digit counter
                // WARNING: this assumes the current local node ordering; it must
                // match ElementNodeIndexer flattening conventioning!
                ++local_nNd[N - 1];
                ++globalNodeIndex;
                for (size_t d = N - 1; local_nNd[d] == m_NbNodesPerDimensionPerElement[d]; --d) {
                    if (d == 0) return; // "most significant" digit has overflowed; we are done
                    globalNodeIndex += m_NodeGlobalIndexIncrement[d - 1] - m_NbNodesPerDimensionPerElement[d] * m_NodeGlobalIndexIncrement[d];
                    local_nNd[d] = 0;
                    ++local_nNd[d - 1];
                }
            }
        }
    }

    // If no domain bounding box is specified, use unit squares/cubes
    TensorProductSimulator(const EigenNDIndex &numElemsPerDim)
        : TensorProductSimulator(BBox<VNd>(VNd::Zero().eval(),
                                           numElemsPerDim.template cast<Real_>().matrix().eval()),
                                 numElemsPerDim) { }

    const BBox<VNd> &domain() const { return m_domain; }

    // Set the masked height for layer-by-layer fabrication process, given physical height
    void setFabricationMaskHeight(Real_ h, bool updateYoungModuli = true) {
        if (h < 0 || h > m_domain.maxCorner[BUILD_DIRECTION])
            throw std::runtime_error("Fabrication height (" + std::to_string(h) + ") has to be in between 0 and " + std::to_string(m_domain.maxCorner[BUILD_DIRECTION]));

        m_fabricationMaskHeight = h;

        // Determine the vertical index of the first layer of completely masked elements
        m_firstMaskedElementLayerIdx = std::ceil(m_fabricationMaskHeight / m_element_stretch[BUILD_DIRECTION] - 1e-10);

        // Determine the vertical index of the first layer of detached nodes. These are the nodes
        // immediately above the  "bottommost" nodes of the first layer of masked elements.
        m_firstDetachedNodeLayerIdx = m_firstMaskedElementLayerIdx * (m_NbNodesPerDimensionPerElement[BUILD_DIRECTION] - 1) + 1;

        // std::cout << "Set fabrication mask height: " << m_fabricationMaskHeight << std::endl;
        // std::cout << "m_firstMaskedElementLayerIdx: " << m_firstMaskedElementLayerIdx << std::endl;
        // std::cout << "m_firstDetachedNodeLayerIdx: " << m_firstDetachedNodeLayerIdx << std::endl;

        if (updateYoungModuli)
            m_updateYoungModuli();
    }

    void decrementFabricationMaskHeightByLayer(int layerIncrement) {
        if (size_t(m_firstMaskedElementLayerIdx) > NbElementsPerDimension()[BUILD_DIRECTION]) throw std::runtime_error("Mask must already be applied");
        if (m_firstMaskedElementLayerIdx < layerIncrement) throw std::runtime_error("Mask decrement of bounds");

        m_fabricationMaskHeight -= layerIncrement * m_nodeSpacing[BUILD_DIRECTION];
        m_firstMaskedElementLayerIdx -= layerIncrement;
        m_firstDetachedNodeLayerIdx = m_firstMaskedElementLayerIdx * (m_NbNodesPerDimensionPerElement[BUILD_DIRECTION] - 1) + 1;

        IR er = layerElementRange(m_firstMaskedElementLayerIdx, m_firstMaskedElementLayerIdx + layerIncrement);
        IndexRangeVisitor<N, /* Parallel = */ true>::run([&](const EigenNDIndex &eiND) {
                size_t e = elementIndexForGridCellUnchecked(eiND);
                m_youngModulusScaleFactor[e] = 0.0;
            }, er.beginIndex(), er.endIndex());
    }

    // Set the masked height for layer-by-layer fabrication process, given number of layers "removed"
    void setFabricationMaskHeightByLayer(size_t l) {
        setFabricationMaskHeight(m_nodeSpacing[BUILD_DIRECTION] * l);
    }
    // Get the masked height for layer-by-layer fabrication process
    Real_ getFabricationMaskHeight() const { return m_fabricationMaskHeight; }

    void readMaterial (const std::string &materialFilepath) {
        // Set the material tensor
        Materials::Constant<N> mat(materialFilepath);
        setETensor(mat.getTensor());
    }

    const ETensor &getETensor() const {
        return m_E_tensor;
    }

    void setETensor(const ETensor &et) {
        m_E_tensor = et;
        m_updateK0();
        m_numericFactorizationUpToDate = false;
    }

    VNd nodePosition(const EigenNDIndex &ni_Nd) const {
        return m_domain.minCorner.array() + ni_Nd.template cast<Real_>().array() * m_nodeSpacing.array();
    }

    VNd nodePosition(size_t ni) const { return nodePosition(ndIndexForNode(ni)); }

    void applyPeriodicConditions(Real_ /* epsilon */ = 1e-7,
                                 bool /*ignoreMismatch*/ = false,
                                 std::unique_ptr<PeriodicCondition<N>> /*pc*/ = nullptr) {
        periodic_BC = true;
        m_solver.reset();
        m_hessianSparsityPattern.clear();
    }

    // A node is detached from the structure if its neighbor below on the current grid is at or above the fabrication mask height.
    bool isNodeDetached(const EigenNDIndex &ni_Nd) const {
        return int(ni_Nd[BUILD_DIRECTION]) >= m_firstDetachedNodeLayerIdx;
    }

    size_t isNodeDetached(size_t ni) const {
        return isNodeDetached(ndIndexForNode(ni)); }

    EigenNDIndex nondetachedNodesPerDim() const {
        EigenNDIndex result = m_NbNodesPerDimension;
        result[BUILD_DIRECTION] = std::min<size_t>(m_firstDetachedNodeLayerIdx, result[BUILD_DIRECTION]);
        return result;
    }

    EigenNDIndex nonmaskedElementsPerDim() const {
        EigenNDIndex result = m_NbElementsPerDimension;
        result[BUILD_DIRECTION] = std::min<size_t>(m_firstMaskedElementLayerIdx, result[BUILD_DIRECTION]);
        return result;
    }

    bool isElementMasked(const EigenNDIndex &eiND) const {
        return int(eiND[BUILD_DIRECTION]) >= m_firstMaskedElementLayerIdx;
    }

    bool isElementMasked(size_t ei) const {
        EigenNDIndex eiND = ndIndexForElement(ei);
        return isElementMasked(eiND);
    }

    size_t firstDetachedNodeLayerIdx()  const { return m_firstDetachedNodeLayerIdx; }
    size_t firstMaskedElementLayerIdx() const { return m_firstMaskedElementLayerIdx; }

    using IR = IndexRange<EigenNDIndex>;
    IR layerElementRange(size_t lbegin, size_t lend) const {
        EigenNDIndex begin = EigenNDIndex::Zero();
        EigenNDIndex end = NbElementsPerDimension();
        begin[BUILD_DIRECTION] = lbegin;
        end  [BUILD_DIRECTION] = lend;
        return IR(begin, end);
    }

    IR layerNodeRange(size_t lbegin, size_t lend) const {
        IR range = layerElementRange(lbegin, lend);
        range.beginIndex() = firstNodeOfElement(range.beginIndex());
        range.  endIndex() = firstNodeOfElement(range.endIndex() - EigenNDIndex::Ones()) + m_NbNodesPerDimensionPerElement;
        return range;
    }

    // Call `f(nodeBlockStart, nodeBlockSize)` on each contiguous chunk of non-detached nodes in parallel,
    // optionally including `margin` layers of adjacent detached nodes.
    template<class F>
    void maskedNodalVisitBlocks(const F &f, size_t margin=0) const {
        static_assert(BUILD_DIRECTION == 1, "Implementation assumes build direction is 1");
        size_t height = std::min<size_t>(m_firstDetachedNodeLayerIdx, NbNodesPerDimension()[BUILD_DIRECTION]);
               height = std::min<size_t>(height + margin, NbNodesPerDimension()[BUILD_DIRECTION]); // add in the margin (second step is to avoid overflow)

        size_t blockSize = height * (NbNodesPerDimension().prod() / NbNodesPerDimension().head(2).prod());
        return tbb::parallel_for(tbb::blocked_range<size_t>(0, m_NbNodesPerDimension[0]),
                [&, blockSize](const tbb::blocked_range<size_t> &r) {
                    for (size_t outerIndex = r.begin(); outerIndex < r.end(); ++outerIndex)
                    f(outerIndex * m_NodeGlobalIndexIncrement[0], blockSize);
                });
    }

    template<class F>
    return_type<F> maskedNodalReduceSum(const F &f, const return_type<F> &initValue = return_type<F>()) const {
        static_assert(BUILD_DIRECTION == 1, "Implementation assumes build direction is 1");
        size_t height = std::min<size_t>(m_firstDetachedNodeLayerIdx, NbNodesPerDimension()[BUILD_DIRECTION]);
        size_t blockSize = height * (NbNodesPerDimension().prod() / NbNodesPerDimension().head(2).prod());
        return tbb::parallel_reduce(tbb::blocked_range<size_t>(0, m_NbNodesPerDimension[0]), initValue,
                 [&, blockSize](const tbb::blocked_range<size_t> &r, return_type<F> total) {
                        for (size_t outerIndex = r.begin(); outerIndex < r.end(); ++outerIndex)
                            total += f(outerIndex * m_NodeGlobalIndexIncrement[0], blockSize);
                    return total;
                 }, std::plus<return_type<F>>());
    }

    Real maskedNodalSquaredNorm(const VField &x) const {
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("maskedNodalSquaredNorm");
        return maskedNodalReduceSum([&](size_t nodeStart, size_t numNodes) { return x.middleRows(nodeStart, numNodes).squaredNorm(); });
    }

    Real maskedNodalDotProduct(const VField &u, const VField &v) const {
        return maskedNodalReduceSum([&](size_t nodeStart, size_t numNodes) { return u.middleRows(nodeStart, numNodes).cwiseProduct(v.middleRows(nodeStart, numNodes)).sum(); });
    }

    void maskedNodalCopyParallel(const VField &in, VField &out, size_t margin = 0) const {
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("maskedNodalCopyParallel");
        out.resize(in.rows(), in.cols());
        maskedNodalVisitBlocks([&](size_t nodeStart, size_t numNodes) { out.middleRows(nodeStart, numNodes) = in.middleRows(nodeStart, numNodes); }, margin);
    }

    void maskedNodalSetZeroParallel(VField &out, size_t margin = 0) const {
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("maskedNodalCopyParallel");
        out.resize(numNodes(), N);
        maskedNodalVisitBlocks([&](size_t nodeStart, size_t numNodes) { out.middleRows(nodeStart, numNodes).setZero(); }, margin);
    }

    // Support for building/updating the force/Dirichlet boundary conditions in
    // an dense format that is then compressed into the sparse format used by
    // the solver.
    struct BCBuilder {
        BCBuilder(const TensorProductSimulator &tps) {
            const std::vector<size_t> &forceNodes            = tps.m_forceNodes;
            const VField &forceNodeForces                    = tps.m_forceNodeForces;
            const std::vector<size_t> &dirichletNodes        = tps.m_dirichletNodes;
            const VField &dirichletNodeDisplacements         = tps.m_dirichletNodeDisplacements;
            const std::vector<ComponentMask> &componentMasks = tps.m_dirichletComponents;

            if (    forceNodes.size() != size_t(           forceNodeForces.rows())) throw std::runtime_error(    "Force size mismatch");
            if (dirichletNodes.size() != size_t(dirichletNodeDisplacements.rows())) throw std::runtime_error("Dirichlet size mismatch");
            if (dirichletNodes.size() != componentMasks.size())                     throw std::runtime_error("Dirichlet size mismatch");

            forces                .setZero(tps.numNodes(), N);
            dirichletDisplacements.setZero(tps.numNodes(), N);
            dirichletComponents    .resize(tps.numNodes());

            for (size_t fni = 0; fni < forceNodes.size(); ++fni) {
                size_t ni      = forceNodes[fni];
                forces.row(ni) = forceNodeForces.row(fni);
            }

            for (size_t dni = 0; dni < dirichletNodes.size(); ++dni) {
                size_t ni                      = dirichletNodes[dni];
                dirichletDisplacements.row(ni) = dirichletNodeDisplacements.row(dni);
                dirichletComponents[ni]        = componentMasks[dni];
            }
        }

        void setDirichlet(size_t ni, const VNd &val, const ComponentMask &cmask_new) {
            auto &cmask_old = dirichletComponents.at(ni);
            for (size_t c = 0; c < N; ++c) {
                if (!cmask_new.has(c)) continue;
                // If a new component is being constrained, merge
                if (!cmask_old.has(c)) {
                    cmask_old.set(c);
                    dirichletDisplacements(ni, c) = val[c];
                }
                // Otherwise, make sure there isn't a conflict
                else {
                    if (std::abs(dirichletDisplacements(ni, c) - val[c]) > 1e-10)
                        throw std::runtime_error("Conflicting dirichlet displacements.");
                }
            }
        }

        void setDirichletComponent(size_t ni, size_t d, Real_ val) {
            dirichletComponents.at(ni).set(d);
            dirichletDisplacements(ni, d) = val;
        }

        void setForce(size_t ni, const VNd &f) { forces.row(ni) = f; }

        void apply(TensorProductSimulator &tps) const {
            // Compress Dirichlet conditions
            std::vector<size_t> &dirichletNodes                            = tps.m_dirichletNodes;
            VField &dirichletNodeDisplacements                             = tps.m_dirichletNodeDisplacements;
            std::vector<ComponentMask> &dirichletNodeComponents            = tps.m_dirichletComponents;
            std::vector<bool> &hasFullDirichlet                            = tps.m_hasFullDirichlet;
            std::vector<bool> &hasDirichlet                                = tps.m_hasDirichlet;
            std::unordered_map<size_t, size_t> &dirichletConstraintForNode = tps.m_dirichletConstraintForNode;

            const size_t nn = dirichletComponents.size();
            dirichletNodes.clear();
            hasDirichlet    .assign(nn, false);
            hasFullDirichlet.assign(nn, false);
            for (size_t ni = 0; ni < nn; ++ni) {
                if (dirichletComponents[ni].hasAny(N)) {
                    dirichletNodes.push_back(ni);
                    hasDirichlet[ni] = true;
                    hasFullDirichlet[ni] = dirichletComponents[ni].hasAll(N);
                }
            }

            const size_t ndn = dirichletNodes.size();
            dirichletNodeComponents.resize(ndn);
            dirichletNodeDisplacements.resize(ndn, N);
            dirichletConstraintForNode.clear();
            for (size_t dni = 0; dni < ndn; ++dni) {
                size_t ni = dirichletNodes[dni];
                dirichletNodeComponents[dni] = dirichletComponents[ni];
                dirichletNodeDisplacements.row(dni) = dirichletDisplacements.row(ni);
                dirichletConstraintForNode[ni] = dni;
            }

            // Compress Neumann conditions
            std::vector<size_t> &forceNodes = tps.m_forceNodes;
            VField &forceNodeForces         = tps.m_forceNodeForces;

            forceNodes.clear();
            for (size_t ni = 0; ni < nn; ++ni) {
                if (forces.row(ni).squaredNorm() != 0.0)
                    forceNodes.push_back(ni);
            }

            const size_t nfn = forceNodes.size();
            forceNodeForces.resize(nfn, N);
            for (size_t fni = 0; fni < nfn; ++fni)
                forceNodeForces.row(fni) = forces.row(forceNodes[fni]);
        }

        VField forces, dirichletDisplacements;
        std::vector<ComponentMask> dirichletComponents;
    };

    friend struct BCBuilder;

    bool hasFullDirichlet(size_t ni) const { return m_hasFullDirichlet[ni]; } // Are all components of a node constrained?
    bool     hasDirichlet(size_t ni) const { return     m_hasDirichlet[ni]; } // Are at least some components constrained?

    const ComponentMask &dirichletComponents(size_t ni) const { return m_dirichletComponents[m_dirichletConstraintForNode.at(ni)]; }

    void zeroOutDirichletComponents(VField &u) const {
        parallel_for_range(m_dirichletNodes.size(), [&](size_t dni) {
                const auto &dc = m_dirichletComponents[dni];
                const size_t ni = m_dirichletNodes[dni];
                for (size_t d = 0; d < N; ++d) {
                    if (dc.has(d)) u(ni, d) = 0;
                }
            });
    }

    void enforceDirichletConditions(VField &u) const {
        parallel_for_range(m_dirichletNodes.size(), [&](size_t dni) {
                const auto &dc = m_dirichletComponents[dni];
                const size_t ni = m_dirichletNodes[dni];
                for (size_t d = 0; d < N; ++d) {
                    if (dc.has(d)) u(ni, d) = m_dirichletNodeDisplacements(dni, d);
                }
            });
    }


    // Translate constraint conditions read from file into conditions on every single node.
    // Constraints can be Dirichlet-like (displacements) or Neumann-like (forces) but can be applied also to internal nodes.
    // Note: forces are expressed in Newtons, they are not defined per unit area as usual Neumann conditions.
    // TODO: properly handle addition of new loads...
    void applyDisplacementsAndLoads(const std::vector<CondPtr<N>> &conds) {
        if (m_dirichletNodes.size() + m_forceNodes.size() > 0) throw std::runtime_error("Boundary condition updates unsupported");
        // Set up evaluator environment
        ExpressionEnvironment env;
        env.setVectorValue("mesh_size_", m_domain.dimensions());
        env.setVectorValue("mesh_min_",  m_domain.minCorner);
        env.setVectorValue("mesh_max_",  m_domain.maxCorner);

        BCBuilder bcBuilder(*this);

        for (const auto &cond : conds) {
            env.setVectorValue("region_size_", cond->region->dimensions());
            env.setVectorValue("region_min_",  cond->region->minCorner);
            env.setVectorValue("region_max_",  cond->region->maxCorner);
            std::runtime_error illegalCondition("Illegal constraint type, only \"dirichlet\" and \"force\" accepted");
            std::runtime_error unimplementedForceType("Illegal force type, only \"force\" accepted");

            if (auto nc = dynamic_cast<const NeumannCondition<N> *>(cond.get())) {  // Force constraint
                // Identify the nodes involved in the constraint
                std::vector<size_t> nodesInRegion;
                for (size_t ni = 0; ni < m_numNodes; ni++)
                    if (nc->containsPoint(nodePosition(ni).template cast<double>()))
                        nodesInRegion.push_back(ni);
                if (nodesInRegion.size() == 0)
                    throw std::runtime_error("Force constraint region unmatched");

                // Store force conditions on the nodes
                for (size_t ni : nodesInRegion) {
                    env.setXYZ(nodePosition(ni));
                    if (nc->type == NeumannType::Force) // Force is distributed uniformly among all nodes in the region
                        bcBuilder.setForce(ni, (nc->traction(env) / nodesInRegion.size()).template cast<Real_>());
                    else throw unimplementedForceType;
                }
            }
            else if (auto dc = dynamic_cast<const DirichletCondition<N> *>(cond.get())) {  // Displacement constraint
                size_t numNodesInRegion = 0;
                for (size_t ni = 0; ni < m_numNodes; ni++) {
                    auto p = nodePosition(ni).template cast<double>().eval();
                    if (dc->containsPoint(p)) {
                        ++numNodesInRegion;
                        env.setXYZ(p);
                        bcBuilder.setDirichlet(ni, dc->displacement(env).template cast<Real_>(), dc->componentMask);
                    }
                }
                if (numNodesInRegion == 0)
                    throw std::runtime_error("Dirichlet region unmatched");
                m_solver.reset(); // Changing dirichlet conditions invalidates the symbolic factorization
            }
            else throw illegalCondition;
        }

        bcBuilder.apply(*this);
    }

    void applyDisplacementsAndLoadsFromFile(const std::string &path) {
        bool noRigidMotion;
        auto conds = readBoundaryConditions<N>(path, m_domain, noRigidMotion);
        applyDisplacementsAndLoads(conds);
    }

    void addDirichletCondition(const VNd &u, const VNd &minCorner, const VNd &maxCorner, const std::string &componentMask = "xyz") {
        BBox<VNd> region(minCorner, maxCorner);
        ComponentMask cmask(componentMask);
        BCBuilder bcBuilder(*this);

        for (size_t ni = 0; ni < m_numNodes; ni++) {
            if (region.containsPoint(nodePosition(ni)))
                bcBuilder.setDirichlet(ni, u, cmask);
        }

        bcBuilder.apply(*this);
    }

    // Assemble a vector of size numNodes() encoding the condition on displacement imposed for each direction of the node
    using ArrayXNb = Eigen::Array<bool, Eigen::Dynamic, N>;
    ArrayXNb getDirichletMask() const {
        const size_t nn = numNodes();
        ArrayXNb mask = ArrayXNb::Zero(nn, size_t(N));
        for (size_t dni = 0; dni < m_dirichletNodes.size(); ++dni)
            mask.row(m_dirichletNodes[dni]) = m_dirichletComponents[dni].template getArray<N>();

        return mask;
    }

    // Assemble a vector of size numNodes() encoding the force imposed for each direction of the node
    ArrayXNb getForceMask() const {
        ArrayXNb indicators = ArrayXNb::Zero(numNodes(), size_t(N));
        for (size_t fni = 0; fni < m_forceNodes.size(); fni++) {
            size_t ni = m_forceNodes[fni];
            const auto &f = m_forceNodeForces.row(fni);
            if (f[0] != 0)               indicators(ni, 0) = true; // X force
            if (f[1] != 0)               indicators(ni, 1) = true; // Y force
            if ((N == 3) && (f[2] != 0)) indicators(ni, 2) = true; // Z force
        }
        return indicators;
    }

    VXd getBCIndicatorField() const {
        std::bitset<N> indicator;
        VXd result = VXd::Zero(numNodes());
        for (size_t dni = 0; dni < m_dirichletNodes.size(); ++dni) {
            indicator.reset();
            for (size_t c = 0; c < N; ++c)
                indicator.set(c, m_dirichletComponents[dni].has(c));
            result[m_dirichletNodes[dni]] += indicator.to_ulong();
        }
        for (size_t fni = 0; fni < m_forceNodes.size(); ++fni) {
            indicator.reset();
            for (size_t c = 0; c < N; ++c)
                indicator.set(c, m_forceNodeForces.row(fni)[c] != 0.0);
            result[m_forceNodes[fni]] += (1 << N) * indicator.to_ulong();
        }
        return result;
    }

    void setUniformDensities(Real_ density) {
        if ((density > 1.0) || (density < 0))
            throw std::runtime_error("Density value (" + std::to_string(density) + ") has to be in between 0 and 1");
        m_densities.fill(density);
        m_updateYoungModuli();
    }

    void setDensity(size_t flatIndex, Real_ density) {
        if ((density > 1.0) || (density < 0))
            throw std::runtime_error("Density value (" + std::to_string(density) + ") has to be in between 0 and 1");
        m_densities[flatIndex] = density;
        m_updateYoungModuli();
    }

    // Get the grid index of a MeshIO element.
    size_t elementIndexFromMeshIO(const MeshIO::IOElement &e, const std::vector<MeshIO::IOVertex> &vertices, const BBox<Point3D> &bbox) const {
        Point3D center(Point3D::Zero());
        for (size_t v : e)
            center += vertices.at(v).point;
        center *= 1.0 / e.size();

        // Transform center to the simulator coordinate system from "bbox".
        // Get the center of the elements in the simulator coordinate system
        center = bbox.interpolationCoordinates(center);
        Point centerND;
        for (size_t i = 0; i < N; ++i)
            centerND[i] = center[i] * m_element_stretch[i] * m_NbElementsPerDimension[i];
        // return the corresponding index
        return getElementIndex(centerND);
    }

    // Fills vertices and elements with the vertices and elements of the simulator
    void getMesh(std::vector<MeshIO::IOVertex> &vertices,
                 std::vector<MeshIO::IOElement> &elements) const {
        if (!(N==2 || N == 3))
            throw std::runtime_error("Field writer not supported for dimension " + std::to_string(N));

        vertices.clear();
        elements.clear();

        for (size_t i = 0; i < numNodes(); ++i)
            vertices.push_back(nodePosition(i).template cast<double>().eval());

        size_t numVertPerElem = (N==2) ? 4 : 8;

        for (size_t ei = 0; ei < m_numElements; ++ei) {
            MeshIO::IOElement Ei;

            for (size_t pair = 0; pair < numVertPerElem/2; ++pair) {
                size_t v1 = elemVertexGlobalIndex(ei, pair*2);
                size_t v2 = elemVertexGlobalIndex(ei, pair*2+1);

                // Gmsh ordering: http://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
                if (pair % 2 == 0) {
                    Ei.push_back(v1);
                    Ei.push_back(v2);
                } else {
                    Ei.push_back(v2);
                    Ei.push_back(v1);
                }
            }
            elements.push_back(Ei);
        }
    }

    VXd getDensities() const {
        const auto &v = m_densities.data();
        return Eigen::Map<const VXd>(v.data(), v.size());
    }

    void setDensities(const Eigen::Ref<const VXd> &rho) {
        if (size_t(rho.size()) != m_densities.size()) throw std::runtime_error("Incorrect rho size");
        auto &v = m_densities.data();
        Eigen::Map<VXd>(v.data(), v.size()) = rho;
        m_updateYoungModuli();
    }

    // Copy upscaled densities from a coarser grid
    void setDensitiesFromCoarseGrid(size_t upscalingFactor, const VXd& rho) {
        if (numElements() != std::pow(upscalingFactor, N) * rho.size()) throw std::runtime_error("Incorrect rho size");
        NDVector<Real_> rho_c((EigenNDIndex)(m_NbElementsPerDimension / upscalingFactor)); // Densities on the coarse grid
        Eigen::Map<VXd> (rho_c.data().data(), rho_c.size()) = rho;
        for (size_t ei = 0; ei < m_densities.size(); ++ei) {
            EigenNDIndex eiND = m_densities.template unflattenIndex<N>(ei);
            EigenNDIndex eiND_c = eiND / upscalingFactor;
            m_densities[ei] = rho_c(eiND_c);
        }
        m_updateYoungModuli();
    }

    const NDVector<Real_> &elementDensities() const { return m_densities; }
    void setElementDensities(const NDVector<Real_> &d) {
        if (m_densities.size() != d.size())
            throw std::runtime_error("size mismatch");
        m_densities = d;
        m_updateYoungModuli();
    }

    // Writes the complete K Triplet matrix to outFile in binary format
    void writeFEMMatrix( const std::string &outFile, TMatrix &K) const {
        K.dumpBinary(outFile);
    }

    // Set the stretching in the simulator (m_element_stretch), as well as in all elements (stretching is identical for all elements)
    void setStretchings(const VNd &s) {
        m_element_stretch = s;
        m_element.setStretchings(s);
        m_updateK0();
    }

    // Assemble the global stiffnessMatrix into K, which should already be
    // initialized with the correct size. If K is a SuiteSparseMatrix, it
    // should also be initized with the correct sparsity pattern.
    // If `sparsity = true` we ensure nonzero values are written to every entry
    // that could possibly become nonzero.
    // WARNING: the stiffness matrix is stored in double precision regardless of the
    // `Real_` type of this simulator. This is because we currently only provide double
    // precision bindings for Cholmod.
    template<typename SparseMatrix_>
    void m_assembleStiffnessMatrix(SparseMatrix_ &K, bool sparsity = false) const {
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("Assembling stiffness matrix");
        auto accumToSparseMatrix = [&](size_t ei, const PerElementStiffness &Ke, SparseMatrix_ &_K) {
            constexpr size_t nNodes = Element::nNodes;
            for (size_t j = 0; j < nNodes; ++j) {
                size_t dj = DoF(elemNodeGlobalIndex(ei, j));
                for (size_t i = 0; i < nNodes; ++i) {
                    size_t di = DoF(elemNodeGlobalIndex(ei, i));
                    if (di > dj) continue; // accumulate only upper triangle.
                    for (size_t cj = 0; cj < N; ++cj) {
                        _K.addNZStrip(N * di, N * dj + cj,
                                Ke.col(N * j + cj).segment(N * i, std::min((N * dj + cj) - (N * di) + 1, size_t(N))).template cast<double>());
                    }
                }
            }
        };

        if (hasCachedElementStiffness() && !sparsity) {
            for (size_t ei = 0; ei < m_numElements; ++ei)
                accumToSparseMatrix(ei, cachedElementStiffnessMatrix(ei), K);
        }

        else {
            // Assembly (optimized exploiting regularity of the mesh)
            PerElementStiffness Ke, k0 = fullDensityElementStiffnessMatrix();
            for (size_t ei = 0; ei < m_numElements; ++ei) {
                Ke = elementYoungModulusScaleFactor(ei) * k0; // Each element contributes differently to the global stiffness matrix depending only on its Young modulus
                if (sparsity) Ke.setOnes();
                accumToSparseMatrix(ei, Ke, K);
            }
        }
    }

    void m_cacheSparsityPattern() const {
        TMatrix Ktrip(N * numDoFs(), N * numDoFs());
        Ktrip.symmetry_mode = TMatrix::SymmetryMode::UPPER_TRIANGLE;
        Ktrip.reserve(KeSize * KeSize * numElements()); // upper bound; difficult to predict exactly in periodic case.
        m_assembleStiffnessMatrix(Ktrip, true);
        Ktrip.sumRepeated();
        m_hessianSparsityPattern = SuiteSparseMatrix(Ktrip);
        m_hessianSparsityPattern.fill(1.0);
    }

    // Check if *any* block stiffness matrix has been cached.
    // WARNING: the user must manually ensure the cached matrix is kept up-to-date when densities are
    // updated (by calling `updateBlockK`).
    bool hasBlockK() const { return m_blockK.m != 0; }

    // Clear out the cached block stiffness matrix.
    void deallocateBlockK() { m_blockK.m = m_blockK.n = 0; }

    void allocateBlockK() {
        if (hasBlockK()) return;
        // Generate the "scalar-valued" sparsity pattern if it has not yet been created.

        const size_t nn = numNodes();
        auto &Ax = m_blockK.Ax;
        m_blockK.m = nn;
        m_blockK.n = nn;

        auto &Ai = m_blockK.Ai;
        auto &Ap = m_blockK.Ap;

        Ap.reserve(nn + 1);
        Ap.assign(1, 0); // first column start
        Ai.clear();
        Ai.reserve(std::pow(2, N) * numNodesPerElem * nn); // overestimate assuming no shared nodes.

        std::vector<size_t> adj;
        for (size_t ni = 0; ni < nn; ++ni) {
            adj.clear();
            EigenNDIndex n = ndIndexForNode(ni);
            visitIncidentElements(n, [&adj](const size_t /* ei */, const EigenNDIndex &/* e */, const size_t /* local_n */, const ENodesArray &enodes) {
                    for (size_t i = 0; i < numNodesPerElem; ++i)
                        adj.push_back(enodes[i]);
                });
            std::sort(adj.begin(), adj.end());
            adj.erase(std::unique(adj.begin(), adj.end()), adj.end());
            Ai.reserve(Ai.size() + adj.size());
            for (size_t nj : adj) Ai.push_back(nj);
            Ap.push_back(Ai.size()); // column end
        }
        Ai.shrink_to_fit();
        Ax.resize(Ai.size());
    }

    void accumulateToBlockK(size_t ei, const PerElementStiffness &Ke) {
        const size_t enode_offset = flattenedFirstNodeOfElement1D(ei);
        for (size_t j = 0; j < Element::nNodes; ++j) {
            size_t gj = m_referenceElementNodes[j] + enode_offset;
            SuiteSparse_long hint = std::numeric_limits<SuiteSparse_long>::max();
            for (size_t i = 0; i < Element::nNodes; ++i) {
                size_t gi = m_referenceElementNodes[i] + enode_offset;
                hint = m_blockK.addNZ(gi, gj, Ke.template block<N, N>(N * i, N * j), hint);
            }
        }
    }

    // Update the block-sparse-matrix representation of stiffness matrix K.
    // Note that this representation does *not* leverage symmetry and explicitly
    // stores the upper and lower triangle. This is to simplify and accelerate
    // the GS smoothing iterations and matvecs.
    void updateBlockK() {
        if (periodic_BC) throw std::runtime_error("Periodic case unimplemented");

        allocateBlockK();
        const size_t nn = numNodes();
        UNUSED(nn);

        m_blockK.setZero();

        // Basic sanity size checks for the existing sparsity pattern.
        assert((m_blockK.m == nn) && (m_blockK.n == nn) && (m_blockK.Ap.size() == nn + 1) && (m_blockK.Ap.back() == m_blockK.Ai.size()));

        // Assemble the block sparse matrix from the per-element stiffness matrices.
        if (hasCachedElementStiffness()) {
            FINE_BENCHMARK_SCOPED_TIMER_SECTION timer2("Assemble Cached Ke");
            for (size_t ei = 0; ei < m_numElements; ++ei)
                accumulateToBlockK(ei, cachedElementStiffnessMatrix(ei));
        }
        else {
            FINE_BENCHMARK_SCOPED_TIMER_SECTION timer2("Assemble Ke");
            // Assembly (exploits the identicalness of all elements)
            PerElementStiffness Ke;
            for (size_t ei = 0; ei < m_numElements; ++ei) {
                Ke = elementYoungModulusScaleFactor(ei) * m_K0;
                accumulateToBlockK(ei, Ke);
            }
        }
    }

    const BlockSuiteSparseMatrix &blockK() const { return m_blockK; }
          BlockSuiteSparseMatrix &blockK()       { return m_blockK; }

    Real_ elementYoungModulusScaleFactor(size_t ei) const { return m_youngModulusScaleFactor[ei]; }
    VXd getYoungModulusScaleFactor() const { return m_youngModulusScaleFactor;}

    // Derivative of 1/2 f . u
    template<class Derived>
    void computeComplianceGradient(const VField &u, Eigen::MatrixBase<Derived> &g) const {
        const size_t ne = numElements();
        if (size_t(g.size()) != ne) throw std::runtime_error("Unexpected size of g");

        bool hasSelfWeight = (m_gravity.squaredNorm() != 0);
        ISF intPhi = integratedShapeFunctions();

        const PerElementStiffness &k0 = fullDensityElementStiffnessMatrix();
        const size_t nodesPerElement = NbNodesPerElement(); // coincides with KeSize / N;
        parallel_for_range(numElements(), [&](size_t e) {
            // Masked elements do not contribute to compliance.
            if (isElementMasked(e)) {
                g[e] = 0;
                return;
            }
            Eigen::Matrix<Real_, KeSize, 1> u_e;
            // Get the element node displacements
            const size_t offset = flattenedFirstNodeOfElement1D(e);
            for (size_t lni = 0; lni < nodesPerElement; ++lni)
                u_e.template segment<N>(lni * N) = u.row(m_referenceElementNodes[lni] + offset);

            // Derivative of compliance with respect to density `i`, assuming `f` is fixed.
            if (m_interpolationLaw == InterpolationLaw::SIMP)
                g[e] = -0.5 * m_gamma * std::pow(elementDensity(e), m_gamma - 1.0) * (m_E_0 - m_E_min) * u_e.dot(k0 * u_e);
            else
                g[e] = -0.5 * (1 + m_q) * (m_E_0 - m_E_min) / std::pow(1 + m_q * (1 - elementDensity(e)), 2) * u_e.dot(k0 * u_e);
            if (hasSelfWeight) {
                // df/drho . u term
                for (size_t lni = 0; lni < nodesPerElement; ++lni)
                    g[e] += intPhi[lni] * m_gravity.dot(u_e.template segment<N>(lni * N)) * elementVolume(e);
            }
        });
    }

    // Add derivative of 1/2 f . u to g
    template<class Derived>
    void accumulateComplianceGradient(const VField &u, Eigen::MatrixBase<Derived> &g) const {
        const size_t ne = numElements();
        if (size_t(g.size()) != ne) throw std::runtime_error("Unexpected size of g");

        bool hasSelfWeight = (m_gravity.squaredNorm() != 0);
        ISF intPhi = integratedShapeFunctions();

        IR er = layerElementRange(0, std::min<size_t>(NbElementsPerDimension()[BUILD_DIRECTION], m_firstMaskedElementLayerIdx));

        const PerElementStiffness &k0 = fullDensityElementStiffnessMatrix();
        const size_t nodesPerElement = NbNodesPerElement(); // coincides with KeSize / N;
        IndexRangeVisitor<N, /* Parallel = */ true>::run([&](const EigenNDIndex &eiND) {
            size_t e = elementIndexForGridCellUnchecked(eiND);
            Eigen::Matrix<Real_, KeSize, 1> u_e;
            // Get the element node displacements
            const size_t offset = flattenedFirstNodeOfElement(eiND);
            for (size_t lni = 0; lni < nodesPerElement; ++lni)
                u_e.template segment<N>(lni * N) = u.row(m_referenceElementNodes[lni] + offset);

            // Derivative of compliance with respect to density `i`, assuming `f` is fixed.
            if (m_interpolationLaw == InterpolationLaw::SIMP)
                g[e] += -0.5 * m_gamma * std::pow(elementDensity(e), m_gamma - 1.0) * (m_E_0 - m_E_min) * u_e.dot(k0 * u_e);
            else
                g[e] += -0.5 * (1 + m_q) * (m_E_0 - m_E_min) / std::pow(1 + m_q * (1 - elementDensity(e)), 2) * u_e.dot(k0 * u_e);
            if (hasSelfWeight) {
                // df/drho . u term
                for (size_t lni = 0; lni < nodesPerElement; ++lni)
                    g[e] += intPhi[lni] * m_gravity.dot(u_e.template segment<N>(lni * N)) * elementVolume(e);
            }
            
        }, er.beginIndex(), er.endIndex());
    }

    // Evaluate the gradient of compliance given the equilibrium displacement `u`.
    NDVector<Real_> complianceGradient(const VField &u) const {
        NDVector<Real_> g(NbElementsPerDimension());
        auto g_flat = Eigen::Map<VXd>(g.data().data(), g.size());
        computeComplianceGradient(u, g_flat);
        return g;
    }

    // Evaluate the gradient of compliance given the equilibrium displacement `u`.
    VXd complianceGradientFlattened(const VField &u) const {
        VXd g(numElements());
        computeComplianceGradient(u, g);
        return g;
    }

    VXd elementEnergyDensity(const VField &u) const {
        const size_t ne = numElements();
        VXd result(ne);

        const PerElementStiffness &k0 = fullDensityElementStiffnessMatrix();
        const size_t nodesPerElement = NbNodesPerElement(); // coincides with KeSize / N;
        parallel_for_range(numElements(), [&](size_t e) {
            Eigen::Matrix<Real_, KeSize, 1> u_e;
            // Get the element node displacements
            const size_t offset = flattenedFirstNodeOfElement1D(e);
            for (size_t lni = 0; lni < nodesPerElement; ++lni)
                u_e.template segment<N>(lni * N) = u.row(m_referenceElementNodes[lni] + offset);

            result[e] = 0.5 * elementYoungModulusScaleFactor(e) * u_e.dot(k0 * u_e);
        });
        return result;
    }

    // Compute the element stiffness matrix assuming that all the elements
    // are assigned the same fabrication material and the density is 1.0.
    const PerElementStiffness &fullDensityElementStiffnessMatrix() const { return m_K0; }
    const MNd                 &fullDensityElementStiffnessMatrixDiag(size_t lni) const { return m_K0Diag[lni]; }

    PerElementStiffness elementStiffnessMatrix(size_t ei) const {
        if (hasCachedElementStiffness()) return m_KeCache[ei];
        return elementYoungModulusScaleFactor(ei) * fullDensityElementStiffnessMatrix();
    }

    const PerElementStiffness &cachedElementStiffnessMatrix(const size_t e) const {
        return m_KeCache.at(e);
    }

    auto &getStiffnessMatrixCacheVec() { return m_KeCache; }

    bool hasCachedElementStiffness() const {
        return m_KeCache.size() == numElements();
    }

    void invalidateNumericFactorization() { m_numericFactorizationUpToDate = false; }

    // Override the default element stiffness matrices (computed by scaling the
    // fullDensityElementStiffnessMatrix) with the stiffness matrices in `Kes`.
    // This is helpful, e.g., for the coarsened simulators used in Multigrid.
    void cacheCustomElementStiffnessMatrices(aligned_std_vector<PerElementStiffness> &&Kes) {
        m_KeCache = std::move(Kes);
        invalidateNumericFactorization();
    }

    void cacheElementStiffnessMatrices() {
        aligned_std_vector<PerElementStiffness> Ke;
        const size_t ne = numElements();
        Ke.reserve(ne);
        for (size_t i = 0; i < ne; ++i)
            Ke.push_back(elementStiffnessMatrix(i));
        cacheCustomElementStiffnessMatrices(Ke);
    }

    void clearCachedElementStiffness() { m_KeCache.clear(); invalidateNumericFactorization(); }

    // Compute global load under unit strain cstrain
    // F: global load
    // cstrain: constant unit strain
    VField constantStrainLoad(const typename Element::SMatrix &cstrain) const {
        typedef typename Element::ElementLoad ElementLoad;

        auto accumToGlobalVField = [&](size_t ei, const ElementLoad &l, VField &_F) {
            constexpr size_t nNodes = Element::nNodes;

            for (size_t j = 0; j < nNodes; ++j) {
                size_t dj = DoF(elemNodeGlobalIndex(ei, j));
                _F.row(dj) += l.col(j).transpose();
            }

        };

        VField F(numDoFs(), int(N));
        F.setZero();

        // Build all element loads in parallel, then collect
        std::vector<ElementLoad> elemLoads(m_numElements);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, m_numElements),
            [&](const tbb::blocked_range<size_t> &r) {
                for (size_t ei = r.begin(); ei < r.end(); ++ei)
                    m_element.constantStrainLoad(elemLoads[ei], m_E_tensor, m_densities[ei], cstrain);
            }
        );

        for (size_t i = 0; i < m_numElements; ++i)
            accumToGlobalVField(i, elemLoads[i], F);

        return F;
    }

    void elementStrain(size_t ei, const VField &u, Strain &e) const {
        auto nodeIndexGetter = [=](size_t n) { return elemNodeGlobalIndex(ei, n); };
        m_element.strain(nodeIndexGetter, u, e);
    }

    void elementAverageStrain(size_t ei, const VField &u, SMatrix &e) const {
        auto nodeIndexGetter = [=](size_t n) { return elemNodeGlobalIndex(ei, n); };
        m_element.averageStrain(nodeIndexGetter, u, e);
    }

    // Sample the finite element field `u` at the points specified by the rows of `p`.
    MXd sampleNodalField(const MXd &u, const VField &p) {
        const size_t np = p.rows();
        MXd result(np, u.cols());

        parallel_for_range(np, [&](size_t i) {
            // Obtain query point by closest point projection: simply clamp
            // each coordinate to within the bounds.
            Point q = m_domain.clamp(p.row(i).transpose());

            EigenNDIndex eiND;
            Point refCoords;
            std::tie(eiND, refCoords) = getElementAndReferenceCoordinates(q);

            auto nodeIndexGetter = [&](size_t n) { return elemNodeGlobalIndex(eiND, n); };
            result.row(i) = Element::interpolate(nodeIndexGetter, u, q).transpose();
        });

        return result;
    }

    void findFixedVars(std::vector<bool> &isFixed) const {
        std::vector<size_t> fixedVars;
        std::vector<Real_>   fixedVarValues;
        getDirichletVarsAndValues(fixedVars, fixedVarValues);
        // Pin down the null space
        for (size_t i = 0; i < numNodes(); ++i) {
            if (!isNodeDetached(i)) continue;
            for (size_t d = 0; d < N; ++d) {
                fixedVars.push_back(N * i + d);
                fixedVarValues.push_back(0);
            }
        }
        for (size_t i : fixedVars) isFixed.at(i) = true;
        for (Real_ v : fixedVarValues) { if (v != 0) throw std::runtime_error("Nonzero Dirichlet constraints currently unsupported"); }
    }

    // Solve for equilibrium under DoF load f
    VField solve(const VField &f) const {
        std::vector<bool> isFixed(N * numDoFs());
        findFixedVars(isFixed);

        VXd rhs(f.size());
        Eigen::Map<Eigen::Matrix<Real_, Eigen::Dynamic, N, Eigen::RowMajor>>(rhs.data(), f.rows(), f.cols()) = f;
        removeFixedEntriesInPlace(rhs, isFixed);

        if (!m_numericFactorizationUpToDate || !m_solver) {
            SuiteSparseMatrix K = getK();
            m_numericFactorizationUpToDate = true;

            K.rowColRemoval([&](SuiteSparse_long i) { return isFixed[i]; });
            // Discard symbolic factorization, fixed vars set has changed.
            if (m_solver && (m_solver->m() != size_t(K.m))) m_solver.reset();
            if (!m_solver) {
                auto Hsp = m_hessianSparsityPattern;
                Hsp.rowColRemoval([&](SuiteSparse_long i) { return isFixed[i]; });
                m_solver = std::make_unique<CholmodFactorizer>(Hsp);
            }
            m_solver->updateFactorization(K);
        }

        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("Elasticity Cholesky Solve");
        // TODO: do Cholesky factorization/backsub in floating point type `Real_`
        // (which could be single precision).
        // Until this is fixed, we should use sufficient coarsening so that the Cholesky
        // factorization is not a bottleneck.

        VXd x = m_solver->solve(rhs.template cast<double>().eval()).template cast<Real_>();
        VField x_node = dofToNodeField(extractFullSolution(x, isFixed));
        return x_node;
    }

    // Construct a vector of reduced components by removing the entries of "x" corresponding
    // to fixed variables. This is a (partial) inverse of extractFullSolution.
    void removeFixedEntriesInPlace(VXd &x, const std::vector<bool> &isFixed) const {
        size_t back = 0;
        for (size_t i = 0; i < size_t(x.size()); ++i)
            if (!isFixed[i]) x[back++] = x[i];
        x.conservativeResize(back);
    }

    // Extract the full linear system solution vector "x" from the reduced linear
    // system solution "xReduced" (which was solved by removing the rows/columns for fixed variables).
    VXd extractFullSolution(const VXd &xReduced, const std::vector<bool> &isFixed) const {
        VXd x(N * numDoFs());
        size_t back = 0;
        for (size_t i = 0; i < size_t(x.size()); ++i) {
            if (!isFixed[i]) x[i] = xReduced[back++];
            else             x[i] = 0.0;
        }
        assert(back == size_t(xReduced.size()));
        return x;
    }

    // Get the integral of the scalar-valued shape functions over the canonical
    // reference element [0, 1]^N
    using ISF = NDArray<Real_, N, (Degrees + 1)...>;
    ISF integratedShapeFunctions() const {
        NDArray<Real_, N, (Degrees + 1)...> result =
            TensorProductQuadrature<Real_, Degrees...>::integrate([&](const VNd &p) {
                        return TensorProductPolynomialEvaluator<Real_, Degrees...>::evaluate(p); // evaluate all shape functions at quadrature point `p`
                    });
        return result;
    }

    // Solve for equilibrium under loads and displacements imposed from .bc file
    VField solveWithImposedLoads() const { return solve(buildLoadVector()); }

    // Compute load on the DoFs from Neumann-like constraints
    VField buildLoadVector() const {
        VField load(numDoFs(), int(N));
        load.setZero();
        for (size_t fni = 0; fni < m_forceNodes.size(); ++fni)
            load.row(DoF(m_forceNodes[fni])) = m_forceNodeForces.row(fni);

        if (m_gravity.squaredNorm() != 0) {
            const size_t ne = numElements();
            auto intPhi = integratedShapeFunctions();
            for (size_t ei = 0; ei < ne; ++ei) {
                size_t offset = flattenedFirstNodeOfElement1D(ei);
                if (isElementMasked(ei)) continue;
                for (size_t lni = 0; lni < numNodesPerElem; ++lni) {
                    size_t ni = m_referenceElementNodes[lni] + offset;
                    load.row(ni) += m_gravity * (intPhi[lni] * elementDensity(ei) * elementVolume(ei));
                }
            }
        }
        return load;
    }

    // Update load vector `f` to account for the masking of the layer interval [lbegin, lend).
    // Assumes input `f` holds the load vector for layers [0, lend).
    void addLayerRemovalDeltaLoadVector(size_t lbegin, size_t lend, VField &f) const {
        if ((m_gravity.squaredNorm() == 0) || (std::abs(m_gravity.squaredNorm() - std::pow(m_gravity[BUILD_DIRECTION], 2)) > 1e-10))
            throw std::runtime_error("Unexpected gravity vector");

        auto intPhi = integratedShapeFunctions();
        visitElementLayersMulticolored</* Parallel = */ true>(
            [&f, &intPhi, this](const EigenNDIndex &eiND) {
                size_t offset = flattenedFirstNodeOfElement(eiND);
                size_t ei = elementIndexForGridCellUnchecked(eiND);
                Scalar weight = (m_gravity[BUILD_DIRECTION] * (elementDensity(ei) * elementVolume(ei)));
                for (size_t lni = 0; lni < numNodesPerElem; ++lni)
                    f(m_referenceElementNodes[lni] + offset, BUILD_DIRECTION) -= weight * intPhi[lni];
            }, lbegin, lend);
    }

    // Evaluate the dot products of each vector in `us` with the change in load vector induced by voiding
    // layers [lbegin, lend).
    VXd dotLayerRemovalDeltaLoadVector(size_t lbegin, size_t lend, const std::deque<std::unique_ptr<VField>> &us) const {
        if ((m_gravity.squaredNorm() == 0) || (std::abs(m_gravity.squaredNorm() - std::pow(m_gravity[BUILD_DIRECTION], 2)) > 1e-10))
            throw std::runtime_error("Unexpected gravity vector");
        ISF intPhi = integratedShapeFunctions();
        constexpr size_t nn = ISF::size();
        using ISF_Eig = Eigen::Matrix<Scalar, nn, 1>;
        ISF_Eig intPhi_eigen = Eigen::Map<const ISF_Eig>(intPhi.data());

        const size_t nu = us.size();
        std::vector<const VField *> u_ptrs;
        for (const auto &uptr : us) u_ptrs.push_back(uptr.get());

        IR range = layerElementRange(lbegin, lend);

        auto results = IndexRangeVisitorThreadLocal<N, VXd>::run([&](const EigenNDIndex &eiND, VXd &out) {
                size_t offset = flattenedFirstNodeOfElement(eiND);
                size_t ei = elementIndexForGridCellUnchecked(eiND);
                Scalar weight = (m_gravity[BUILD_DIRECTION] * (elementDensity(ei) * elementVolume(ei)));
                for (size_t i = 0; i < nu; ++i) {
                    const auto &u = *u_ptrs[i];
                    ISF_Eig u_corners;
                    for (size_t lni = 0; lni < numNodesPerElem; ++lni)
                        u_corners[lni] = u(m_referenceElementNodes[lni] + offset, BUILD_DIRECTION);
                    out[i] -= weight * u_corners.dot(intPhi_eigen);
                }
            }, /* constructor */ [nu](VXd &v) { v.setZero(nu); }, range.beginIndex(), range.endIndex());

        VXd result = VXd::Zero(nu);
        for (const auto &r : results)
            result += r.v;

        return result;
    }

    // Evaluate the change in `ui . K uj` for each `ui, uj` in `us` induced by
    // voiding layers [lbegin, lend).
    // (***Lower triangle only***)
    void layerRemovalDeltaUKU(size_t lbegin, size_t lend, const std::deque<std::unique_ptr<VField>> &us_deque, VField &scratch, MXd &result) const {
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("layerRemovalDeltaUKU");
        const size_t nu = us_deque.size();

        const PerElementStiffness &K0 = fullDensityElementStiffnessMatrix();
        const auto &enodes = referenceElementNodes();

        using LocalDisplacements = Eigen::Matrix<Real_, PerElementStiffness::RowsAtCompileTime, 1>;

        IR elementRange = layerElementRange(lbegin, lend);
        IR nodeRange    = layerNodeRange(lbegin, lend);
        EigenNDIndex nodeRangeSizes = nodeRange.endIndex() - nodeRange.beginIndex();
        const size_t nn = nodeRangeSizes.prod();
        auto compactFlatIdxForNode              = [&](const EigenNDIndex &niND) { return NDVector<Real_>::flatIndexConstexpr(niND -    nodeRange.beginIndex(), nodeRangeSizes); };
        static_assert(all_true<(Degrees == 1)...>::value, "The index conversion below assumes degree 1...");
        auto compactFlattenedFirstNodeOfElement = [&](const EigenNDIndex &eiND) { return NDVector<Real_>::flatIndexConstexpr(eiND - elementRange.beginIndex(), nodeRangeSizes); };
        ENodesArray compact_enodes;
        for (size_t i = 0; i < numNodesPerElem; ++i)
            compact_enodes[i] = compactFlatIdxForNode(ndIndexForNode(enodes[i]) + nodeRange.beginIndex());

        std::vector<const VField *> us;

        // We assume the scratch is zeroed out if it was allocated!
        // (The dot product loop below maintains this invariant).
        VField &delta_Ku = scratch;
        if (size_t(delta_Ku.rows()) != nn)
            setZeroParallel(delta_Ku, nn, N);

        for (size_t i = 0; i < nu; ++i) {
            us.push_back(us_deque[i].get());
            const VField &u = *us[i];
            visitElementLayersMulticolored</*Parallel = */ true>([&](const EigenNDIndex &eiND) {
                    const size_t ei           = (eiND * m_ElementIndexIncrement).sum();
                    const size_t enode_offset = flattenedFirstNodeOfElement(eiND);

                    LocalDisplacements Ke_u_local(K0.template middleCols<N>(0) * u.row(enodes[0] + enode_offset).transpose());
                    // Loop over nodal displacements
                    for (size_t m = 1; m < numNodesPerElem; ++m)
                        Ke_u_local += K0.template middleCols<N>(N * m) * u.row(enodes[m] + enode_offset).transpose();
                    Ke_u_local *= elementYoungModulusScaleFactor(ei);

                    // Loop over nodal matvec contributions
                    const size_t compact_enode_offset = compactFlattenedFirstNodeOfElement(eiND);
                    for (size_t m = 0; m < numNodesPerElem; ++m)
                        delta_Ku.row(compact_enodes[m] + compact_enode_offset) -= Ke_u_local.template segment<N>(N * m).transpose();
                }, lbegin, lend);

            // Compute lower triangle
            auto results = IndexRangeVisitorThreadLocal<N, VXd>::run([&, nu](const EigenNDIndex &niND, VXd &out) {
                size_t ni_compact = compactFlatIdxForNode(niND);
                size_t ni = flatIndexForNodeConstexpr(niND);
                VNd row = delta_Ku.row(ni_compact);
                for (size_t j = 0; j <= i; ++j)
                    out[j] += row.dot(us[j]->row(ni));
                delta_Ku.row(ni_compact).setZero();
            }, /* constructor */ [nu](VXd &v) { v.setZero(nu); }, nodeRange.beginIndex(), nodeRange.endIndex());

            for (const auto &rr : results)
                result.row(i) += rr.v.transpose();
        }
    }

    // Apply the stiffness matrix `K` to a displacement field `u`, computing
    // the elastic restoring forces.
    template<bool ZeroInit = true, bool Negate = false>
    void applyK(Eigen::Ref<const VField> u, VField &result) const {
        if (periodic_BC) throw std::runtime_error("Periodic case unimplemented");
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("applyK " + description());
        using LocalDisplacements = Eigen::Matrix<Real_, KeSize, 1>;

        const auto &enodes = m_referenceElementNodes;

        if (hasCachedElementStiffness()) {
            if (ZeroInit) setZeroParallel(result, numNodes(), N);
            visitElementsMulticolored([&](const EigenNDIndex &eiND) {
                    const size_t ei           = (eiND * m_ElementIndexIncrement).sum();
                    const size_t enode_offset = flattenedFirstNodeOfElement(eiND);
                    const auto &Ke = m_KeCache[ei];
                    LocalDisplacements Ke_u_local(Ke.template block<KeSize, N>(0, 0) * u.row(enodes[0] + enode_offset).transpose());
                    // Loop over nodal displacements
                    for (size_t m = 1; m < numNodesPerElem; ++m)
                        Ke_u_local += Ke.template block<KeSize, N>(0, N * m) * u.row(enodes[m] + enode_offset).transpose();
                    // Loop over nodal matvec contributions
                    for (size_t m = 0; m < numNodesPerElem; ++m) {
                        if (Negate) result.row(enodes[m] + enode_offset) -= Ke_u_local.template segment<N>(N * m).transpose();
                        else        result.row(enodes[m] + enode_offset) += Ke_u_local.template segment<N>(N * m).transpose();
                    }
                }, /* parallel = */ true);
        }
        else {
            SpecializedTPSStencils<Real_, Degrees...>::template applyK<ZeroInit, Negate>(*this, u, result);
        }
    }

    // Visit the elements in batches of the same color, where the coloring
    // is constructed so that elements of the same color have no nodes in common.
    // (There are 2^N such colors).
    // This is useful, e.g., for parallel lock-free stiffness matrix assembly/application.
    template<class F>
    void visitElementsMulticolored(F &&visitor, bool parallel = true, bool skipMasked = false) const {
        EigenNDIndex nbElementsToVisit = skipMasked ? nonmaskedElementsPerDim() : NbElementsPerDimension();
        HypercubeCornerVisitor<N>::run([&](const EigenNDIndex &colorOffset) {
            if ((colorOffset >= nbElementsToVisit).any()) return;
            EigenNDIndex numElemsOfColorPerDim = (nbElementsToVisit - 1 - colorOffset) / 2 + 1;

            // Visit elements of a color in an arbitrary order
            auto processElement = [&](const EigenNDIndex &eiNDWithinColor) { visitor(2 * eiNDWithinColor + colorOffset); };

            if (parallel) IndexRangeVisitor<N, /* Parallel = */  true>::run(processElement, EigenNDIndex::Zero().eval(), numElemsOfColorPerDim);
            else          IndexRangeVisitor<N, /* Parallel = */ false>::run(processElement, EigenNDIndex::Zero().eval(), numElemsOfColorPerDim);
        });
    }

    // Visit each element in the range of layers [lbegin, lend) along the build direction.
    // Elements are visited in parallel using a multicoloring approach that ensures the
    // visitor can write to the elements' nodes while retaining thread safety.
    template<bool Parallel, class F>
    void visitElementLayersMulticolored(F &&visitor, size_t lbegin, size_t lend) const {
        EigenNDIndex begin = EigenNDIndex::Zero();
        EigenNDIndex end = NbElementsPerDimension();
        begin[BUILD_DIRECTION] = lbegin;
        end  [BUILD_DIRECTION] = lend;
        EigenNDIndex nbElementsToVisit = end - begin;

        HypercubeCornerVisitor<N>::run([&](const EigenNDIndex &colorOffset) {
            if ((colorOffset >= nbElementsToVisit).any()) return;
            EigenNDIndex numElemsOfColorPerDim = (nbElementsToVisit - 1 - colorOffset) / 2 + 1;

            // Visit elements of a color in an arbitrary order
            auto processElement = [&](const EigenNDIndex &eiNDWithinColor) { visitor(2 * eiNDWithinColor + colorOffset + begin); };

            IndexRangeVisitor<N, Parallel>::run(processElement, EigenNDIndex::Zero().eval(), numElemsOfColorPerDim);
        });
    }


    VXd debugMulticolorElementVisit() const {
        VXd result(numElements());
        size_t i = 0;

        visitElementsMulticolored([&](const EigenNDIndex &globalE) {
                std::cout << "visiting element: " << (globalE).transpose() << "(" << m_densities.flatIndex(globalE) << ")" << std::endl;
                result[m_densities.flatIndex(globalE)] = i++;
            }, /* parallel */ false);
        return result;
    }

    VField applyK(Eigen::Ref<const VField> u) const {
        VField result;
        applyK</* ZeroInit = */ true>(u, result);
        return result;
    }

    template<bool ZeroInit = true, bool Negate = false>
    void applyBlockK(Eigen::Ref<const VField> u, VField &result) const {
        if (!hasBlockK()) throw std::logic_error("Attempting to apply a nonexisting block matrix");
        FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("applyBlockK " + description());
        m_blockK.template applyTransposeParallel<ZeroInit, Negate>(u, result);
    }

    SuiteSparseMatrix getK() const {
        if (m_hessianSparsityPattern.nnz() == 0)
            m_cacheSparsityPattern();
        SuiteSparseMatrix K = m_hessianSparsityPattern;
        K.setZero();
        m_assembleStiffnessMatrix(K);
        return K;
    }

    // Returns a VField of the size of the mesh, given the values of the VField at the dof nodes
    template<class Vec>
    VField dofToNodeField(const Vec &dofValues) const {
        if (size_t(dofValues.size()) != N * numDoFs()) {
            throw std::runtime_error("dofToNodeField : invalid number of values in input : "
                                     + std::to_string(dofValues.size()) + " -- should be "
                                     + std::to_string(numDoFs()));
        }

        const size_t nn = numNodes();
        VField F(nn, int(N));

        for (size_t i = 0; i < nn; ++i)
            F.row(i) = Eigen::Map<const VNd>(dofValues.data() + N * DoF(i), int(N));
        return F;
    }

    EigenNDIndex firstNodeOfElement(const EigenNDIndex &eiND) const {
        return eigenNDIndexWrapper(eiND) * (m_NbNodesPerDimensionPerElement - 1);
    }
    EigenNDIndex firstNodeOfElement(size_t ei) const {
        return firstNodeOfElement(m_densities.template unflattenIndex<N>(ei));
    }
    size_t flattenedFirstNodeOfElement1D(size_t ei) const {
        size_t result = 0;
        for (size_t d = N - 1; d < N; --d) { // right-to-left (least significant index to most significant)
            // Get d^th entry of unflattened index of `ei`
            size_t ei_d = ei % m_NbElementsPerDimension[d];
            ei /= m_NbElementsPerDimension[d];

            // Convert to d^th entry of unflattened index of element's first node
            size_t ni_d = ei_d * (m_NbNodesPerDimensionPerElement[d] - 1);

            // Accumulate to flattened index of first element node.
            result += ni_d * m_NodeGlobalIndexIncrement[d];
        }
        return result;
    }

    size_t flattenedFirstNodeOfElement(const EigenNDIndex &eiND) const {
        return (eiND * m_NodeGlobalIndexIncrementPerElementIncrement).sum();
    }

    // Get the 1D index of the local node `nd_local_idx` of element `ei`.
    template<class NdLocalIndex, typename enable = decltype(std::declval<NdLocalIndex>()[0])>
    size_t elemNodeGlobalIndex(size_t ei, const NdLocalIndex &nd_local_idx) const {
        // 1D index ei => ND element index => minimum corner node index,
        // sum => ND node index => 1D index
        return flatIndexForNode(firstNodeOfElement(ei) + eigenNDIndexWrapper(nd_local_idx));
    }

    // Returns the flattened index of the node "n" in element "ei"
    // where "n" is the linear linear index of the nodes belonging to the element "ei"
    // and "ei" is the linear index of the element in the vector m_densities
    size_t elemNodeGlobalIndex(size_t ei, size_t n) const {
        // 1D local node index n => ND node index offset from element min corner,
        return elemNodeGlobalIndex(ei, ElementNodeIndexer::unflattenIndex(n));
    }

    size_t elemNodeGlobalIndex(const EigenNDIndex &eiND, size_t n) const {
        // 1D index ei => ND element index => minimum corner node index,
        // sum => ND node index => 1D index
        return flatIndexForNode(firstNodeOfElement(eiND) + eigenNDIndexWrapper(ElementNodeIndexer::unflattenIndex(n)));
    }

    // Similar as elemNode, except "v" is the flat index of a vertex in the element "ei"
    // Vertices are the nodes at the corner of the elements, so there are 2 vertices per dimension
    size_t elemVertexGlobalIndex(size_t ei, size_t v) const {
        // Global ND element index
        EigenNDIndex e = ndIndexForElement(ei);
        // Global ND node index
        EigenNDIndex localVtxIndex;
        NDVector<Real_>::unflattenIndex(v, EigenNDIndex::Constant(2), localVtxIndex);

        // Go from vertex to node index.
        EigenNDIndex nodeIndex = (e + localVtxIndex) * (m_NbNodesPerDimensionPerElement - 1);

        // return flat global linear index of the node
        return flatIndexForNode(nodeIndex);
    }

    template<typename Visitor>
    void visitLocalNodeNdIndices(Visitor &&visit) const {
        EigenNDIndex local_nNd = EigenNDIndex::Zero();
        while (true) {
            visit(local_nNd);

            // Increment N-digit counter
            ++local_nNd[N - 1];
            for (size_t d = N - 1; local_nNd[d] == m_NbNodesPerDimensionPerElement[d]; --d) {
                if (d == 0) return; // "most significant" digit has overflowed; we are done
                local_nNd[d] = 0;
                ++local_nNd[d - 1];
            }
        }
    }

    template<typename Visitor>
    void visitElementNodes(size_t ei, Visitor &&visit) const {
        size_t offset = flattenedFirstNodeOfElement1D(ei);
        for (size_t i = 0; i < numNodesPerElem; ++i)
            visit(m_referenceElementNodes[i] + offset);
    }

    template<typename Derived, typename Visitor>
    void visitElementNodes(const Eigen::ArrayBase<Derived> &ei, Visitor &&visit) const {
        size_t offset = flattenedFirstNodeOfElement(ei);
        for (size_t i = 0; i < numNodesPerElem; ++i)
            visit(m_referenceElementNodes[i] + offset);
    }

    using ENodesArray = Eigen::Array<size_t, numNodesPerElem, 1>;
    ENodesArray elementNodes(size_t ei) const {
        return m_referenceElementNodes + flattenedFirstNodeOfElement1D(ei);
    }

    ENodesArray elementNodes(const EigenNDIndex &e) const {
        return m_referenceElementNodes + flattenedFirstNodeOfElement(e);
    }

    const ENodesArray &referenceElementNodes() const { return m_referenceElementNodes; }

    size_t elementIndexForGridCell(const EigenNDIndex &cellIdxs) const {
        return m_densities.flatIndex(cellIdxs);
    }

    size_t elementIndexForGridCellUnchecked(const EigenNDIndex &cellIdxs) const {
        return (cellIdxs * m_ElementIndexIncrement).sum();
    }

    EigenNDIndex ndIndexForElement(size_t ei) const { return m_densities.template unflattenIndex<N>(ei); }

    EigenNDIndex ndIndexForNode         (size_t ni) const { EigenNDIndex result; NDVector<Real_>::unflattenIndex         (ni, m_NbNodesPerDimension, result); return result; }
    EigenNDIndex ndIndexForNodeConstexpr(size_t ni) const { EigenNDIndex result; NDVector<Real_>::unflattenIndexConstexpr(ni, m_NbNodesPerDimension, result); return result; }

    size_t flatIndexForNode         (const EigenNDIndex &n) const { return NDVector<Real_>::flatIndex         (n, m_NbNodesPerDimension); }
    size_t flatIndexForNodeConstexpr(const EigenNDIndex &n) const { return NDVector<Real_>::flatIndexConstexpr(n, m_NbNodesPerDimension); }

    // Get periodic variable corresponding to node ni
    // (nodes on the "max face" of the period cell
    // get the index of the corresponding node on the
    // "min face")
    // (Use a linear indexing of the grid of nodes with the
    // maximum "layers" removed)
    size_t DoF(size_t ni) const {
        if (!periodic_BC) // without periodic conditions, DoF always coincide with node index
            return ni;

        // 1D node index => ND node index => ND dof node index => 1D dof node index on smaller grid

        // 1D node index => ND node index
        EigenNDIndex NDnodeIndices = ndIndexForNode(ni);

        // ND node index => ND dof node index
        for (size_t dim = 0; dim < N; ++dim) {
            if (NDnodeIndices[dim] == m_NbNodesPerDimension[dim]-1)
                NDnodeIndices[dim] = 0;
        }

        // ND dof node index => 1D dof node index on smaller grid
        EigenNDIndex NbNodesPerDimentionsOnSmallerGrid = m_NbNodesPerDimension;
        for (int dim = 0; dim < NbNodesPerDimentionsOnSmallerGrid.size(); dim++)
            --NbNodesPerDimentionsOnSmallerGrid[dim];
        return NDVector<Real_>::flatIndex(NDnodeIndices, NbNodesPerDimentionsOnSmallerGrid);
    }

    // Number of Degree of freedom is 1 less than number of nodes in each dimension because of periodic boundary condition
    size_t numDoFs() const {
        if (!periodic_BC) // without periodic conditions, DoF always coincide with node index
            return numNodes();

        size_t numDofs = 1;

        for (size_t dim = 0; dim < N; ++dim) {
            numDofs *= m_NbNodesPerDimension[dim]-1;
        }

        return numDofs;
    }

    size_t numNodes() const { return m_numNodes; }

    size_t numElements() const { return m_numElements; }

    Real_ volume() const {
        Real_ volume = 1;
        for (size_t dim = 0; dim < N; ++dim)
            volume *= m_NbElementsPerDimension[dim] * m_element_stretch[dim];
        return volume;
    }

    const ETensor &elementElasticityTensor(size_t /* ei */) const { return m_E_tensor; }
    Real_ elementDensity(size_t ei) const { return m_densities[ei]; }
    Real_ elementVolume (size_t /* ei */) const { return m_element.Volume(); }

    const EigenNDIndex &gridShape()                     const { return m_NbElementsPerDimension;                          }
    const EigenNDIndex &NbElementsPerDimension()        const { return m_NbElementsPerDimension;               }
    const EigenNDIndex &NbNodesPerDimensionPerElement() const { return m_NbNodesPerDimensionPerElement;        }
    const EigenNDIndex &NbNodesPerDimension()           const { return m_NbNodesPerDimension;                  }
          size_t        NbNodesPerElement()             const { return m_NbNodesPerDimensionPerElement.prod(); }

    // Get the origin (minCorner) of the design domain.
    const Point &origin() const { return m_domain.minCorner; }

    void setInterpolationLaw(InterpolationLaw law) { m_interpolationLaw = law; m_updateYoungModuli(); }
    void setRAMPFactor  (Real_ val) { m_q     = val; m_updateYoungModuli(); }
    void setE_min       (Real_ val) { m_E_min = val; m_updateYoungModuli(); }
    void setE_0         (Real_ val) { m_E_0   = val; m_updateYoungModuli(); }
    void setSIMPExponent(Real_ val) { m_gamma = val; m_updateYoungModuli(); }
    Real_ E_min()        const { return m_E_min; }
    Real_ E_0()          const { return m_E_0;   }
    Real_ SIMPExponent() const { return m_gamma; }
    Real_ RAMPFactor()   const { return m_q; }
    InterpolationLaw interpolationLaw() const { return m_interpolationLaw; }

    // NDindex of the element in which the point belongs
    EigenNDIndex getElementNDIndex(const Point &point) const {
        EigenNDIndex result;
        for (size_t d = 0; d < N; ++d) {
            Point relativePoint = point - m_domain.minCorner;    // shift origin to domain minimum corner
            Real_ floatIndex = relativePoint[d] / m_element_stretch[d];
            // Clamp samples samples on the grid boundary back into the grid.
            if (std::abs(floatIndex - m_NbElementsPerDimension[d]) < 1e-10) {
                result[d] = m_NbElementsPerDimension[d] - 1;
                continue;
            }
            result[d] = size_t(floatIndex);
            if (result[d] >= m_NbElementsPerDimension[d]) {
                std::cout << "dimension = " << d << std::endl;
                std::cout << "N = " << N << std::endl;
                std::cout << point << std::endl;
                std::cout << m_element_stretch[d] << std::endl;
                std::cout << point[d] << std::endl;
                throw std::runtime_error("Point out of bounds: " + std::to_string(result[d]) + " vs " + std::to_string(m_NbElementsPerDimension[d]));
            }
        }
        return result;
    }

    size_t getElementIndex(const Point &point) const { return m_densities.flatIndex(getElementNDIndex(point)); }

    // Get the index of the element containing point `p_in` and the reference/canonical
    // coordinates of the point within this element.
    std::pair<EigenNDIndex, Point> getElementAndReferenceCoordinates(Eigen::Ref<const Point> p_in) const {
        std::pair<EigenNDIndex, Point> result;
        Point &p_out = result.second;
        result.first = getElementNDIndex(p_in);
        // rescale p into [0,1]^d
        p_out = p_in - nodePosition(firstNodeOfElement(result.first));
        for (size_t d = 0; d < N; ++d)
            p_out[d] /= getStretchings()[d];
        return result;
    }

    // Set whether the no rigid translation constraint should be implemented
    // using a node pinning constraint.
    void setUsePinNoRigidTranslationConstraint(bool use) {
        m_useNRTPinConstraint = use;
    }

    void setGravity(const VNd &g) { m_gravity = g; }
    const VNd &getGravity() const { return m_gravity; }

    const VNd    &stretchings() const { return m_element_stretch; }
    const VNd &getStretchings() const { return m_element_stretch; }

    // Call visitor(ei, e, local_n, elementNodeOffset) for each incident element `ei`
    // where `local_n` is the index of `n` in element `ei` and
    // `elementNodes` is an array of global node indices for element `ei`.
    template<class F>
    void visitIncidentElements(const EigenNDIndex &globalNode, F &&visitor) const {
        SpecializedTPSStencils<Real_, Degrees...>::visitIncidentElements(*this, globalNode, std::forward<F>(visitor));
    }

    // Append to dirichletVars and dirichletValues the conditions stored in this simulator
    void getDirichletVarsAndValues(std::vector<size_t> &dirichletVars,
                                   std::vector<Real_> &dirichletValues) const {
        // Validate and convert to per-periodic-DoF constraints.
        // constraintDisplacements[i] holds the displacement to which
        // components constraintComponents[i] of DoF constraintDoFs[i] are
        // constrained.
        std::vector<Point>         constraintDisplacements;
        std::vector<size_t>        constraintDoFs;
        std::vector<ComponentMask> constraintComponents;
        // Index into the above arrays of a DoF's constraint, or -1 for none.
        // I.e. if constraintDoFs[i] > -1, the following holds:
        //  constraintDoFs[constraintIndex[i]] = i
        std::vector<int> constraintIndex(numDoFs(), -1);
        for (size_t dni = 0; dni < m_dirichletNodes.size(); ++dni) {
            size_t ni = m_dirichletNodes[dni];
            size_t dof = DoF(ni);
            VNd   u = m_dirichletNodeDisplacements.row(dni).transpose();
            auto cm = m_dirichletComponents[dni];

            if (constraintIndex[dof] < 0) {
                constraintIndex[dof] = constraintDoFs.size();
                constraintDoFs.push_back(dof);
                constraintDisplacements.push_back(u);
                constraintComponents   .push_back(cm);
            }
            else { // Error if there was already a constraint on dof
                std::cerr << "WARNING: Dirichlet condition on periodic "
                          << "boundary applies to all identified nodes."
                          << std::endl;
                auto diff = u - constraintDisplacements[constraintIndex[dof]];
                bool cdiffer = (cm != constraintComponents[constraintIndex[dof]]);
                if ((diff.norm() > 1e-10) || cdiffer) {
                    throw std::runtime_error("Mismatched Dirichlet constraint on periodic DoF");
                }
                // Ignore redundant but compatible Dirichlet conditions.
            }
        }


        for (size_t i = 0; i < constraintDoFs.size(); ++i) {
            for (size_t c = 0; c < N; ++c) {
                if (!constraintComponents[i].has(c)) continue;
                dirichletVars.push_back(N * constraintDoFs[i] + c);
                dirichletValues.push_back(constraintDisplacements[i][c]);
            }
        }
    }

    // Get a description of this simulator (currently just the element size)
    const std::string &description() const {
        if (m_description.size() == 0) {
            m_description = std::to_string(m_NbElementsPerDimension[0]);
            for (size_t i = 1; i < size_t(m_NbElementsPerDimension.size()); ++i)
                m_description += "x" + std::to_string(m_NbElementsPerDimension[i]);
        }
        return m_description;
    }

    // Support for layer-by-layer simulation:
    // Get the subregion of this structure that extends from the base plate (H = H_min)
    // up along the build direction to height fraction `hfrac` (in [0, 1]).
    static constexpr size_t BUILD_DIRECTION = 1;
    size_t getIntermediateFabricationShapeNumHeightElems(Real_ hfrac) const {
        size_t full_helems = NbElementsPerDimension()[BUILD_DIRECTION];
        size_t result = std::round(hfrac * full_helems);
        if (std::abs(hfrac * full_helems - result) > 1e-10) throw std::runtime_error("hfrac chops off a noninteger number of element layers");
        return result;
    }

    // Construct a simulator for an incremental fabrication shape.
    // validateBoundaryConditions: whether to verify if the original simulator (`this`) has
    // self-weight boundary conditions already imposed.
    // The created fabrication constraint always has the self-weight + Dirichlet build platform
    // conditions applied.
    std::shared_ptr<TPS> getIntermediateFabricationShape(Real_ hfrac, bool validateBoundaryConditions = true, InterpolationLaw law = InterpolationLaw::SIMP) const {
        if ((hfrac < 0) || (hfrac > 1)) throw std::runtime_error("hfrac is out of bounds");

        auto numElemsPerDim = NbElementsPerDimension();
        numElemsPerDim[BUILD_DIRECTION] = getIntermediateFabricationShapeNumHeightElems(hfrac);

        auto subdomain = domain();
        subdomain.maxCorner[BUILD_DIRECTION] = subdomain.minCorner[BUILD_DIRECTION] + hfrac * (subdomain.maxCorner[BUILD_DIRECTION] - subdomain.minCorner[BUILD_DIRECTION]);

        // Raw `new` is needed here for proper aligned allocation :(
        //  https://eigen.tuxfamily.org/bz/show_bug.cgi?id=1049
        std::shared_ptr<TPS> result(new TPS(subdomain, numElemsPerDim));
        result->setInterpolationLaw(law);

        // Configure material properties, densities, boundary conditions, etc.
        result->setETensor(getETensor());
        transferDensitiesToIntermediateFabricationShape(*result);

        result->m_E_min = m_E_min;
        result->m_E_0   = m_E_0;
        result->m_gamma = m_gamma;

        std::runtime_error unexpectedBC("Original simulator has unexpected boundary conditions for layer-by-layer simulation");
        if (m_gravity.squaredNorm() == 0) {
            if (validateBoundaryConditions) throw unexpectedBC;
            VNd g(VNd::Zero());
            g[BUILD_DIRECTION] = -1;
            result->setGravity(g);
        }
        else result->setGravity(getGravity());

        if (validateBoundaryConditions) {
            if (!m_forceNodes.empty()) throw unexpectedBC;
            for (size_t ni = 0; ni < numNodes(); ++ni) {
                if (nodePosition(ni)[BUILD_DIRECTION] == subdomain.minCorner[BUILD_DIRECTION]) {
                    size_t dni = m_dirichletConstraintForNode.at(ni);
                    if (!m_dirichletComponents[dni].hasAll(N)) throw unexpectedBC;
                    if (m_dirichletNodeDisplacements.row(dni).squaredNorm() != 0) throw unexpectedBC;
                }
                else {
                    auto it = m_dirichletConstraintForNode.find(ni);
                    if (it != m_dirichletConstraintForNode.end()) {
                        size_t dni = it->second;
                        if (m_dirichletComponents[dni].hasAny(N)) throw unexpectedBC;
                    }
                }
            }
        }

        BCBuilder bcBuilder(*result);
        ComponentMask cmaskAll;
        cmaskAll.set();
        for (size_t ni = 0; ni < result->numNodes(); ++ni) {
            if (result->nodePosition(ni)[BUILD_DIRECTION] == subdomain.minCorner[BUILD_DIRECTION])
                bcBuilder.setDirichlet(ni, VNd::Zero(), cmaskAll);
        }
        bcBuilder.apply(*result);

        return result;
    }

    // Construct a coarser simulator *without applying any boundary conditions*.
    std::shared_ptr<TPS> downsample(size_t downsamplingLevels) const {
        size_t downsamplingFactor = std::pow(2, downsamplingLevels);

        auto numElemsPerDim = NbElementsPerDimension();
        for (size_t d = 0; d < N; ++d) {
            if (numElemsPerDim[d] % downsamplingFactor != 0)
                throw std::runtime_error("Grid size must be divisible by 2^downsamplingLevels");
            numElemsPerDim[d] /= downsamplingFactor;
        }

        // std::make_shared doesn't use the correct aligned allocator...
        //  https://eigen.tuxfamily.org/bz/show_bug.cgi?id=1049
        auto result = std::shared_ptr<TPS>(new TPS(domain(), numElemsPerDim));

        result->setETensor(getETensor());
        result->m_E_min = m_E_min;
        result->m_E_0   = m_E_0;
        result->m_gamma = m_gamma;

        return result;
    }

    size_t getDownsamplingFactor(const TPS &downsampled) const {
        size_t downsamplingFactor = NbElementsPerDimension()[0] / downsampled.NbElementsPerDimension()[0];
        for (size_t d = 0; d < N; ++d) {
            if (downsamplingFactor * downsampled.NbElementsPerDimension()[d] != NbElementsPerDimension()[d])
                throw std::runtime_error("Invalid downsampled simulator");
        }
        return downsamplingFactor;
    }

    // Very simple density field coarsening: use the average of all fine
    // densities within a cell as the coarse density.
    void downsampleDensityFieldTo(const VXd &densities, TPS &coarseTPS) const {
        size_t downsamplingFactor = getDownsamplingFactor(coarseTPS);
        const size_t ne = numElements();
        if (size_t(densities.size()) != ne) throw std::runtime_error("Invalid input densities size (" + std::to_string(densities.size()) + " vs " + std::to_string(ne) + ")");

        VXd coarseDensities;
        coarseDensities.setZero(coarseTPS.numElements());
        Real_ weight = 1.0 / std::pow(downsamplingFactor, N);
        for (size_t ei = 0; ei < ne; ++ei) {
            EigenNDIndex eiND = m_densities.template unflattenIndex<N>(ei);
            EigenNDIndex eiND_c = eiND / downsamplingFactor;
            coarseDensities[coarseTPS.elementIndexForGridCell(eiND_c)] += densities[ei] * weight;
        }

        coarseTPS.setDensities(coarseDensities);
    }

    // "backprop"/upscale a gradient with respect to the coarsed densities into a gradient with respect
    // with respect to the fine densities.
    VXd upsampleDensityGradientFrom(const TPS &coarseTPS, const VXd &g_coarse) {
        size_t downsamplingFactor = getDownsamplingFactor(coarseTPS);

        const size_t ne_c = coarseTPS.numElements(),
                     ne   = this->    numElements();
        if (size_t(g_coarse.size()) != ne_c) throw std::runtime_error("Invalid coarse gradient size");
        Real_ weight = 1.0 / std::pow(downsamplingFactor, N);

        VXd result(ne);
        for (size_t ei = 0; ei < ne; ++ei) {
            EigenNDIndex eiND_c = m_densities.template unflattenIndex<N>(ei) / downsamplingFactor;
            result[ei] = g_coarse[coarseTPS.elementIndexForGridCell(eiND_c)] * weight;
        }
        return result;
    }

    size_t validatedIntermediateFabricationShapeSize(const TPS &intermediateTPS) const {
        size_t numIntermediateHElems = intermediateTPS.NbElementsPerDimension()[BUILD_DIRECTION];
        auto sizeOK = (intermediateTPS.NbElementsPerDimension().array() == NbElementsPerDimension().array()).eval();
        sizeOK[BUILD_DIRECTION] = numIntermediateHElems <= NbElementsPerDimension()[BUILD_DIRECTION];
        if (!sizeOK.all()) throw std::runtime_error("Intermediate shape is of unexpected size");
        return numIntermediateHElems;
    }

    void transferDensitiesToIntermediateFabricationShape(TPS &intermediateTPS) const {
        for (size_t ei = 0; ei < intermediateTPS.numElements(); ++ei) {
            EigenNDIndex eiND = intermediateTPS.ndIndexForElement(ei);
            intermediateTPS.m_densities[ei] = m_densities[elementIndexForGridCellUnchecked(eiND)];
        }
        intermediateTPS.setDensities(intermediateTPS.getDensities());
    }

    VField transferVFieldToIntermediateFabricationShape(const TPS &intermediateTPS, const VField &u) const {
        VField result(intermediateTPS.numNodes(), int(N));
        validatedIntermediateFabricationShapeSize(intermediateTPS);
        size_t numIntermediateYNodes = intermediateTPS.NbNodesPerDimension()[BUILD_DIRECTION];
        for (size_t ni = 0; ni < numNodes(); ++ni) {
            EigenNDIndex niND = ndIndexForNode(ni);
            if (niND[BUILD_DIRECTION] >= numIntermediateYNodes) continue;
            result.row(intermediateTPS.flatIndexForNode(niND)) = u.row(ni);
        }
        return result;
    }

    // Accumulate a per-element scalar field (e.g., a density or compliance gradient field) defined
    // on an intermediate shape simulator `intermediateTPS` to the corresponding
    // entries of a per-element scalar field `rho_accum` for *this* simulator.
    // WARNING: `rho_accum` **must** be passed as an Eigen::Ref<> so that `pybind11` truly
    // passes by reference (declaring it as `Eigen::VectorXd &` is not sufficient).
    void accumElementScalarFieldFromIntermediateFabricationShape(const TPS &intermediateTPS,
                                                                 Eigen::Ref<const VXd> rho_in,
                                                                 Eigen::Ref<      VXd> rho_accum) const {
        for (size_t ei = 0; ei < intermediateTPS.numElements(); ++ei) {
            EigenNDIndex eiND = intermediateTPS.ndIndexForElement(ei);
            rho_accum[elementIndexForGridCellUnchecked(eiND)] += rho_in[ei];
        }
    }

    // Apply symmetry boundary conditions on the domain's "minimum face" or
    // "maximumFace" for the specified axes. For instance, if symmetry_axes[0]
    // is `true` and minMaxFace[1] is `0`,
    // then the u_X component on the min-X face is constrained to zero (so that
    // after deformation the structure still stitches together with its
    // reflected copy.
    void applySymmetryConditions(Eigen::Array<bool, N, 1> symmetry_axes, Eigen::Array<bool, N, 1> minMaxFace = Eigen::Array<bool, N, 1>::Zero()) {
        BCBuilder bcBuilder(*this);
        for (size_t ni = 0; ni < numNodes(); ++ni) {
            auto p = nodePosition(ni);
            for (size_t d = 0; d < N; ++d) {
                if (!symmetry_axes[d]) continue;
                if (std::abs(p[d] - (minMaxFace[d] ? m_domain.maxCorner[d] : m_domain.minCorner[d])) < 1e-10)
                    bcBuilder.setDirichletComponent(ni, d, 0.0);
            }
        }
        bcBuilder.apply(*this);
    }

    Real unmaskedYoungModulusScaleFactor(size_t ei) const {
        if (m_interpolationLaw == InterpolationLaw::SIMP)
           return m_E_min + std::pow(m_densities[ei], m_gamma) * (m_E_0 - m_E_min);
        else
            return m_E_min + m_densities[ei] * (m_E_0 - m_E_min) / (1 + m_q * (1 - m_densities[ei]));
    }

private:
    void m_getPeriodicConditionFixedVariables(std::vector<size_t> &fixedVars, std::vector<Real_> &fixedVarValues) const {
        if (!periodic_BC)
            throw std::runtime_error("Trying to set fixed variables for periodic conditions when periodic "
                                     + std::string("conditions are not set."));
        fixedVars.clear();
        fixedVarValues.clear();

        for (size_t dim = 0; dim < N; ++dim) {
            // fix the translation by setting the first node to the origin
            // rotations are fixed by periodic conditions on dofs.
            fixedVars.push_back(dim);
            fixedVarValues.push_back(0);
        }
    }

    void m_updateK0() {
        m_element.Stiffness(m_K0, m_E_tensor, 1);
        // Copy upper triangle into the lower triangle
        m_K0.template triangularView<Eigen::StrictlyLower>() =
                m_K0.template triangularView<Eigen::StrictlyUpper>().transpose();

        for (size_t lni = 0; lni < numNodesPerElem; ++lni)
            m_K0Diag[lni] = m_K0.template block<N, N>(lni * N, lni * N);
    }

    void m_updateYoungModuli() {
        const size_t ne = numElements();
        m_youngModulusScaleFactor.resize(ne);
        if (m_fabricationMaskHeight < m_domain.maxCorner[BUILD_DIRECTION]) {
            parallel_for_range(numElements(), [&](size_t ei) {
                m_youngModulusScaleFactor[ei] = isElementMasked(ei) ? 0.0 : unmaskedYoungModulusScaleFactor(ei);
            });
        }
        else {
            parallel_for_range(numElements(), [&](size_t ei) {
                m_youngModulusScaleFactor[ei] = unmaskedYoungModulusScaleFactor(ei);
            });
        }
        invalidateNumericFactorization();
    }

    PerElementStiffness m_K0;
    std::array<MNd, numNodesPerElem> m_K0Diag;
    aligned_std_vector<PerElementStiffness> m_KeCache;

    Element m_element; // Representative grid element (to save memory, we assume a regular grid where all elements are related by a rigid transformation)
    size_t m_numElements;

    BBox<VNd> m_domain;              // The design/simulation domain (grid bounding box)
    size_t m_numNodes;

    ETensor m_E_tensor = ETensor(1, 0); // The material tensor is the same for all elements
    VNd m_element_stretch;              // Stretching is also the same for all elements
    NDVector<Real_> m_densities;         // But the density changes

    VNd m_nodeSpacing;

    bool periodic_BC = false;           // Whether periodic condition have been applied
    bool m_useNRTPinConstraint = false;
    VNd m_gravity = VNd::Zero();        // The gravity vector to use for self-weight. Self-weight is disabled by setting this vector to zero.
                                        // This gravity vector is actually `rho_base * g` where `rho_base` is the base material's mass density.
                                        // In other words, it has units `(kg / m^N) * (m / s^2) = N / m^N.

    EigenNDIndex m_NbNodesPerDimensionPerElement; // Number of nodes belonging to one element, per dimension
    EigenNDIndex m_NbNodesPerDimension;           // Number of nodes per dimension
    EigenNDIndex m_NbElementsPerDimension;        // Number of elements per dimension
    EigenNDIndex m_NodeGlobalIndexIncrement;      // Increment in flattened node index induced by changing a ND node index
    EigenNDIndex m_NodeGlobalIndexIncrementPerElementIncrement; // Increment in flattened node index induced by changing an ND **element** index
    EigenNDIndex m_ElementIndexIncrement;         // Increment in flattened element index induced by changing an ND element index
    ENodesArray  m_referenceElementNodes;         // Node indices for the 0th element (from which the others can be determined).

    // Cached connectivity information
    const typename Stencils::ElementsAdjacentNode m_elementsAdjacentNode = Stencils::elementsAdjacentNode();

    InterpolationLaw m_interpolationLaw = InterpolationLaw::SIMP;
    Real_ m_q = 3;                   // RAMP factor: valid between 0 and (E0 - Emin)/Emin
    Real_ m_E_min = 1e-4;            // minimum Young modulus multiplier (to be used in SIMP formula)
    Real_ m_E_0 = 1;                 // Young modulus multiplier of solid voxels (to be used in SIMP formula)
    Real_ m_gamma = 3;               // SIMP exponent
    VXd   m_youngModulusScaleFactor; // Cached Young's modulus attenuation factor

    mutable std::string m_description; // simulator description, cached for re-use

    BlockSuiteSparseMatrix m_blockK;

    // Cached solver and sparsity pattern are mutable because they do not affect
    // user-visible state.
    // The sparsity pattern is for the full stiffness matrix (before boundary
    // conditions have been applied) and therefore should remain fixed throughout
    // the simulator's lifetime unless the periodicity conditions change.
    // However, the constructor should not build it since it is not needed for
    // the finer levels of the multigrid solver.
    // The factorizer must be rebuilt from scratch whenever the Dirichlet/
    // periodic conditions change. When just the densities or forces change we can
    // reuse the symbolic factorization, but must update the numeric factorization;
    // this is indicated by `m_numericFactorizationUpToDate == false`.
    mutable SuiteSparseMatrix m_hessianSparsityPattern;
    mutable bool m_numericFactorizationUpToDate = false;
    mutable std::unique_ptr<CholmodFactorizer> m_solver;

    std::vector<bool>   m_hasFullDirichlet, m_hasDirichlet;
    std::vector<size_t> m_dirichletNodes;
    std::vector<size_t> m_forceNodes;
    VField m_forceNodeForces;
    VField m_dirichletNodeDisplacements;
    std::vector<ComponentMask> m_dirichletComponents;
    std::unordered_map<size_t, size_t> m_dirichletConstraintForNode;

    // Fabrication mask: functionality for masking out/disabling all nodes/elements above a certain height.
    Real_ m_fabricationMaskHeight       = std::numeric_limits<Real_>::infinity();
    int m_firstDetachedNodeLayerIdx  = LAYER_MASK_NONE;
    int m_firstMaskedElementLayerIdx = LAYER_MASK_NONE;

    friend struct SpecializedTPSStencils<Real_, Degrees...>;

public:
    static constexpr int LAYER_MASK_NONE = std::numeric_limits<int>::max();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Work around alignment issues when using SIMD
};

#endif //MESHFEM_TENSORPRODUCTSIMULATOR_HH
