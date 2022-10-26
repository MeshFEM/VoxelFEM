////////////////////////////////////////////////////////////////////////////////
// LayerByLayer.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Support for accelerating layer-by-layer simulation.
*///////////////////////////////////////////////////////////////////////////////
#ifndef LAYERBYLAYER_HH
#define LAYERBYLAYER_HH

#define ASSUME_ZERO_RESIDUAL_INIT 0

#include "TensorProductSimulator.hh"
#include "ParallelVectorOps.hh"
#include "MultigridSolver.hh"

template<class TPS>
struct LayerByLayerInit {
    enum class Method { ZERO, FD, SUBSPACE };
    size_t N = 3;

    void reset() { }
};

template<class TPS>
struct LayerByLayerEvaluator {
    using MG     = MGSolverForTPS_t<TPS>;
    using Real   = typename TPS::Scalar;
    using VField = typename TPS::VField;
    using VXd    = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using MXd    = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    static constexpr size_t N = TPS::N;

    LayerByLayerEvaluator(std::shared_ptr<TPS> lblSim) {
        m_sim = lblSim;
        selectInitMethod("N=3");
    }

    struct InitializationGenerator {
        InitializationGenerator(const TPS &tps) : m_tps(tps) { }
        virtual void reset() { }
        virtual void constructGuess(const VField &f, VField &g) = 0;

        // Warning: invalidates u
        virtual void finalizeLayer(size_t /* lbegin */, size_t /* lend */, VField &u, const VField &/* f */, const VField &/* r */, Real /* compliance */) {
            m_addToHistory(u);
        }

        const TPS &tps() const { return m_tps; }

        virtual ~InitializationGenerator() { }
    protected:
        virtual void m_addToHistory(VField &/* u */) { }
        const TPS &m_tps;
    };

    struct InitGenZero : public InitializationGenerator {
        using InitializationGenerator::InitializationGenerator;
        virtual void constructGuess(const VField &/* f */, VField &g) override {
            setZeroParallel(g, this->tps().numNodes(), N);
        }
    };

    struct InitGenWithHistory : public InitializationGenerator {
        InitGenWithHistory(const TPS &tps, size_t s) : InitializationGenerator(tps), m_maxHistSize(s) { }

        virtual void reset() override { m_storage.clear(); }

        size_t histSize() const { return m_storage.size(); }
        const VField &u(size_t l) const { return *m_storage[l]; }

    protected:
        virtual void m_addToHistory(VField &u_new) override {
            if (m_storage.size() == m_maxHistSize) {
                auto u_stale = std::move(m_storage.back());
                m_storage.pop_back();
                m_storage.push_front(std::move(u_stale));
            }
            else {
                if (m_storage.size() > m_maxHistSize) throw std::logic_error("History size exceeds limits.");
                m_storage.push_front(std::make_unique<VField>());
            }
            // Invalidates `u_new`
            std::swap(u_new, *m_storage.front());
        }
        size_t m_maxHistSize = 2;
        std::deque<std::unique_ptr<VField>> m_storage;
    };

    struct InitGenFD : public InitGenWithHistory {
        using InitGenWithHistory::histSize;
        using InitGenWithHistory::u;

        InitGenFD(const TPS &tps, size_t maxStencilSize) : InitGenWithHistory(tps, maxStencilSize) { }

        virtual void constructGuess(const VField &/* f */, VField &g) override {
            if (histSize() == 0) setZeroParallel(g, this->tps().numNodes(), N);
            if (histSize() == 1) copyParallel(u(0), g);
            if (histSize() == 2) g = 2 * u(0) - u(1);
            if (histSize() > 2) throw std::runtime_error("Unimplemented");
        }
    };

    struct InitGenSubspace : public InitGenWithHistory {
        using EigenNDIndex = typename TPS::EigenNDIndex;
        using InitGenWithHistory::histSize;
        using InitGenWithHistory::u;

        InitGenSubspace(const TPS &tps, size_t n) : InitGenWithHistory(tps, n) { }

        virtual void constructGuess(const VField &/* f */, VField &g) override {
            // Update A
            const TPS &sim = this->tps();

            size_t s = histSize();
            if (s == 0) {
                setZeroParallel(g, sim.numNodes(), N);
                return;
            }

            Eigen::JacobiSVD<Eigen::MatrixXd> solver(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::VectorXd c = solver.solve(b);

            // std::cout << "A: " << A << std::endl;
            // std::cout << "b: " << b.transpose() << std::endl;
            // std::cout << "c: " << c.transpose() << std::endl;

            g.resize(sim.numNodes(), N); // Note the `u` field referenced by `g` was invalidated by `finalize_layer`, so a new one may need to be allocated.
            // g.setConstant(std::numeric_limits<double>::quiet_NaN()); // Uninitialized values debugging
            sim.maskedNodalVisitBlocks([&](size_t nodeStart, size_t numNodes) {
                    g    .middleRows(nodeStart, numNodes)  = c[0] * u(0).middleRows(nodeStart, numNodes);
                    for (size_t i = 1; i < s; ++i)
                        g.middleRows(nodeStart, numNodes) += c[i] * u(i).middleRows(nodeStart, numNodes);
                });
#if 1
            // Note: NaNs in the first layer of detached nodes will leak into the topmost
            // nondetached layer during `applyK` since 0 * NaN = NaN.
            EigenNDIndex nodeSizes = sim.nondetachedNodesPerDim();
            if (nodeSizes[TPS::BUILD_DIRECTION] != sim.NbNodesPerDimension()[TPS::BUILD_DIRECTION]) {
                EigenNDIndex nbegin = EigenNDIndex::Zero();
                EigenNDIndex nend   = sim.NbNodesPerDimension();
                nbegin[TPS::BUILD_DIRECTION] = nodeSizes[TPS::BUILD_DIRECTION];
                // Due to vectorization, `applyK` will generate values also for
                // nodes [first_detached, ..., first_detached + SIMD_WIDTH - 2]
                // in the worst case. This means NaNs can leak into these values from the layers
                // `first_detached + SIMD_WIDTH - 1`; make sure those are cleared of NaN.
                // TODO: Figure out a better way to fix this.
                nend  [TPS::BUILD_DIRECTION] = std::min(nbegin[TPS::BUILD_DIRECTION] + VOXELFEM_SIMD_WIDTH, nend[TPS::BUILD_DIRECTION]);
                IndexRangeVisitor<N, /* Parallel = */ true>::run([&](const EigenNDIndex &coarseNode) {
                        size_t n_c = sim.flatIndexForNodeConstexpr(coarseNode.template cast<size_t>());
                        g.row(n_c).setZero();
                    }, nbegin, nend);
            }
#endif
        }

        virtual void finalizeLayer(size_t lbegin, size_t lend, VField &u_l, const VField &/* f */, const VField &r, Real compliance) override {
            FINE_BENCHMARK_SCOPED_TIMER_SECTION timer("finalizeLayer");
            // Warning: invalidates u_l
            this->m_addToHistory(u_l);

            size_t s = histSize();
            const auto &sim = this->tps();
            // Update b with the recurrence relation. The following is
            // mathematically equivalent to the brute force implementation:
            //     for (size_t i = 0; i < s; ++i)
            //         b[i] = dotProductParallel(u(i), f_lbegin);
            //  where `f_lbegin` is the right-hand side vector of the *next* simulation.
            VXd b_old = b;
            {
                BENCHMARK_SCOPED_TIMER_SECTION timer("update b");
                b.resize(s);
                b[0] = compliance;
                b.tail(s - 1) = b_old.head(s - 1);
                b += sim.dotLayerRemovalDeltaLoadVector(lbegin, lend, this->m_storage);
            }

            // Update A with the recurrence relation.
            // The following is mathematically equivalent to the brute force implementation:
            //      for (size_t i = 0; i < s; ++i) {
            //          for (size_t j = i; j < s; ++j) {
            //              A(j, i) = A(i, j) = dotProductParallel(K * u(i), u(j));
            //          }
            //      }
            MXd A_old = A;
            A.resize(s, s);
            A(0, 0) = compliance;
            A.col(0).tail(s - 1) = b_old.head(s - 1);
#if !ASSUME_ZERO_RESIDUAL_INIT
            BENCHMARK_START_TIMER_SECTION("U^T r");
            std::vector<const VField *> us;
            for (size_t i = 0; i < s; ++i)
                us.push_back(this->m_storage[i].get());

#if 1
            using EigenNDIndex = typename TPS::EigenNDIndex;
            using VNd          = typename TPS::VNd;
            auto results = IndexRangeVisitorThreadLocal<N, VXd>::run([&](const EigenNDIndex &niND, VXd &out) {
                size_t ni = sim.flatIndexForNodeConstexpr(niND);
                VNd r_row = r.row(ni);
                for (size_t i = 0; i < s; ++i)
                    out[i] += r_row.dot(us[i]->row(ni));
            }, /* constructor */ [s](VXd &v) { v.setZero(s); }, EigenNDIndex::Zero().eval(), sim.nondetachedNodesPerDim());


            for (const auto &rr : results)
                A.col(0) -= rr.v;
#else
#if 0       // This version is about the same as above.
            constexpr size_t MAX_N = 8;
            // Use a fixed-size type to avoid memory allocations in tbb::parallel_reduce
            using UtRType = Eigen::Matrix<Real, Eigen::Dynamic, 1, Eigen::ColMajor, MAX_N, 1>;
            if (s > MAX_N) throw std::runtime_error("N is limited to MAX_N=" + std::to_string(MAX_N));
            A.col(0) -= sim.maskedNodalReduceSum([&](size_t nodeStart, size_t numNodes) {
                        UtRType utr(s);
                        for (size_t i = 0; i < s; ++i)
                            utr[i] = r.middleRows(nodeStart, numNodes).cwiseProduct(us[i]->middleRows(nodeStart, numNodes)).sum();
                        return utr;
                    }, UtRType::Zero(s));
#else       // This version appears slowest...
            for (size_t i = 0; i < s; ++i)
                A(i, 0) -= sim.maskedNodalDotProduct(u(i), r);
#endif
#endif

            // for (size_t i = 0; i < s; ++i)
            //     A(i, 0) -= dotProductParallel(u(i), r);
            BENCHMARK_STOP_TIMER_SECTION("U^T r");
#endif
            A.bottomRightCorner(s - 1, s - 1) = A_old.topLeftCorner(s - 1, s - 1);
            sim.layerRemovalDeltaUKU(lbegin, lend, this->m_storage, m_scratch, A);
            A.template triangularView<Eigen::Upper>() = A.template triangularView<Eigen::Lower>().transpose();
        }

        // Subspace linear system
        MXd A;
        VXd b;
    private:
        VField m_scratch;
    };

          TPS &sim()       { return *m_sim; }
    const TPS &sim() const { return *m_sim; }

    void selectInitMethod(const std::string &method) {
        if      (method ==  "zero")    m_initGen = std::make_unique<InitGenZero>(sim());
        else if (method == "constant") m_initGen = std::make_unique<InitGenFD>(sim(), 1);
        else if (method ==    "fd")    m_initGen = std::make_unique<InitGenFD>(sim(), 2);
        else if (method.substr(0, 2) == "N=") m_initGen = std::make_unique<InitGenSubspace>(sim(), std::stoi(method.substr(2)));
        else throw std::runtime_error("Unrecognized method " + method);
    }

    using LBLCallback = std::function<void(size_t, Real, const VXd &, const VField &)>; // cb(l, compliance, grad_compliance, u)
    void run(MG &solver, bool zeroInit, size_t layerIncrement,
             // Solver options
             size_t maxIter,
             const Real tol, typename MG::PCGCallback it_callback = nullptr,
             size_t mgIterations = 1,
             size_t mgSmoothingIterations = 1,
             bool fmg = false, bool verbose = false, LBLCallback lblCallback = nullptr) {
        TPS &tps = sim();

        const size_t numLayers = tps.NbElementsPerDimension()[TPS::BUILD_DIRECTION];

        VField f, u;
        if (!zeroInit) copyParallel(m_uFullDesign, u);

        m_initGen->reset();

        m_layersAccumulated = 0;
        m_totalCompliance = 0;
        setZeroParallel(m_totalComplianceGradient, tps.numElements(), 1);

        BENCHMARK_START_TIMER_SECTION("Build load");
        solver.setFabricationMaskHeightByLayer(numLayers);
        f = tps.buildLoadVector();
        BENCHMARK_STOP_TIMER_SECTION("Build load");

        for (size_t l = numLayers; l > 0; l -= std::min(layerIncrement, l)) {
            if (l < numLayers) {
                solver.decrementFabricationMaskHeightByLayer(layerIncrement);
                BENCHMARK_START_TIMER_SECTION("Update load");
                tps.addLayerRemovalDeltaLoadVector(l, l + layerIncrement, f);
                BENCHMARK_STOP_TIMER_SECTION("Update load");
            }

            // Construct an initial guess unless this is the full design and
            // one was already copied from the previous full-design solution.
            BENCHMARK_START_TIMER_SECTION("Construct initial guess");
            if ((l < numLayers) || (size_t(u.rows()) != tps.numNodes()))
                m_initGen->constructGuess(f, u);
            BENCHMARK_STOP_TIMER_SECTION("Construct initial guess");

            // Note: the initial guess should always already satisfy the `u = 0` Dirichlet conditions!
            try {
                solver.preconditionedConjugateGradient(u, f, maxIter, tol, it_callback, mgIterations, mgSmoothingIterations, fmg, /* dirichletAlreadySatisified = */ true);
            }
            catch (std::exception &e) {
                std::cout << "PCG exception " << e.what() << " at l = " << l << std::endl;
                throw e;
            }

            BENCHMARK_START_TIMER_SECTION("Compute compliance");
            Real compliance = tps.maskedNodalDotProduct(f, u);
            BENCHMARK_STOP_TIMER_SECTION("Compute compliance");
            if (verbose)
                std::cout << "Layer " << l << ": " << compliance << std::endl;
            if (lblCallback) {
                lblCallback(l, compliance, tps.complianceGradientFlattened(u), u);
            }

            m_totalCompliance += compliance;
            BENCHMARK_START_TIMER_SECTION("Compute gradient");
            tps.accumulateComplianceGradient(u, m_totalComplianceGradient);
            BENCHMARK_STOP_TIMER_SECTION("Compute gradient");
            ++m_layersAccumulated;

            if (l == numLayers)
                copyParallel(u, m_uFullDesign);

            if (l >= layerIncrement) {
                // The simulation we just ran is associated with the yet-unmasked layers [l - layerIncrement, l).
                // Warning: invalidates u!
                m_initGen->finalizeLayer(l - layerIncrement, l, u, f, solver.pcgResidual(), compliance);
            }
        }
    }

    // WARNING: does not include the multiplier needed to account for reflectional symmetries!
    Real objective() const { return 0.5 * m_totalCompliance / m_layersAccumulated; }
    VXd   gradient() const { return m_totalComplianceGradient / m_layersAccumulated; }

private:
    Real m_totalCompliance;
    VXd m_totalComplianceGradient;
    size_t m_layersAccumulated = 0;
    VField m_uFullDesign;
    std::unique_ptr<InitializationGenerator> m_initGen;
    std::shared_ptr<TPS> m_sim;
};

#endif /* end of include guard: LAYERBYLAYER_HH */
