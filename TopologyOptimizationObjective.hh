#ifndef MESHFEM_TOPOLOGYOPTIMIZATIONOBJECTIVE_HH
#define MESHFEM_TOPOLOGYOPTIMIZATIONOBJECTIVE_HH

#include "NDVector.hh"

template<typename _Sim>
struct Objective {
    using Scalar = typename _Sim::Scalar;

    virtual ~Objective() {};

    /// Evaluation of objective function.
    /// @param[in] xPhys physical variables
    /// @param[out] out objective value
    virtual Scalar evaluate(const NDVector<Scalar> &xPhys) const = 0;

    /// Derivative of objective w.r.t. physical variables.
    /// @param[in] xPhys physical variables
    /// @param[out] out objective derivatives
    virtual NDVector<Scalar> gradient() const = 0;

    virtual void updateCache(const NDVector<Scalar> &xPhys) = 0;
};

template<typename _Sim>
struct ComplianceObjective : public Objective<_Sim> {
    using VField              = typename _Sim::VField;
    using PerElementStiffness = typename _Sim::PerElementStiffness;
    using Scalar              = typename _Sim::Scalar;
    constexpr static size_t N = _Sim::N;

    ComplianceObjective(_Sim &simulator, bool skipSolve = false) : m_sim(simulator) {
        m_f = m_sim.buildLoadVector();
        if (!skipSolve) updateCache(m_sim.elementDensities());
    }

    Scalar evaluate(const NDVector<Scalar> &/* xPhys */) const override {
        return compliance();
    }

    Scalar compliance() const {
        return 0.5 * (m_f.array() * m_u.array()).sum();
    }

    NDVector<Scalar> gradient() const override {
        return m_sim.complianceGradient(m_u);
    }

    // Update cached displacements (solving Ku=f) with K defined by the new densities
    void updateCache(const NDVector<Scalar> &xPhys) override {
        m_sim.setElementDensities(xPhys);
        m_u = m_sim.solve(m_f);
    }

    const VField &u() const { return m_u; }
    const VField &f() const { return m_f; }

protected:
    // Finite element simulator
    _Sim &m_sim;

    // Cached nodal displacements computed solving Ku=f
    VField m_u;

    // Cached load vector (rhs of the linear system)
    VField m_f;
};

#include "MultigridSolver.hh"

template<typename _Sim>
struct MultigridComplianceObjective : public ComplianceObjective<_Sim> {
    using Base   = ComplianceObjective<_Sim>;
    using MG     = typename MGSolverForTPS<_Sim>::type;
    using Scalar = typename Base::Scalar;
    using Base::m_sim;
    using Base::m_f;
    using Base::m_u;
	using VField = typename MG::VField;

    MultigridComplianceObjective(std::shared_ptr<MG> mg_)
        : Base(mg_->getSimulator(0), /* skipSolve */ true), mg(mg_) {
        m_u.setZero(m_sim.numDoFs(), size_t(Base::N));
        updateCache(m_sim.elementDensities());
    }

    // Update cached displacements (solving Ku=f) with K defined by the new densities
    void updateCache(const NDVector<Scalar> &xPhys) override {
        m_sim.setElementDensities(xPhys);
        std::function<void(size_t, const VField &, const VField &)> solver_cb = nullptr;
        if (residual_cb)
            solver_cb = [&](size_t it, const VField &/* x */, const VField &r) { residual_cb(it, r.norm()); };
        if (zeroInit) m_u.setZero(m_u.rows(), m_u.cols());
        mg->preconditionedConjugateGradient(m_u, m_f, cgIter, tol,
                                            solver_cb, mgIterations, mgSmoothingIterations, fullMultigrid);
    }

    std::shared_ptr<MG> mg;
    size_t cgIter = 100;
    Scalar tol = 1e-5;
    size_t mgIterations = 1, mgSmoothingIterations = 2;
    bool fullMultigrid = true;
    bool zeroInit = false;
	std::function<void(size_t, double r)> residual_cb = nullptr;
};

#endif // MESHFEM_TOPOLOGYOPTIMIZATIONOBJECTIVE_HH
