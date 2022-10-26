#ifndef MESHFEM_TOPOLOGYOPTIMIZATIONPROBLEM_HH
#define MESHFEM_TOPOLOGYOPTIMIZATIONPROBLEM_HH

#include <iostream>
#include <cmath>
#include <functional>
#include <memory>

#include "TensorProductSimulator.hh"
#include "TopologyOptimizationObjective.hh"
#include "TopologyOptimizationFilter.hh"
#include "TopologyOptimizationConstraint.hh"

// A general problem class for topology optimization.
// Provides methods for evaluating objective function, its derivatives,
// constraints and derivatives of the constraints.
template<typename _Sim>
class TopologyOptimizationProblem {
public:
    using Scalar          = typename _Sim::Scalar;
    using ObjectivePtr    = typename std::shared_ptr<Objective<_Sim>>;
    using FiltersList     = typename std::vector<std::shared_ptr<Filter<Scalar>>>;
    using ConstraintsList = typename std::vector<std::shared_ptr<Constraint<Scalar>>>;
    using VXd             = typename _Sim::VXd;
    using VXdRow          = Eigen::Matrix<Scalar,              1, Eigen::Dynamic>;
    using MXd             = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    TopologyOptimizationProblem(_Sim &simulator, const ObjectivePtr &objective,
        const ConstraintsList &constraints, const FiltersList &filters = FiltersList()):
        m_sim(simulator), m_objective(objective),
        m_filters(filters, m_sim.NbElementsPerDimension()), m_constraints(constraints) { }

    virtual ~TopologyOptimizationProblem() { };

    size_t numVars()        const { return m_filters.numVars();  }
    size_t numConstraints() const { return m_constraints.size(); }
    Eigen::Map<const VXd> getVars() const { return Eigen::Map<const VXd>(m_filters.designVars().data().data(), m_filters.numVars()); }

    // Compute all the needed fields given the value of the design variables
    // This method updates the state of the TOP object affecting the results of all the following function evaluations
    virtual bool setVars(Eigen::Ref<const VXd> x, bool /* forceUpdate */ = false) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("setVars");

        // Update Objective cache
        m_filters.setDesignVars(x);
        m_objective->updateCache(m_filters.physicalVars());

        m_varsAreSet = true;
        return true;
    }

    // For use in the binary search of the Optimality Criteria method
    // (seeking the Lagrange multiplier estimate that satisfies the volume constraint):
    // evaluate the volume constraint "c >= 0"; at the optimum we will have c == 0.
    // The following additional density arrays are needed to avoid memory allocations in each call:
    // @param[inout] x         input design/"blueprint" variables; will be overwritten with the filtered variables.
    // @param[inout] xscratch  scratch space for filtering the density variables; must be initialized to the proper size.
    Scalar evaluateOCConstraintAtVars(NDVector<Scalar> &x, NDVector<Scalar> &xscratch) const {
        if ((m_constraints.size() != 1) ||
            (nullptr == std::dynamic_pointer_cast<TotalVolumeConstraint<Scalar>>(m_constraints[0]))) {
            throw std::runtime_error("Applicable only for a topology optimization with a single (volume) constraint");
        }
        m_filters.applyInPlace(x, xscratch);

        return m_constraints[0]->evaluate(x);
    }

    // Compute value of objective function at current design variables
    virtual Scalar evaluateObjective() const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("evaluateObjective");
        if (!m_varsAreSet) throw std::runtime_error("Must call setVars first!");

        return m_objective->evaluate(m_filters.physicalVars());
    }

    // Compute gradient of objective function w.r.t. design variables
    void evaluateObjectiveGradient(VXd &result) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("evaluateObjectiveGradient");
        if (!m_varsAreSet) throw std::runtime_error("Must call setVars first!");

        NDVector<Scalar> gradient = m_objective->gradient();
        m_filters.backprop(gradient);
        result = gradient.flattened();
    }
    virtual VXd evaluateObjectiveGradientAndReturn() const {
        VXd result;
        evaluateObjectiveGradient(result);
        return result;
    }
    // Compute value of the constraints at current design variables
    // Return a vector of dimension m_constraints.size()
    virtual VXd evaluateConstraints() const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("evaluateConstraints");
        if (!m_varsAreSet) throw std::runtime_error("Must call setVars first!");

        size_t nConstr = m_constraints.size();
        VXd constrValues(nConstr);
        const NDVector<Scalar> &xPhys = m_filters.physicalVars();
        for(size_t i = 0; i < nConstr; i++)
            constrValues[i] = m_constraints[i]->evaluate(xPhys);  // constrValues[i] is imposed positive inside the optimizer
        return constrValues;
    }

    // Compute derivatives of all the constraints w.r.t. design variables
    // Return a matrix of dimensions [numConstraints(), numVars()]
    void evaluateConstraintsJacobian(MXd &jacobian) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("evaluateConstraintsJacobian");
        if (!m_varsAreSet) throw std::runtime_error("Must call setVars first!");

        // From dConstraint_n/dPhysical to dConstraint_n/dDesign
        jacobian.resize(numConstraints(), numVars());
        const NDVector<Scalar> &xPhys = m_filters.physicalVars();
        NDVector<Scalar> gradConstraint(m_sim.NbElementsPerDimension()),
                       scratch       (m_sim.NbElementsPerDimension());
        for (size_t n = 0; n < numConstraints(); n++) {
            m_constraints[n]->backprop(xPhys, gradConstraint);  // dConstraint_n/dPhysical
            m_filters.backprop(gradConstraint, scratch);
            // Cast jacobian to Eigen::Matrix format
            jacobian.row(n) = Eigen::Map<const VXdRow>(gradConstraint.data().data(), numVars());
        }
    }
    MXd evaluateConstraintsJacobianAndReturn() const {
        MXd result;
        evaluateConstraintsJacobian(result);
        return result;
    }

    VXd                       getDensities() const { return m_sim.getDensities(); }
    ObjectivePtr              getObjective() const { return m_objective;          }
    const FiltersList      &    getFilters() const { return m_filters.filters();  }
    const ConstraintsList  &getConstraints() const { return m_constraints;        }
    const FilterChain<Scalar> &filterChain() const { return m_filters; }

    const _Sim &getSimulator() const { return m_sim; }

    void setObjective(ObjectivePtr objective) { m_objective = objective; }
    void setConstraints(ConstraintsList constraints) { m_constraints = constraints; }

protected:
    // Finite element simulator
    _Sim &m_sim;

    // Check if the state has been set. True if setVars() was called at least once
    bool m_varsAreSet = false;

    // Objective function
    ObjectivePtr m_objective;

    // Chain of filters applied to the design variables
    FilterChain<Scalar> m_filters;

    // Constraints collection. Methods for evaluating constraint and
    // computing its derivative physical variables are provided
    ConstraintsList m_constraints;
};

#endif // MESHFEM_TOPOLOGYOPTIMIZATIONPROBLEM_HH
