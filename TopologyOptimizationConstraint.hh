#ifndef MESHFEM_TOPOLOGYOPTIMIZATIONCONSTRAINT_HH
#define MESHFEM_TOPOLOGYOPTIMIZATIONCONSTRAINT_HH

#include "NDVector.hh"
#include <MeshFEM/Utilities/NameMangling.hh>

template<typename Real_>
struct Constraint {
    static std::string mangledName() { return "Constraint" + floatingPointTypeSuffix<Real_>(); }

    virtual ~Constraint() {};

    /// Evaluation of volume constraint.
    /// @param[in] vars physical variables
    /// @param[out] out constraint value
    virtual Real_ evaluate(const NDVector<Real_> &vars) const = 0;

    /// Derivative of constraint w.r.t. physical variables.
    /// @param[in] vars physical variables
    /// @param[out] out constraint derivatives
    virtual void backprop(const NDVector<Real_> &vars, NDVector<Real_> &out) const = 0;
};

template<typename Real_>
struct TotalVolumeConstraint : public Constraint<Real_> {
    static std::string mangledName() { return "TotalVolumeConstraint" + floatingPointTypeSuffix<Real_>(); }

    TotalVolumeConstraint(Real_ volume): m_volumeFraction(volume) { }

    Real_ evaluate(const NDVector<Real_> &vars) const override {
        return 1.0 - (vars.flattened().mean()) / m_volumeFraction;
    }

    void backprop(const NDVector<Real_> &vars, NDVector<Real_> &out) const override {
        out.fill(-1.0 / (m_volumeFraction * vars.size()));
    }

    // A value in [0, 1] indicating the fraction of solid voxels
    Real_ m_volumeFraction;
};

#endif // MESHFEM_TOPOLOGYOPTIMIZATIONCONSTRAINT_HH
