#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>
namespace py = pybind11;

#include <MeshFEM/Utilities/MeshConversion.hh>
#include "../TensorProductSimulator.hh"
#include "../TopologyOptimizationProblem.hh"
#include "../TopologyOptimizationFilter.hh"
#include "../NDVector.hh"
#include "../OptimalityCriterion.hh"
#include "../LayerByLayer.hh"

#include <map>
#include <functional>
#include <utility>

template<size_t Degree>
std::string degreeString() { return std::to_string(Degree); }

template<size_t Degree1, size_t Degree2, size_t... Degrees>
std::string degreeString() { return std::to_string(Degree1) + "_" + degreeString<Degree2, Degrees...>(); }

// We need to create a distinct python type per template instantiation of
// TensorProductSimulator; we use a "name mangling" scheme to give each of
// these types distinct names. These types will be hidden in
// `pyVoxelFEM.detail` to make the module's interface a bit cleaner.
template<typename Real_, size_t... Degrees>
std::string nameMangler(const std::string &name) {
    return name + degreeString<Degrees...>() + floatingPointTypeSuffix<Real_>();
}

using DynamicEigenNDIndex = Eigen::Array<size_t, Eigen::Dynamic, 1>;

enum class NumberType {
    DOUBLE,
    FLOAT
};

template<typename T>
struct NumberTypeGetter;

template<> struct NumberTypeGetter<double> { static constexpr NumberType type = NumberType::DOUBLE; };
template<> struct NumberTypeGetter< float> { static constexpr NumberType type = NumberType::FLOAT;  };

// "Factory" for creating an instantiation of the appropriate tensor product
// simulator type.
using FactoryType = std::function<py::object(const std::array<Eigen::VectorXd, 2> &, const DynamicEigenNDIndex &)>;
static std::map<std::tuple<NumberType, std::vector<size_t>>, FactoryType> g_factories;

// Extended functionalities to support layer-by-layer simulations
// Follow on https://pybind11.readthedocs.io/en/stable/advanced/classes.html#combining-virtual-functions-and-inheritance
// and https://github.com/pybind/pybind11/blob/master/tests/test_virtual_functions.cpp
template<typename _Sim>
class PyTopologyOptimizationProblem : public TopologyOptimizationProblem<_Sim> {
    using Scalar = typename _Sim::Scalar;
    using VXd    = typename _Sim::VXd;
    using TopologyOptimizationProblem<_Sim>::TopologyOptimizationProblem;
    bool setVars(Eigen::Ref<const VXd> x, bool /* forceUpdate */ = false) override { PYBIND11_OVERRIDE(bool, TopologyOptimizationProblem<_Sim>, setVars, x); }
    Scalar evaluateObjective() const override { PYBIND11_OVERRIDE(Scalar, TopologyOptimizationProblem<_Sim>, evaluateObjective,); }
    VXd evaluateObjectiveGradientAndReturn() const override { PYBIND11_OVERRIDE_NAME(VXd, TopologyOptimizationProblem<_Sim>, "evaluateObjectiveGradient", evaluateObjectiveGradientAndReturn,); }
};

template<typename Real_, size_t... Degrees>
void addTPSBindings(py::module &m, py::module &detail_module) {
    using TPS          = TensorProductSimulator<Real_, Degrees...>;
    using MG           = MultigridSolver<Real_, Degrees...>;
    using VField       = typename TPS::VField;
    using VXd          = Eigen::Matrix<Real_, Eigen::Dynamic, 1>;
    constexpr size_t N = sizeof...(Degrees);

    py::class_<TPS, std::shared_ptr<TPS>>(detail_module, (nameMangler<Real_, Degrees...>("TensorProductSimulator")).c_str())
        .def("numNodes",                           &TPS::numNodes)
        .def("numElements",                        &TPS::numElements)
        .def("getDensities",                       &TPS::getDensities)
        .def("setDensities",                       &TPS::setDensities, py::arg("rho"))
        .def("readMaterial",                       &TPS::readMaterial,        py::arg("materialPath"))
        .def("setDensity",                         &TPS::setDensity,   py::arg("ei"),           py::arg("value"))
        .def("setUniformDensities",                &TPS::setUniformDensities, py::arg("density"))
        .def("setDensitiesFromCoarseGrid",         &TPS::setDensitiesFromCoarseGrid, py::arg("upscalingFactor"), py::arg("rho"))
        .def("elementDensity",                     &TPS::elementDensity,      py::arg("ei"))
        .def("elementYoungModulusScaleFactor",     &TPS::elementYoungModulusScaleFactor, py::arg("ei"))
        .def("getYoungModulusScaleFactor",         &TPS::getYoungModulusScaleFactor)
        .def("setFabricationMaskHeightByLayer",    &TPS::setFabricationMaskHeightByLayer, py::arg("l"))
        .def("getFabricationMaskHeight",           &TPS::getFabricationMaskHeight)
        .def("solve",                              [](TPS &sim, const VField &f) { return sim.solve(f); }, py::arg("f"))
        .def("solveWithImposedLoads",              &TPS::solveWithImposedLoads)
        .def("complianceGradient",                 &TPS::complianceGradientFlattened)
        .def("multigridSolver",                    [](std::shared_ptr<TPS> tps, size_t numCoarseningLevels) { return std::shared_ptr<MG>(new MG(tps, numCoarseningLevels)); }) // `make_shared` would not call the correct aligned `new` overload.
        .def("getK",                               &TPS::getK)
        .def("applyK",                             [](const TPS &tps, const VField &u) { return tps.applyK(u); }, py::arg("u"))
        .def("getDirichletVarsAndValues",          [](const TPS &tps) { std::vector<size_t> vars; std::vector<Real_> vals; tps.getDirichletVarsAndValues(vars, vals); return std::make_tuple(vars, vals); })
        .def("buildLoadVector",                    &TPS::buildLoadVector)
        .def("constantStrainLoad",                 &TPS::constantStrainLoad,  py::arg("eps"))
        .def("nodePosition",                       [](const TPS &tps, size_t ni) { return tps.nodePosition(ni); }, py::arg("ni"))
        .def("applyDisplacementsAndLoadsFromFile", &TPS::applyDisplacementsAndLoadsFromFile, py::arg("bcPath"))
        .def("elementIndexForGridCell",            &TPS::elementIndexForGridCell, py::arg("cellIdxs"))
        .def("getForceMask",                       &TPS::getForceMask)
        .def("getDirichletMask",                   &TPS::getDirichletMask)
        .def("getBCIndicatorField",                &TPS::getBCIndicatorField)
        .def("elementStiffnessMatrix",             &TPS::elementStiffnessMatrix, py::arg("ei"))
        .def("elementNodes",                       [](const TPS &s, size_t ei) { return s.elementNodes(ei); }, py::arg("ei"))
        .def("elemNodeGlobalIndex",                [](const TPS &s, size_t ei, size_t n) { return s.elemNodeGlobalIndex(ei, n); }, py::arg("ei"), py::arg("n"))
        .def("addDirichletCondition",              &TPS::addDirichletCondition, py::arg("u"), py::arg("minCorner"), py::arg("maxCorner"), py::arg("componentMask") = "xyz")
        .def("mesh",                               [](std::shared_ptr<TPS> sptr) { return sptr; }, py::return_value_policy::reference, "Hack to support MeshFEM's `simu_tils` helpers")
        .def_property_readonly("domain",           [](const TPS &tps) { return std::make_pair(tps.domain().minCorner, tps.domain().maxCorner); })
        .def_property_readonly("bbox",             [](const TPS &tps) { return std::make_pair(tps.domain().minCorner, tps.domain().maxCorner); })
        .def_property_readonly("gridShape",              [](const TPS &tps) { return tps.NbElementsPerDimension(); })
        .def_property_readonly("NbElementsPerDimension", [](const TPS &tps) { return tps.NbElementsPerDimension(); })
        .def_property_readonly("NbNodesPerDimension",    [](const TPS &tps) { return tps.NbNodesPerDimension(); })
        .def_property("interpolationLaw",          &TPS::interpolationLaw, &TPS::setInterpolationLaw)
        .def_property("E_0",                       &TPS::E_0, &TPS::setE_0)
        .def_property("E_min",                     &TPS::E_min, &TPS::setE_min)
        .def_property("gamma",                     &TPS::SIMPExponent, &TPS::setSIMPExponent)
        .def_property("q",                         &TPS::RAMPFactor, &TPS::setRAMPFactor)
        .def_property("gravity",                   &TPS::getGravity, &TPS::setGravity)
        .def_property("ETensor",                   &TPS::getETensor, &TPS::setETensor, "Elasticty tensor")
        .def_property("dx",                        &TPS::getStretchings, &TPS::setStretchings, "Get the dimensions of a grid cell")
        .def_property_readonly("elementVolume",    [](const TPS &tps) { return tps.elementVolume(0); })
        .def("fullDensityElementStiffnessMatrix",  &TPS::fullDensityElementStiffnessMatrix)
        .def("clearCachedElementStiffness",        &TPS::clearCachedElementStiffness)
        .def("debugMulticolorElementVisit",        &TPS::debugMulticolorElementVisit)

        // Post-processing
        .def("sampleNodalField",                   &TPS::sampleNodalField, py::arg("u"), py::arg("p"), "Sample the nodal FEM field `u` at the query points specified by the rows of `p`")

        .def("elementEnergyDensity",               &TPS::elementEnergyDensity, py::arg("u"))

        // Intermediate shapes along the build direction (Y)
        .def("getIntermediateFabricationShape",                         &TPS::getIntermediateFabricationShape,                         py::arg("yfrac"),           py::arg("validateBoundaryConditions") = true, py::arg("law") = InterpolationLaw::SIMP)
        .def("transferDensitiesToIntermediateFabricationShape",         &TPS::transferDensitiesToIntermediateFabricationShape,         py::arg("intermediateTPS"))
        .def("transferVFieldToIntermediateFabricationShape",            &TPS::transferVFieldToIntermediateFabricationShape,            py::arg("intermediateTPS"), py::arg("u"))
        .def("accumElementScalarFieldFromIntermediateFabricationShape", &TPS::accumElementScalarFieldFromIntermediateFabricationShape, py::arg("intermediateTPS"), py::arg("rho_in"), py::arg("rho_accum"))

        .def("applySymmetryConditions", &TPS::applySymmetryConditions, py::arg("symmetry_axes"), py::arg("minMaxFace") = Eigen::Array<bool, N, 1>::Zero())

        .def("downsample",                  &TPS::downsample,                  py::arg("downsamplingLevels"))
        .def("downsampleDensityFieldTo",    &TPS::downsampleDensityFieldTo,    py::arg("densities"), py::arg("coarseTPS"))
        .def("upsampleDensityGradientFrom", &TPS::upsampleDensityGradientFrom, py::arg("coarseTPS"), py::arg("g_coarse"))

        // `getMesh` takes ouput arrays as references, and we cannot easily bind this
        // type of method. But we can make custom bindings
        // function directly.
        .def("getMesh", [](const TPS &tps) {
                std::vector<MeshIO::IOVertex > vertices;
                std::vector<MeshIO::IOElement> elements;
                tps.getMesh(vertices, elements);
                return std::make_tuple(getV(vertices), getF(elements));
            })
    ;
    py::class_<MG, std::shared_ptr<MG>>(detail_module, (nameMangler<Real_, Degrees...>("MultigridSolver")).c_str())
        .def("getSimulator",    py::overload_cast<const size_t>(&MG::getSimulator), py::arg("l"), py::return_value_policy::reference)
        .def("computeResidual", [](MG &mg, size_t l, const VField &u, const VField &b) { VField r; mg.computeResidual(l, u, b, r); return r; }, py::arg("l"), py::arg("u"), py::arg("b"))
        .def("applyK",    [](MG &mg, size_t l, const VField &u) { return mg.applyK(l, u); }, py::arg("l"), py::arg("u"))
        .def("zeroOutDirichletComponents", [](MG &mg, const size_t l, VField u) {
                mg.zeroOutDirichletComponents(l, u);
                return u;
            }, py::arg("l"), py::arg("u"))
        .def("updateStiffnessMatrices", &MG::updateStiffnessMatrices)
        .def("setSymmetricGaussSeidel", &MG::setSymmetricGaussSeidel, py::arg("symmetric"))
        .def("solve", [](MG &mg, const VField &u, const VField &f, size_t numSteps, size_t numSmoothingSteps, bool stiffnessUpdated,
                        bool zeroDirichlet, std::function<void(size_t, const VField &)> it_callback, bool fmg) {
                    return mg.solve(u, f, numSteps, numSmoothingSteps, stiffnessUpdated, zeroDirichlet, it_callback, fmg);
                },
                py::arg("u"), py::arg("f"), py::arg("numSteps"),
                py::arg("numSmoothingSteps"), py::arg("stiffnessUpdated") = false,
                py::arg("zeroDirichlet") = false, py::arg("it_callback") = nullptr,
                py::arg("fullMultigrid") = false)
        .def("setFabricationMaskHeightByLayer", &MG::setFabricationMaskHeightByLayer, py::arg("h"))
        .def("preconditionedConjugateGradient",
                [](MG &mg, const VField &u, const VField &f, size_t maxIter, double tol,
                   typename MG::PCGCallback &cb, size_t mgIterations, size_t mgSmoothingIterations,
                   bool fmg) {
                    // TODO: apply PCG in place. This requires taking `u`
                    // as an Eigen::Ref<VField>, and this type needs to propagate
                    // through all the code to avoid internal copies that occur when
                    // when `Eigen::Ref<VField>` is passed to a function accepting
                    // `const VField &`.
                    VField x = u;
                    mg.preconditionedConjugateGradient(x, f, maxIter, tol, cb, mgIterations, mgSmoothingIterations, fmg);
                    return x;
                },
                py::arg("u"), py::arg("b"), py::arg("maxIter"), py::arg("tol"),
                py::arg("it_callback") = nullptr,
                py::arg("mgIterations") = 1,
                py::arg("mgSmoothingIterations") = 1,
                py::arg("fullMultigrid") = false)
        .def("debug_get_x", &MG::debug_get_x, py::arg("l"))
        .def("debug_get_b", &MG::debug_get_b, py::arg("l"))
        .def("debugMulticolorVisit", &MG::debugMulticolorVisit)
    ;

    // Register the factory for this TensorProductSimulator instantiation in g_factories
    std::vector<size_t> degs{Degrees...};

    g_factories.emplace(std::make_pair(std::make_pair(NumberType(NumberTypeGetter<Real_>::type), degs), [](const std::array<Eigen::VectorXd, 2> &extremeNodes, const DynamicEigenNDIndex &elementsPerDimension) {
            // Note: while py::cast is not yet documented in the official documentation,
            // it accepts the return_value_policy as discussed in:
            //      https://github.com/pybind/pybind11/issues/1201
            // by setting the return value policy to take_ownership, we can avoid
            // memory leaks and double frees regardless of the holder type for FEMMesh.

            // Initialize domain bounding box and call appropriate constructor
            BBox<VecN_T<Real_, N>> domain;
            domain.minCorner = extremeNodes[0].cast<Real_>();
            domain.maxCorner = extremeNodes[1].cast<Real_>();
            return py::cast(new TPS(domain, elementsPerDimension), py::return_value_policy::take_ownership);
        }));

    using TOProblem       = TopologyOptimizationProblem<TPS>;
    using PyTOProblem     = PyTopologyOptimizationProblem<TPS>;
    using FiltersList     = typename TOProblem::FiltersList;
    using ConstraintsList = typename TOProblem::ConstraintsList;
    using ObjectivePtr    = typename TOProblem::ObjectivePtr;

    // "Constructor" functions overloads that will create an instance of the appropriate instantiation.
    m.def("TopologyOptimizationProblem", [](TPS &tps, ObjectivePtr objective, ConstraintsList constraints, FiltersList filters) {
        return std::make_unique<TOProblem>(tps, objective, constraints, filters);
        }, py::arg("simulator"), py::arg("objective"), py::arg("constraints"), py::arg("filters"))
    ;
    m.def("ComplianceObjective", [](TPS &tps) {
            return std::make_shared<ComplianceObjective<TPS>>(tps);
        }, py::arg("simulator"))
    ;
    m.def("MultigridComplianceObjective", [](std::shared_ptr<MG> mg) {
            return std::make_shared<MultigridComplianceObjective<TPS>>(mg);
        }, py::arg("mg_solver"))
    ;

    // Topology Optimization problem
    py::class_<TOProblem, PyTOProblem>(detail_module, (nameMangler<Real_, Degrees...>("TopologyOptimizationProblem")).c_str())
        .def(py::init<TPS&, ObjectivePtr, ConstraintsList, FiltersList>())
        .def("evaluateObjective",           &TOProblem::evaluateObjective)
        .def("evaluateObjectiveGradient",   &TOProblem::evaluateObjectiveGradientAndReturn)
        .def("evaluateConstraints",         &TOProblem::evaluateConstraints)
        .def("evaluateConstraintsJacobian", &TOProblem::evaluateConstraintsJacobianAndReturn)
        .def("numVars",                     &TOProblem::numVars)
        .def("getVars",                     [](const TOProblem &top) -> VXd { return top.getVars(); }) // force a copy to be returned.
        .def("setVars",                     &TOProblem::setVars, py::arg("x"), py::arg("forceUpdate") = false)
        .def("getDensities",                &TOProblem::getDensities)
        .def_property("objective",          &TOProblem::getObjective, &TOProblem::setObjective)
        .def_property_readonly("filters",     &TOProblem::getFilters)
        .def_property_readonly("filterChain", &TOProblem::filterChain, py::return_value_policy::reference)
        .def_property("constraints",        &TOProblem::getConstraints, &TOProblem::setConstraints)
        ;
    // Return class name that matches the type of tps
    m.def("getClassName", [&](TPS &, std::string name) {
        return std::string("pyVoxelFEM.") + std::string("detail.") + nameMangler<Real_, Degrees...>(name);
        }, py::arg("simulator"), py::arg("name"))
    ;
    // Objective functions
    py::class_<Objective<TPS>, std::shared_ptr<Objective<TPS>>>(detail_module, (nameMangler<Real_, Degrees...>("Objective")).c_str())
        ;
    using CO = ComplianceObjective<TPS>;
    py::class_<CO, Objective<TPS>, std::shared_ptr<CO>>(
        detail_module, (nameMangler<Real_, Degrees...>("ComplianceObjective")).c_str())
        .def("gradient",   &CO::gradient)
        .def("compliance", &CO::compliance)
        .def("u",          &CO::u)
        .def("f",          &CO::f)
        ;
    using MGCO = MultigridComplianceObjective<TPS>;
    py::class_<MGCO, ComplianceObjective<TPS>, std::shared_ptr<MGCO>>(detail_module, (nameMangler<Real_, Degrees...>("MultigridComplianceObjective")).c_str())
        .def_readonly ("mg",                    &MGCO::mg)
        .def_readwrite("cgIter",                &MGCO::cgIter)
        .def_readwrite("tol",                   &MGCO::tol)
        .def_readwrite("mgIterations",          &MGCO::mgIterations)
        .def_readwrite("mgSmoothingIterations", &MGCO::mgSmoothingIterations)
        .def_readwrite("fullMultigrid",         &MGCO::fullMultigrid)
        .def_readwrite("zeroInit",              &MGCO::zeroInit)
        .def_readwrite("residual_cb",           &MGCO::residual_cb)
        .def("updateCache",                     &MGCO::updateCache, py::arg("xPhys"))
        ;


    using LBL = LayerByLayerEvaluator<TPS>;
    py::class_<LBL>(detail_module, (nameMangler<Real_, Degrees...>("LayerByLayerEvaluator")).c_str())
        .def("selectInitMethod", &LBL::selectInitMethod, py::arg("method"), "Select method by name ['zero', 'fd', 'N=1', 'N=2', ...]")
        .def("run",              &LBL::run, py::arg("solver"), py::arg("zeroInit"), py::arg("layerIncrement"), py::arg("maxIter"), py::arg("tol"),
                                            py::arg("it_callback") = nullptr, py::arg("mgIterations") = 1, py::arg("mgSmoothingIterations") = 1,
                                            py::arg("fullMultigrid") = false, py::arg("verbose") = false, py::arg("lblCallback") = nullptr)
        .def("objective",        &LBL::objective)
        .def("gradient",         &LBL::gradient)
        ;
    m.def("LayerByLayerEvaluator", [](std::shared_ptr<TPS> tps) { return std::make_unique<LBL>(tps); }, py::arg("lblSim"));

    // Optimization algorithms
    using OCO = OCOptimizer<TOProblem>;
    py::class_<OCO>(detail_module, (nameMangler<Real_, Degrees...>("OCOptimizer")).c_str())
        .def(py::init<TOProblem &>(), py::arg("problem"))
        .def("step", &OCO::step, py::arg("m") = 0.2, py::arg("p") = 0.5, py::arg("ctol") = 1e-6, py::arg("inplace") = true)
        ;
    m.def("OCOptimizer", [](TOProblem &p) { return std::make_unique<OCO>(p); });

}

template<typename Real_>
void addBindings(py::module &m, py::module &detail_module) {
    addTPSBindings<Real_, 1, 1>   (m, detail_module);
    // addTPSBindings<Real_, 2, 2>   (m, detail_module);
#if !VOXELFEM_DISABLE_3D
    addTPSBindings<Real_, 1, 1, 1>(m, detail_module);
#endif
    // addTPSBindings<Real_, 2, 2, 2>(m, detail_module);
    using VXd = Eigen::Matrix<Real_, Eigen::Dynamic, 1>;

    // Filters
    using Filter_ = Filter<Real_>;
    py::class_<Filter_, std::shared_ptr<Filter_>>(detail_module, Filter_::mangledName().c_str())
        .def("setInputDimensions",  &Filter_::setInputDimensions,  py::arg("gridDims"))
        .def("setOutputDimensions", &Filter_::setOutputDimensions, py::arg("gridDims"))
        .def_property_readonly("inputDimensions",  &Filter_::inputDimensions)
        .def_property_readonly("outputDimensions", &Filter_::outputDimensions)
        .def("apply", [](Filter_ &filter, const VXd &x) -> VXd {
            filter.checkGridDimensionsAreSet();
            NDVector<Real_> in(filter.inputDimensions()), out(filter.outputDimensions());
            in.fill(x);
            filter.apply(in, out);
            return Eigen::Map<const VXd>(out.data().data(), out.size());
        }, py::arg("x"));
        ;

    using FC = FilterChain<Real_>;
    py::class_<FC, std::shared_ptr<FC>>(m, FC::mangledName().c_str())
        .def(py::init<typename FC::Filters, const EigenNDIndex &>(), py::arg("filters"), py::arg("outGridDimensions"))
        .def("numVars",          &FC::numVars)
        .def("numPhysicalVars",  &FC::numVars)
        .def("gridDims",         &FC::gridDims,         "Input grid dimensions")
        .def("physicalGridDims", &FC::physicalGridDims, "Output grid dimensions")
        .def("setDesignVars",    &FC::setDesignVars, py::arg("xDesign"))
        .def("backprop",         [](const FC &fc, Eigen::Ref<const VXd> &g) -> VXd {
                                       if (size_t(g.size()) != fc.numPhysicalVars()) throw std::runtime_error("Size mismatch");
                                       NDVector<Real_> g_nd(fc.physicalGridDims());
                                       g_nd.flattened() = g;
                                       fc.backprop(g_nd);
                                       return g_nd.flattened();
                                   }, py::arg("g"))
        .def("designVars",       [](const FC &fc) { return fc.designVars  ().flattened(); })
        .def("physicalVars",     [](const FC &fc) { return fc.physicalVars().flattened(); })
        .def_property_readonly("filters", &FC::filters, py::return_value_policy::reference)
        ;

    using PyF = PythonFilter<Real_>;
    py::class_<PyF, Filter_, std::shared_ptr<PyF>>(m, PyF::mangledName().c_str())
        .def(py::init<>())
        .def_readwrite("apply_cb",    &PyF::apply_cb)
        .def_readwrite("backprop_cb", &PyF::backprop_cb)
        ;

    using PF = ProjectionFilter<Real_>;
    py::class_<PF, Filter_, std::shared_ptr<PF>>(m, PF::mangledName().c_str())
        .def(py::init<Real_>(), py::arg("beta"))
        .def(py::init<>())
        .def("invert", &PF::invert, py::arg("filteredValue"))
        .def_property("beta", &PF::getBeta, &PF::setBeta)
        ;

    using SF = SmoothingFilter<Real_>;
    py::class_<SF, Filter_, std::shared_ptr<SF>> pySF(m, SF::mangledName().c_str());
    py::enum_<typename SF::Type>(pySF, "Type")
        .value("Const",  SF::Type::Const)
        .value("Linear", SF::Type::Linear)
        ;

    pySF
        .def(py::init<size_t, typename SF::Type>(), py::arg("radius") = 1, py::arg("type") = SF::Type::Const)
        .def_readwrite("radius", &SF::radius)
        .def_readwrite("type",   &SF::type)
        ;

    using UF = UpsampleFilter<Real_>;
    py::class_<UF, Filter_, std::shared_ptr<UF>>(m, UF::mangledName().c_str())
        .def(py::init<size_t>(), py::arg("factor"))
        ;

    using VCF = VertexToCellFilter<Real_>;
    py::class_<VCF, Filter_, std::shared_ptr<VCF>>(m, VCF::mangledName().c_str())
        .def(py::init<>())
        ;

    using LF = LangelaarFilter<Real_>;
    py::class_<LF, Filter_, std::shared_ptr<LF>>(m, LF::mangledName().c_str())
        .def(py::init<>())
        ;

    // Constraints
    using C = Constraint<Real_>;
    py::class_<C, std::shared_ptr<C>>(detail_module, C::mangledName().c_str());
    using TVC = TotalVolumeConstraint<Real_>;
    py::class_<TVC, C, std::shared_ptr<TVC>>(m, TVC::mangledName().c_str())
        .def(py::init<Real_>(), py::arg("volumeFraction"))
        .def_readwrite("volumeFraction", &TVC::m_volumeFraction, py::return_value_policy::reference)
        ;
}

PYBIND11_MODULE(pyVoxelFEM, m)
{
    m.doc() = "Voxel-based finite element codebase";

    py::module::import("MeshFEM");
    py::module detail_module = m.def_submodule("detail");

    // TODO define the enum class
    py::enum_<InterpolationLaw>(m, "InterpolationLaw")
        .value("SIMP", InterpolationLaw::SIMP)
        .value("RAMP", InterpolationLaw::RAMP)
        .export_values();

    addBindings<double>(m, detail_module);
    // addBindings<float >(m, detail_module);

    py::enum_<NumberType>(m, "NumberType")
        .value("DOUBLE", NumberType::DOUBLE)
        .value("FLOAT",  NumberType::FLOAT)
        ;
    // Factory function masquerading as a Python class; based on the list of
    // degrees-per-dimension passed, we instantiate the correct type of
    // TensorProductSimulator.
    m.def("TensorProductSimulator", [](const std::vector<size_t> &degreesPerDimension,
                                       const std::array<Eigen::VectorXd, 2> &domainBBox,
                                       const DynamicEigenNDIndex &elementsPerDimension,
                                       NumberType ntype) {
                auto it = g_factories.find(std::make_pair(ntype, degreesPerDimension));
                if (it == g_factories.end()) throw std::runtime_error("No template instantiation matching degreesPerDimension/number type!");
                return it->second(domainBBox, elementsPerDimension); // call the appropriate factory function, passing elementsPerDimension
            }, py::arg("degreesPerDimension"), py::arg("domainBBox"), py::arg("elementsPerDimension"), py::arg("numberType") = NumberType::DOUBLE);

}
