////////////////////////////////////////////////////////////////////////////////
// OptimalityCriterion.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  An implementation of the optimality criterion method for solving an
//  optimization problem with a single equality constraint (or a single
//  inequality constraint that is known to be active at the optimum).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  03/08/2021 17:48:46
////////////////////////////////////////////////////////////////////////////////
#ifndef OPTIMALITYCRITERION_HH
#define OPTIMALITYCRITERION_HH
#include "NDVector.hh"

template<class Problem>
struct ConstraintEvaluator {
    using Scalar = typename Problem::Scalar;
    ConstraintEvaluator(Problem &p) : m_p(p) {
        auto gridSize = p.getSimulator().NbElementsPerDimension();
        varSizes = p.filterChain().designVars().sizes();
        x.resize(varSizes);
        xscratch.resize(gridSize);
    }
    template<class Vars>
    Scalar operator()(const Vars &steppedVars) {
        x.resize(varSizes);
        x.flattened() = steppedVars;
        return m_p.evaluateOCConstraintAtVars(x, xscratch);
    }
private:
    std::vector<size_t> varSizes;
    Problem &m_p;
    NDVector<Scalar> x, xscratch;
};

template<class Problem>
struct OCOptimizer {
    using Scalar = typename Problem::Scalar;
    using VXd    = typename Problem::VXd;
    using MXd    = typename Problem::MXd;

    static constexpr size_t SIMD_WIDTH = 4;
    using VSIMD = Eigen::Array<Scalar, SIMD_WIDTH, 1>;

    OCOptimizer(Problem &p) : m_p(p), constraint_evaluator(p) {
        lambda_min = 1;
        lambda_max = 2;
    }

    void step(Scalar m = 0.2, Scalar p = 0.5, Scalar ctol = 1e-6, bool inplace = true) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("OC step");
        if (inplace) {
            m_p.evaluateObjectiveGradient(dJ);
            m_p.evaluateConstraintsJacobian(dc);
        }
        else {
            dJ = m_p.evaluateObjectiveGradientAndReturn();
            dc = m_p.evaluateConstraintsJacobianAndReturn();
        }
        auto x0 = m_p.getVars();

        m_steppedVars.resize(x0.size());
        auto ceval = [&](Scalar lambda) {
            size_t numBlocks = size_t(x0.size()) / SIMD_WIDTH;
            parallel_for_range(numBlocks, [&](size_t i) {
                VSIMD v_x0(x0.template segment<SIMD_WIDTH>(SIMD_WIDTH * i)),
                      v_dJ(dJ.template segment<SIMD_WIDTH>(SIMD_WIDTH * i)),
                      v_dc(dc.template   block<1, SIMD_WIDTH>(0, SIMD_WIDTH * i).transpose());
                VSIMD result = ((v_x0 * (v_dJ / (v_dc * lambda)).pow(p)).cwiseMax(v_x0 - m).cwiseMin(v_x0 + m).cwiseMax(0.0).cwiseMin(1.0));
                // work around near-vanishing sensitivities under self-supporting filter.
                m_steppedVars.template segment<SIMD_WIDTH>(SIMD_WIDTH * i) = result.isFinite().select(result, v_x0);
            });
            // left-over
            for (size_t i = numBlocks * SIMD_WIDTH; i < size_t(x0.size()); ++i) {
                Scalar result = (x0[i] * std::pow(dJ[i] / (dc(0, i) * lambda), p));
                result = std::min(std::max(std::min(std::max(result, x0[i] - m), x0[i] + m), Scalar(0.0)), Scalar(1.0));
                if (!std::isfinite(result)) result = x0[i];
                m_steppedVars[i] = result;
            }

            return constraint_evaluator(m_steppedVars);
        };

        // Note: constraint function "c" is continuous and monotonically increasing,
        // and we seek to find its unique root with a bisection method.
        // (c = (1 - vol/(N_e * target_vol_frac))

        // TODO: determine if we want to reset the Lagrange multiplier brackets:
        // lambda_min = 1, lambda_max = 2;
        BENCHMARK_START_TIMER_SECTION("Bisection");

        // std::cout << "[" << lambda_min << ", " << lambda_max << "]" << std::endl;

        const Scalar dilation = 32;
        Scalar lambda_mid = 0.5 * (lambda_min + lambda_max);
        lambda_max = dilation * lambda_max + (1 - dilation) * lambda_mid;
        lambda_min = std::max(dilation * lambda_min + (1 - dilation) * lambda_mid, Scalar(0.01));

        // std::cout << "[" << lambda_min << ", " << lambda_max << "]" << std::endl;

        size_t nit = 0;
        for (; nit < m_loopCountGuard; ++nit) {
            if (ceval(lambda_min) < 0) break;
            lambda_max = lambda_min; lambda_min /= 2;
        }
        if (nit == m_loopCountGuard) {
            BENCHMARK_STOP_TIMER_SECTION("Bisection");
            throw std::runtime_error("Bracketing constraint(lambda_min) < 0 failed ("+std::to_string(m_loopCountGuard)+" times).");
        }
        if (nit == 0)
            for (; nit < m_loopCountGuard; ++nit) {
                if (ceval(lambda_max) > 0) break;
                lambda_min = lambda_max; lambda_max *= 2;
            }
        if (nit == m_loopCountGuard) {
            BENCHMARK_STOP_TIMER_SECTION("Bisection");
            throw std::runtime_error("Bracketing constraint(lambda_max) > 0 failed ("+std::to_string(m_loopCountGuard)+" times).");
        }

        Scalar violation;
        do {
            lambda_mid = 0.5 * (lambda_min + lambda_max);
            violation = ceval(lambda_mid);
            if (std::abs(violation) <= ctol) break;
            ++nit;
            if (violation < 0) lambda_min = lambda_mid;
            if (violation > 0) lambda_max = lambda_mid;
        } while (true);

        BENCHMARK_STOP_TIMER_SECTION("Bisection");

        m_p.setVars(m_steppedVars);

        // m_p.setVars(stepped_vars_for_lambda(lambda_mid));

        // std::cout << "objective, constraint, lambda estimate: " << m_p.evaluateObjective() << "\t" << m_p.evaluateConstraints()[0] << "\t " << lambda_mid << " (" << lambda_max - lambda_min << ") " << nit << std::endl;
    }

private:
    Scalar lambda_min, lambda_max; // bracket around current guess of Lagrange multipliers
    Problem &m_p;
    ConstraintEvaluator<Problem> constraint_evaluator;

    VXd dJ;
    MXd dc;
    VXd m_steppedVars;
    size_t m_loopCountGuard = 100; // Exit the program if "bracketing" c(lam_min) or c(lam_max) fails.
};

#endif /* end of include guard: OPTIMALITYCRITERION_HH */
