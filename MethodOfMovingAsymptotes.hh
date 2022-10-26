/**
 * @file MethodOfMovingAsymptotes.hh
 * @author Kai Lan (kai.weixian.lan@gmail.com)
 * @brief  Method of Moving Asymptotes for nonlinear optimization problems.
 * Orginal work written by Krister Svanberg in Matlab.
 * @date 2022-07-23
 */
#ifndef METHODOFMOVINGASYMPTOTES
#define METHODOFMOVINGASYMPTOTES
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <algorithm>
// #define EIGEN_USE_BLAS
#include <Eigen/Dense>
#include "ParallelVectorOps.hh"
#include <MeshFEM/GlobalBenchmark.hh>
#include "FixedSizeDeque.hh"


// General optimization problem
// min_x    f_0(x) + a_0 * z + sum_{i=1}^m (c_i * y_i + 1/2 * d_i * y_i^2)
// subject to    f_i(x) - a_i * z - y_i <= 0, i = 1, ..., m
//                     (x_min)_j <= x_j <= (x_max)_j, j = 1, ..., n
//                                  y_i >= 0, i = 1, ..., m
//                                    z >= 0
class MMA {
    using AXd = Eigen::ArrayXd;
    using AXXd = Eigen::ArrayXXd;
    using VXd = Eigen::VectorXd;
    using MXd = Eigen::MatrixXd;
public:
    MMA(int numVars, int numConstr, const AXd& xmin, const AXd& xmax, const std::function<AXd(const AXd&)>& f, const std::function<AXXd(const AXd&)>& df_dx)
        : m(numConstr), n(numVars), x_min(xmin), x_max(xmax), f(f), df_dx(df_dx), subp(*this), x_diff(x_max - x_min),
        a(AXd::Zero(m)), d(AXd::Ones(m)), c(AXd::Constant(m, 1000)) {
        max_outer_iter = 50; // Default: 50 outer iterations
        outer_iter = 0;
        inner_iter = 0;
        df_dx_cur_plus.resize(m+1, n);
        df_dx_cur_minus.resize(m+1, n);
        p_cur.resize(m+1, n);
        q_cur.resize(m+1, n);
        alpha_cur.resize(n);
        beta_cur.resize(n);
        current.u_minus_x.resize(n);
        current.u_minus_x_sq.resize(n);
        current.x_minus_l.resize(n);
        current.x_minus_l_sq.resize(n);
        l_cur.resize(n);
        u_cur.resize(n);
    }
    void enableGCMMA(bool enable) { enableInner = enable; }
    void setInitialVar(const AXd& xinit) {
        x_history.addToHistory(xinit);
    }
    // Generate subproblem:
    // min_x    f_0^{k}(x) + a_0 * z + sum_{i=1}^m (c_i * y_i + 1/2 * d_i * y_i^2)
    // subject to   f_i^{k}(x) - a_i * z - y_i <= 0, i = 1, ..., m
    //                       (alpha)^{k}_j <= x_j <= (beta)^{k}_j, j = 1, ..., n
    //                                        y_i >= 0, i = 1, ..., m
    //                                          z >= 0
    void step() { // enable the inner steps to use GCMMA
        if (x_history.size() == 0) throw std::runtime_error("Must specify an initial value");
        outer_iter ++;
        if (outer_iter <= 2) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
                l_cur.segment(r.begin(), r.size()) = x_cur().segment(r.begin(), r.size()) - asyinit * x_diff.segment(r.begin(), r.size());
                u_cur.segment(r.begin(), r.size()) = x_cur().segment(r.begin(), r.size()) + asyinit * x_diff.segment(r.begin(), r.size());
            });
        }
        else { // gamma_j^{k} = 0.7 if zzz_j < 0, 1.2 if zzz_j > 0, and 1   if zzz_j = 0
               // where zzz_j^{k} = (x_j^{k} - x_j^{k-1}) * (x_j^{k-1} - x_j^{k-2})
            // double tol = 1e-12; // Treat difference lower than this value as 0.
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
                for (int j = r.begin(); j < r.end(); ++j) {
                    double diff = (x_history[0](j) - x_history[1](j)) * (x_history[1](j) - x_history[2](j));
                    double gamma_j;
                    gamma_j = diff > 0? gamma_vals.at(1) : gamma_vals.at(0);
                    // if (diff < -tol) gamma_j = gamma_vals.at(0);
                    // else if (diff > tol) gamma_j = gamma_vals.at(1);
                    // else gamma_j = gamma_vals.at(2);
                    l_cur(j) = std::max(std::min(x_history[0](j) - gamma_j * (x_history[1](j) - l_cur(j)), x_cur()(j) - 0.01 * x_diff(j)), x_cur()(j) - 10 * x_diff(j));
                    u_cur(j) = std::min(std::max(x_history[0](j) + gamma_j * (u_cur(j) - x_history[1](j)), x_cur()(j) + 0.01 * x_diff(j)), x_cur()(j) + 10 * x_diff(j));
                }
            });
        }
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
            alpha_cur.segment(r.begin(), r.size()) = x_min.segment(r.begin(), r.size()).max(l_cur.segment(r.begin(), r.size()) + albefa * (x_cur().segment(r.begin(), r.size()) - l_cur.segment(r.begin(), r.size()))).max(x_cur().segment(r.begin(), r.size()) - move * x_diff.segment(r.begin(), r.size()));
            beta_cur.segment(r.begin(), r.size())  = x_max.segment(r.begin(), r.size()).min(u_cur.segment(r.begin(), r.size()) - albefa * (u_cur.segment(r.begin(), r.size()) - x_cur().segment(r.begin(), r.size()))).min(x_cur().segment(r.begin(), r.size()) + move * x_diff.segment(r.begin(), r.size()));
        });
        // Store f(x^k) and positive and negative parts of df/dx(x^k)
        f_cur = f(x_cur());
        df_dx_cur_plus = df_dx(x_cur());

        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
            df_dx_cur_minus.middleCols(r.begin(), r.size()) = (-df_dx_cur_plus.middleCols(r.begin(), r.size())).max(0);
            df_dx_cur_plus.middleCols(r.begin(), r.size()) = df_dx_cur_plus.middleCols(r.begin(), r.size()).max(0);
            current.u_minus_x.segment(r.begin(), r.size()) = u_cur.segment(r.begin(), r.size()) - x_cur().segment(r.begin(), r.size());
            current.u_minus_x_sq.segment(r.begin(), r.size()) = current.u_minus_x.segment(r.begin(), r.size()).square();
            current.x_minus_l.segment(r.begin(), r.size()) = x_cur().segment(r.begin(), r.size()) - l_cur.segment(r.begin(), r.size());
            current.x_minus_l_sq.segment(r.begin(), r.size()) = current.x_minus_l.segment(r.begin(), r.size()).square();
        });
        if (enableInner) {
            old = current;
            do {
                rho = next_rho();
                for (int i = 0; i < m+1; ++i) {
                    rho_over_diffx = rho(i) / x_diff;
                    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
                        p_cur.row(i).segment(r.begin(), r.size()) = old.u_minus_x_sq.segment(r.begin(), r.size()).transpose() * (1.001 * df_dx_cur_plus.row(i).segment(r.begin(), r.size()) + 0.001 * df_dx_cur_minus.row(i).segment(r.begin(), r.size()) + rho_over_diffx.segment(r.begin(), r.size()).transpose());
                        q_cur.row(i).segment(r.begin(), r.size()) = old.x_minus_l_sq.segment(r.begin(), r.size()).transpose() * (0.001 * df_dx_cur_plus.row(i).segment(r.begin(), r.size()) + 1.001 * df_dx_cur_minus.row(i).segment(r.begin(), r.size()) + rho_over_diffx.segment(r.begin(), r.size()).transpose());
                    });
                }
                r_cur = f_cur - sub_g_eval(true, true); // outer_x
                x_inner = subp.subsolve();
                inner_iter ++;
            } while (!isFeasible());
            x_history.addToHistory(x_inner);
            inner_iter = 0;
        }
        else {
            rho_over_diffx = raa0 / x_diff;
            for (int i = 0; i < m+1; ++i) { // Dot product with `one` vector
                tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
                    p_cur.row(i).segment(r.begin(), r.size()) = current.u_minus_x_sq.segment(r.begin(), r.size()).transpose() * (1.001 * df_dx_cur_plus.row(i).segment(r.begin(), r.size()) + 0.001 * df_dx_cur_minus.row(i).segment(r.begin(), r.size()) + rho_over_diffx.segment(r.begin(), r.size()).transpose());
                    q_cur.row(i).segment(r.begin(), r.size()) = current.x_minus_l_sq.segment(r.begin(), r.size()).transpose() * (0.001 * df_dx_cur_plus.row(i).segment(r.begin(), r.size()) + 1.001 * df_dx_cur_minus.row(i).segment(r.begin(), r.size()) + rho_over_diffx.segment(r.begin(), r.size()).transpose());
                });
            }
            r_cur = f_cur - sub_g_eval(true); // (m+1) x 1
            x_history.addToHistory(subp.subsolve());
        }
    }
    AXd getOptimalVar() const { return x_history[0]; }
private:
    AXd next_rho() const {
        AXd result(m+1);
        if (inner_iter == 0) {
            for (int i = 0; i < m+1; ++i)
                result(i) = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), 0.0, [&](const tbb::blocked_range<size_t> &r, double total) {
                    return total + 0.1/n * (df_dx_cur_plus.row(i).segment(r.begin(), r.size()) + df_dx_cur_minus.row(i).segment(r.begin(), r.size())).matrix()
                            * x_diff.segment(r.begin(), r.size()).matrix();
                }, std::plus<double>());
            return result.max(1e-6);
        }
        double d = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), (double)0.0, [&](const tbb::blocked_range<size_t> &r, double total) {
                        return total + ((u_cur.segment(r.begin(), r.size()) - l_cur.segment(r.begin(), r.size())) * (x_inner.segment(r.begin(), r.size()) - x_cur().segment(r.begin(), r.size())).square()
                        / (current.u_minus_x.segment(r.begin(), r.size())*current.x_minus_l.segment(r.begin(), r.size())*x_diff.segment(r.begin(), r.size()))).sum();
                    }, std::plus<double>());
        for (int i = 0; i < m+1; ++i) {
            double delta = diff_f_subf(i) / d;
            if (delta < 0) result(i) = rho(i);
            else result(i) = std::min(1.1 * (rho(i) + delta), 10 * rho(i));
        }
        return result;
    }
    const AXd& x_cur() const { return x_history[0]; } // Read the current x^{k}

    // g_i(x) = sum_{j=1}^n (p_{ij}/(u_j - x_j) + q_{ij}/(x_j - l_j)), i = 0, ..., m
    AXd sub_g_eval(bool includeAllRows=false, bool useOldData=false) const {
        const intermediate_vars& vars = useOldData? old : current;
        if (includeAllRows) {
            AXd result(m+1);
            for (int constraint = 0; constraint < m+1; ++constraint) {
                result(constraint) = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), double(0.0), [&](const tbb::blocked_range<size_t> &r, double total) {
                    return total + p_cur.row(constraint).middleCols(r.begin(), r.size()) * vars.u_minus_x.segment(r.begin(), r.size()).inverse().matrix()
                                    + q_cur.row(constraint).middleCols(r.begin(), r.size()) * vars.x_minus_l.segment(r.begin(), r.size()).inverse().matrix();
                    }, std::plus<double>());
            }
            return result;
        }
        AXd result(m);
        for (int constraint = 0; constraint < m; ++constraint) {
            result(constraint) = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), double(0.0), [&](const tbb::blocked_range<size_t> &r, double total) {
                return total + p_cur.row(constraint + 1).middleCols(r.begin(), r.size()) * vars.u_minus_x.segment(r.begin(), r.size()).inverse().matrix()
                                + q_cur.row(constraint + 1).middleCols(r.begin(), r.size()) * vars.x_minus_l.segment(r.begin(), r.size()).inverse().matrix();
                }, std::plus<double>());
        }
        return result;
    }
    // dg_i(x)/dx = sum_{j=1}^n (p_{ij}/(u_j - x_j)^2 - q_{ij}/(x_j - l_j)^2), i = 0, ..., m
    // AXXd sub_g_gradient() const {
    //     return p_cur * current.u_minus_x_sq.inverse().matrix().asDiagonal() - q_cur.matrix() * current.x_minus_l_sq.inverse().matrix().asDiagonal(); // m x n
    // }
    AXd sub_f_eval() const {
        return sub_g_eval(true) + r_cur;
    }
    // A outer solution is feasible if f_i^{k,l }(x^{k,l}) > f_i(x^{k,l}), i = 0, 1, ..., m
    // where f_i^{k, l}(x) = sum_{j=0}^n (p_{ij}/(u_j - x_j) + q_{ij}/(x_j - l_j)) + r_i^{k, l}.
    bool isFeasible() {
        diff_f_subf = f(x_inner) - sub_f_eval();
        return diff_f_subf.maxCoeff() < 0;
    }
    // Constant
    const std::vector<double> gamma_vals{0.7, 1.2, 1.0};
    const double raa0 = 1e-5, albefa = 0.1, move = 0.5, asyinit = 0.5;

    // User defined
    bool enableInner = false;
    const int m, n; // m: number of constraints, n: number of variables
    const double a0 = 1;
    const AXd a, c, d; // m x 1
    const AXd x_min, x_max, x_diff; // constraint: (x_min)_j <= x_j <= (x_max)_j, j = 1, ..., n
    const std::function<AXd(const AXd&)> f; // Objective and constraint function
    const std::function<AXXd(const AXd&)> df_dx; // Objective and constraint gradients

    int outer_iter, inner_iter, max_outer_iter;
    FixedSizeDeque<AXd> x_history{3}; // store current and previous two, front to x^{k}, x^{k-1}, x^{k-2}. ***x_history(3) does not work***

    // subproblem vars
    AXd l_cur, u_cur; // l^{k} and u^{k}
    AXd alpha_cur, beta_cur; // alpha^{k} and beta^{k}
    AXd x_inner; // x^{k, l}
    AXd rho; // (m+1) x 1, rho^{k, l}
    MXd p_cur, q_cur; // (m+1) x n, p^{k, l}, q^{k, l}. l = 0 for ordinary MMA.
    AXd r_cur; // r^{k, l}. l = 0 for ordinary MMA.
    AXd f_cur; // (m+1) x 1, f(x^{k})
    AXXd df_dx_cur_plus, df_dx_cur_minus; // (m+1) x n, max((df / dx)(x^{k}), 0) and max(-(df / dx)(x^{k}), 0)
    AXd diff_f_subf, rho_over_diffx;
    struct intermediate_vars {
        AXd u_minus_x, x_minus_l, u_minus_x_sq, x_minus_l_sq; // Intermediate helper
        intermediate_vars& operator= (const intermediate_vars& o) {
            int size = o.u_minus_x.size();
            if (u_minus_x.size() == 0) {
                u_minus_x.resize(size);
                x_minus_l.resize(size);
                u_minus_x_sq.resize(size);
                x_minus_l_sq.resize(size);
            }
            tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t> &r) {
                u_minus_x   .segment(r.begin(), r.size()) = o.u_minus_x   .segment(r.begin(), r.size());
                x_minus_l   .segment(r.begin(), r.size()) = o.x_minus_l   .segment(r.begin(), r.size());
                u_minus_x_sq.segment(r.begin(), r.size()) = o.u_minus_x_sq.segment(r.begin(), r.size());
                x_minus_l_sq.segment(r.begin(), r.size()) = o.x_minus_l_sq.segment(r.begin(), r.size());
            });
            return *this;
        }
    } current, old;

    struct Subproblem { // Only m < n case for now
        Subproblem(MMA& mma) : mma(mma), m(mma.m), n(mma.n) {}
        AXd subsolve() {
            BENCHMARK_SCOPED_TIMER_SECTION timer("Sub solve");
            init_vars();
            double eps = 1;
            while(eps > 1e-7) {
                BENCHMARK_SCOPED_TIMER_SECTION timer("epsi loop");
                double res_old;
                for (int i = 0; i < 10; ++i) { // inner loop guard
                    solve_for_newton_direction(eps);
                    if (i == 0) res_old = squared_residual(eps, true);
                    res_old = newton_step_backtrack(eps, res_old);
                    if (KKT_inf_norm(eps) <= 0.9 * eps) break;
                }
                eps *= 0.1;
            }
            return data.x;
        }


        double KKT_inf_norm(double eps) const {
            BENCHMARK_SCOPED_TIMER_SECTION timer("Computing KKT norm");
            double r = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), 0.0,
                [&](const tbb::blocked_range<size_t> &r, double max) {
                max = std::max(max, eq_a(dpsi_dx, r).cwiseAbs().maxCoeff());
                max = std::max(max, eq_e(eps, r).cwiseAbs().maxCoeff());
                max = std::max(max, eq_f(eps, r).cwiseAbs().maxCoeff());
                return max;
                },
                [&](double x, double y) { return std::max(x, y); }
            );
            r = std::max(r, eq_b().cwiseAbs().maxCoeff());
            r = std::max(r, std::abs(eq_c()));
            r = std::max(r, eq_d(mma.sub_g_eval()).cwiseAbs().maxCoeff());
            r = std::max(r, eq_g(eps).cwiseAbs().maxCoeff());
            r = std::max(r, std::abs(eq_h(eps)));
            r = std::max(r, eq_i(eps).cwiseAbs().maxCoeff());
            return r;
        }
    private:
        void init_vars() {
            data.x.resize(n);
            data.xi.resize(n);
            data.eta.resize(n);
            data.y.setOnes(m);
            data.z = 1;
            data.zeta = 1;
            data.lam.setOnes(m);
            data.s.setOnes(m);
            data.mu = (mma.c / 2).max(1.0);
            Dx.resize(n);
            G.resize(m, n);
            delta.x.resize(n);
            delta.xi.resize(n);
            delta.eta.resize(n);
            dpsi_dx.resize(n);
            plam.resize(n);
            qlam.resize(n);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r){
                data.x.segment(r.begin(), r.size()) = 0.5 * (mma.alpha_cur.segment(r.begin(), r.size()) + mma.beta_cur.segment(r.begin(), r.size()));
                data.xi.segment(r.begin(), r.size()) = (data.x.segment(r.begin(), r.size()) - mma.alpha_cur.segment(r.begin(), r.size())).inverse().max(1.0);
                data.eta.segment(r.begin(), r.size()) = (mma.beta_cur.segment(r.begin(), r.size()) - data.x.segment(r.begin(), r.size())).inverse().max(1.0);
                mma.current.u_minus_x.segment(r.begin(), r.size()) = mma.u_cur.segment(r.begin(), r.size()) - data.x.segment(r.begin(), r.size());
                mma.current.u_minus_x_sq.segment(r.begin(), r.size()) = mma.current.u_minus_x.segment(r.begin(), r.size()).square();
                mma.current.x_minus_l.segment(r.begin(), r.size()) = data.x.segment(r.begin(), r.size()) - mma.l_cur.segment(r.begin(), r.size());
                mma.current.x_minus_l_sq.segment(r.begin(), r.size()) = mma.current.x_minus_l.segment(r.begin(), r.size()).square();
                plam.segment(r.begin(), r.size()) = mma.p_cur.row(0).segment(r.begin(), r.size()).transpose() + mma.p_cur.bottomRows(m).middleCols(r.begin(), r.size()).transpose() * data.lam.matrix();
                qlam.segment(r.begin(), r.size()) = mma.q_cur.row(0).segment(r.begin(), r.size()).transpose() + mma.q_cur.bottomRows(m).middleCols(r.begin(), r.size()).transpose() * data.lam.matrix();
                dpsi_dx.segment(r.begin(), r.size()) = plam.segment(r.begin(), r.size())/mma.current.u_minus_x_sq.segment(r.begin(), r.size())
                    - qlam.segment(r.begin(), r.size())/mma.current.x_minus_l_sq.segment(r.begin(), r.size());
            });
            gvec = mma.sub_g_eval();
        }
        // Solve for del_x, ..., del_s given x, ..., s
        void solve_for_newton_direction(double eps) {
            BENCHMARK_SCOPED_TIMER_SECTION timer("Newton direction searching");
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
                delta.x.segment(r.begin(), r.size()) = dpsi_dx.segment(r.begin(), r.size()) - eps / (data.x.segment(r.begin(), r.size()) - mma.alpha_cur.segment(r.begin(), r.size())) + eps / (mma.beta_cur.segment(r.begin(), r.size()) - data.x.segment(r.begin(), r.size())); // n x 1
                Dx.segment(r.begin(), r.size()) = 2 * (plam.segment(r.begin(), r.size()))/(mma.current.u_minus_x.segment(r.begin(), r.size()) * mma.current.u_minus_x_sq.segment(r.begin(), r.size()))
                                                + 2 * (qlam.segment(r.begin(), r.size()))/(mma.current.x_minus_l.segment(r.begin(), r.size()) * mma.current.x_minus_l_sq.segment(r.begin(), r.size()))
                                                + data.xi.segment(r.begin(), r.size()) / (data.x.segment(r.begin(), r.size()) - mma.alpha_cur.segment(r.begin(), r.size()))
                                                + data.eta.segment(r.begin(), r.size()) / (mma.beta_cur.segment(r.begin(), r.size()) - data.x.segment(r.begin(), r.size()));
                G.middleCols(r.begin(), r.size()) = mma.p_cur.bottomRows(m).middleCols(r.begin(), r.size()) * mma.current.u_minus_x_sq.segment(r.begin(), r.size()).inverse().matrix().asDiagonal()
                                                  - mma.q_cur.bottomRows(m).middleCols(r.begin(), r.size()) * mma.current.x_minus_l_sq.segment(r.begin(), r.size()).inverse().matrix().asDiagonal();
            });

            AXd Dy = mma.d + data.mu / data.y;
            AXd dy = mma.c + mma.d * data.y - data.lam - eps / data.y;

            MXd M(m+1, m+1); // M.topLeftCorner(m, m) = G * Dx.inverse().matrix().asDiagonal() * G.transpose();
            for (int ci = 0; ci < m; ++ci)
                for (int cj = ci; cj < m; ++cj)
                    M(ci, cj) = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), 0.0, [&](const tbb::blocked_range<size_t> &r, double total) {
                            return total + G.row(ci).middleCols(r.begin(), r.size()).dot(
                                            (G.row(cj).middleCols(r.begin(), r.size()).transpose().array() / Dx.segment(r.begin(), r.size())).matrix());
                        }, std::plus<double>());

            M.diagonal().topRows(m) += (data.s / data.lam + Dy.inverse()).matrix();
            M.rightCols(1).topRows(m) = mma.a;
            M(m, m) = - data.zeta / data.z;
            M.triangularView<Eigen::Lower>() = M.transpose();

            VXd rhs(m+1), sol;
            rhs.topRows(m) = (gvec - mma.a * data.z - data.y + mma.r_cur.bottomRows(m) + eps / data.lam + dy / Dy).matrix(); // - G * Dx.inverse().matrix().asDiagonal() * dx.matrix();
            for (int constraint = 0; constraint < m; ++constraint) {
                rhs[constraint] -= tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), 0.0, [&](const tbb::blocked_range<size_t> &r, double total) {
                            return total + G.row(constraint).middleCols(r.begin(), r.size()) * (delta.x.segment(r.begin(), r.size()) / Dx.segment(r.begin(), r.size())).matrix();
                        }, std::plus<double>());
            }
            rhs(m) = mma.a0 - data.lam.matrix().dot(mma.a.matrix()) - eps / data.z;
            sol = M.colPivHouseholderQr().solve(rhs);
            delta.lam = sol.topRows(m);
            delta.z = sol(m);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
                delta.x.segment(r.begin(), r.size()) = -(delta.x.segment(r.begin(), r.size()) + (G.middleCols(r.begin(), r.size()).transpose() * delta.lam.matrix()).array()) / Dx.segment(r.begin(), r.size()); // n x 1
                delta.xi.segment(r.begin(), r.size()) = - data.xi.segment(r.begin(), r.size()) + eps / (data.x.segment(r.begin(), r.size()) - mma.alpha_cur.segment(r.begin(), r.size())) - data.xi.segment(r.begin(), r.size()) * (delta.x.segment(r.begin(), r.size())) / (data.x.segment(r.begin(), r.size()) - mma.alpha_cur.segment(r.begin(), r.size())); // n x 1
                delta.eta.segment(r.begin(), r.size()) = - data.eta.segment(r.begin(), r.size()) + eps / (mma.beta_cur.segment(r.begin(), r.size()) - data.x.segment(r.begin(), r.size())) + data.eta.segment(r.begin(), r.size()) * (delta.x.segment(r.begin(), r.size())) / (mma.beta_cur.segment(r.begin(), r.size()) - data.x.segment(r.begin(), r.size())); // n x 1
            });
            delta.y = delta.lam / Dy - dy / Dy;
            delta.mu = (eps - data.mu * delta.y) / data.y - data.mu;
            delta.zeta = (eps - data.zeta * delta.z) / data.z - data.zeta;
            delta.s = (eps - data.s * delta.lam) / data.lam - data.s;
        }
        // Line search in the Newton direction
        double newton_step_backtrack(double eps, double res_old) {
            BENCHMARK_SCOPED_TIMER_SECTION timer("Newton step seaching");
            double step = step_satisfy_KKT();
            newton_step(step);
            double res = squared_residual(eps);
            while (res > res_old) {
                step /= 2;
                newton_step(-step);
                res = squared_residual(eps);
            }
            return res;
        }
        // Need to satisfy the following KKT constriants, corresponding to eq 5.9 (j - n)
        // Note that we only worry about this when del_var is negative
        // x + t * del_x - alpha > eps
        // beta - (x + t * del_x) > eps
        // w + t * del_w > eps, for w = y, z, s, xi, eta, mu, zeta, lam
        // where `eps` is something small. We will set a relative ratio of 0.01 there, ie, w + t * del_w > 0.01 * w.
        // For numerical stability, consider step = 1 / max{(-1/0.99)*min{del_w/w}, 1}, or 1 / max{(-1.01)*min{del_w/w}, 1} for simplicity.
        double step_satisfy_KKT() const {
            double minCoeff = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), std::numeric_limits<double>::infinity(),
                [&](const tbb::blocked_range<size_t> &r, double min) {
                min = std::min(min, (delta.x.segment(r.begin(), r.size()) / (data.x.segment(r.begin(), r.size()) - mma.alpha_cur.segment(r.begin(), r.size()))).minCoeff());
                min = std::min(min, (delta.x.segment(r.begin(), r.size()) / (data.x.segment(r.begin(), r.size()) - mma.beta_cur.segment(r.begin(), r.size()))).minCoeff());
                min = std::min(min, (delta.xi.segment(r.begin(), r.size()) / data.xi.segment(r.begin(), r.size())).minCoeff());
                min = std::min(min, (delta.eta.segment(r.begin(), r.size()) / data.eta.segment(r.begin(), r.size())).minCoeff());
                return min;
                },
                [&](double x, double y) { return std::min(x, y); }
            );
            minCoeff = std::min({minCoeff, (delta.y / data.y).minCoeff(), delta.z / data.z, (delta.s / data.s).minCoeff(), (delta.mu / data.mu).minCoeff(), delta.zeta / data.zeta, (delta.lam / data.lam).minCoeff()});
            return 1 / std::max(-1.01 * minCoeff, 1.0);
        }
        // Ensure the new objectives are lower than old ones, corresponding tp eq 5.9 (a - i)
        double squared_residual(double eps, bool init=false) {
            if (!init) {
                tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
                    mma.current.u_minus_x.segment(r.begin(), r.size()) = mma.u_cur.segment(r.begin(), r.size()) - data.x.segment(r.begin(), r.size());
                    mma.current.u_minus_x_sq.segment(r.begin(), r.size()) = mma.current.u_minus_x.segment(r.begin(), r.size()).square();
                    mma.current.x_minus_l.segment(r.begin(), r.size()) = data.x.segment(r.begin(), r.size()) - mma.l_cur.segment(r.begin(), r.size());
                    mma.current.x_minus_l_sq.segment(r.begin(), r.size()) = mma.current.x_minus_l.segment(r.begin(), r.size()).square();
                    plam.segment(r.begin(), r.size()) = mma.p_cur.row(0).segment(r.begin(), r.size()).transpose() + mma.p_cur.bottomRows(m).middleCols(r.begin(), r.size()).transpose() * data.lam.matrix();
                    qlam.segment(r.begin(), r.size()) = mma.q_cur.row(0).segment(r.begin(), r.size()).transpose() + mma.q_cur.bottomRows(m).middleCols(r.begin(), r.size()).transpose() * data.lam.matrix();
                    dpsi_dx.segment(r.begin(), r.size()) = plam.segment(r.begin(), r.size())/mma.current.u_minus_x_sq.segment(r.begin(), r.size())
                        - qlam.segment(r.begin(), r.size())/mma.current.x_minus_l_sq.segment(r.begin(), r.size());
                });
                gvec = mma.sub_g_eval();
            }
            BENCHMARK_START_TIMER_SECTION("Computing step residual");
            double sq_norm_diff = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), 0.0, [&](const tbb::blocked_range<size_t> &r, double residual) {
                residual += eq_a(dpsi_dx, r).squaredNorm();
                residual += eq_e(eps, r).squaredNorm();
                residual += eq_f(eps, r).squaredNorm();
                return residual;
            }, std::plus<double>());
            sq_norm_diff += eq_b().squaredNorm() + std::pow(eq_c(), 2) + eq_d(gvec).squaredNorm()
                            + eq_g(eps).squaredNorm() + std::pow(eq_h(eps), 2) + eq_i(eps).squaredNorm();
            BENCHMARK_STOP_TIMER_SECTION("Computing step residual");
            return sq_norm_diff;
        }

        void newton_step(double step) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
                data.x .segment(r.begin(), r.size()) += step * delta.x.segment(r.begin(), r.size()); // n x 1
                data.xi.segment(r.begin(), r.size()) += step * delta.xi.segment(r.begin(), r.size()); // n x 1
                data.eta.segment(r.begin(), r.size()) += step * delta.eta.segment(r.begin(), r.size()); // n x 1
            });
            data.y += step * delta.y;
            data.z += step * delta.z;
            data.lam += step * delta.lam;
            data.mu += step * delta.mu;
            data.zeta += step * delta.zeta;
            data.s += step * delta.s;
        }

        VXd eq_a(const AXd& dpsi_dx, const tbb::blocked_range<size_t> &r) const {
            return dpsi_dx.segment(r.begin(), r.size()) - data.xi.segment(r.begin(), r.size()) + data.eta.segment(r.begin(), r.size());
        }
        VXd eq_b() const { return mma.c + mma.d * data.y  - data.lam - data.mu; }
        double eq_c() const { return mma.a0 - data.zeta  - data.lam.matrix().dot(mma.a.matrix()); }
        VXd eq_d(const AXd& g) const { return g - mma.a * data.z  - data.y + data.s + mma.r_cur.bottomRows(m); }
        VXd eq_e(double eps, const tbb::blocked_range<size_t> &r) const { return data.xi.segment(r.begin(), r.size()) * (data.x.segment(r.begin(), r.size()) - mma.alpha_cur.segment(r.begin(), r.size())) - eps; }
        VXd eq_f(double eps, const tbb::blocked_range<size_t> &r) const { return data.eta.segment(r.begin(), r.size()) * (mma.beta_cur.segment(r.begin(), r.size()) - data.x.segment(r.begin(), r.size())) - eps; }
        VXd eq_g(double eps) const { return data.mu * data.y - eps; }
        double eq_h(double eps) const { return data.zeta * data.z - eps; }
        VXd eq_i(double eps) const { return data.lam * data.s - eps; }

        MMA& mma;
        int m, n;
        MXd G; // Used for solving Newton direction
        AXd Dx; // Used for solving Newton direction
        AXd dpsi_dx, plam, qlam, gvec;
        // delta x, y, z, lambda, xi, eta, mu, zeta, s, for Lagrange function
        // L = sum_{j=1}^n ((p_{0j} + sum_{i=1}^m(lambda_i * p_{ij})) / (u_j - x_j) + (q_{0j} + sum_{i=1}^m (lambda_i * q_{ij})) / (x_j - l_j))
        //      + (a_0 - zeta) * z + sum_{j=1}^n (xi_j * (alpha_j - x_j) + eta_j * (x_j - beta_j))
        //      + sum_{i=1}^m (c_i * y_i + 1/2 * d_i * y_i^2 - lambda_i * (a_i * z + y_i + b_i) - mu_i * y_i)
        struct vars {
            double z, zeta;
            AXd x, xi, eta; // n x 1
            AXd y, lam, mu, s; // m x 1
        } delta, data;
    };

    Subproblem subp;
};
#endif /* METHODOFMOVINGASYMPTOTES */
