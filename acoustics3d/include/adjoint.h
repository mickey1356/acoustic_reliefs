#pragma once

#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "constants.h"

// provide the direct/approx blocks, as well as the geometry
// void gradient(int Ne, const mat3 &Ps, const imat3 &Es, const mat3 &Cs, const mat3 &Ns,
//               const std::vector<std::pair<pii, pii>> &direct, const std::vector<std::pair<pii, pii>> &approx,
//               const MyArgs &args);

namespace bem3d {

// provide the geometry, and optionally the direct/approx blocks (if wanting an approximation)
double gradient(int Ne, int HH, const std::vector<std::vector<pii>> &elements,
                const mat3 &Ps, const imat3 &Es, const mat3 &Cs, const mat3 &Ns, const mat3 &Ls,
                bool forward_only, double freq_band, int n_freqs, bool actual,
                std::vector<cvec> &xs, std::vector<cvec> &ys, std::vector<cvec> &lmbs, std::vector<cvec> &adj_bs, vec &dcdh,
                double approx_ACA_tol, double Q_aca_tol, double solver_tol, bool recompute_matrix, bool silent,
                const std::vector<std::pair<pii, pii>> *direct = 0, const std::vector<std::pair<pii, pii>> *approx = 0);

void compute_der_A(double k, int LL, int HH,
                   const mat3 &Ps, const imat3 &Es, const mat3 &Ls,
                   const std::vector<std::vector<pii>> &elements, const cvec &vf, const cvec &x, vec &der_A, bool silent = false);

void compute_der_B(double k, int Ne, int HH,
                   const mat3 &Ps, const imat3 &Es,
                   const std::vector<std::vector<pii>> &elements, const cvec &lmb, const cvec &x, vec &der_B, bool silent = false);

void compute_der_C(double k, int HH,
                   const mat3 &Ps, const imat3 &Es,
                   const std::vector<std::vector<pii>> &elements, const cvec &lmb, vec &der_C, bool silent = false);

double compute_u(int LL, int HH, const Eigen::ArrayXd &L, const vec &yt, Eigen::ArrayXd &u);

} // namespace bem3d