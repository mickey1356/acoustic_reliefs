#pragma once

#include <complex>

#include <Eigen/Dense>

#include "constants.h"

namespace bem3d {

inline std::complex<double> G(double k, double r) {
    return std::exp(-1.0i * k * r) / (4 * PI * r);
}

inline std::complex<double> dGdr(double k, double r) {
    return std::exp(-1.0i * k * r) * (-1.0i * k * r - 1.0) / (4 * PI * r * r);
}

inline std::complex<double> kernel(double k, const vec3 &p, const vec3 &pt, const vec3 &normal) {
    auto r_vec = p - pt;
    auto r_norm = r_vec.norm();
    return dGdr(k, r_norm) * (-r_vec.dot(normal) / r_norm);
}

std::complex<double> compute_M(double k, const vec3 &p, const vec3 &qa, const vec3 &qb, const vec3 &qc, const vec3 &qn);

Eigen::RowVectorXcd compute_M_row(double k, const vec3 &p, const mat3 &qas, const mat3 &qbs, const mat3 &qcs, const mat3 &qns);

cvec compute_M_col(double k, const mat3 &ps, const vec3 &qa, const vec3 &qb, const vec3 &qc, const vec3 &qn);

cmat compute_boundary_matrix(double k, int Ne, const mat3 &Ps, const imat3 &Es, const mat3 &Cs, const mat3 &Ns);

cmat compute_listener_matrix(double k, int Ne, int L, const mat3 &Ls, const mat3 &Ps, const imat3 &Es, const mat3 &Cs, const mat3 &Ns);

Eigen::RowVectorXcd compute_ACA_row(int r, int cs, int ce, double k, const mat3 &Ps, const imat3 &Es, const mat3 &Cs, const mat3 &Ns);

cvec compute_ACA_col(int rs, int re, int c, double k, const mat3 &Ps, const imat3 &Es, const mat3 &Cs, const mat3 &Ns);

cmat compute_block(int rs, int re, int cs, int ce, double k, const mat3 &Ps, const imat3 &Es, const mat3 &Cs, const mat3 &Ns);

} // namespace bem3d