#include "integrate.h"

namespace bem3d {

std::complex<double> compute_M(double k, const vec3 &p, const vec3 &qa, const vec3 &qb, const vec3 &qc, const vec3 &qn) {
    auto ab = qb - qa;
    auto ac = qc - qa;
    std::complex<double> res = 0;
    for (int i = 0; i < 6; i++) {
        auto pt = qa + QUAD_WEIGHTS_AND_POINTS[i][1] * ab + QUAD_WEIGHTS_AND_POINTS[i][2] * ac;
        res += QUAD_WEIGHTS_AND_POINTS[i][0] * kernel(k, p, pt, qn);
    }
    return res * 0.5;
}

Eigen::RowVectorXcd compute_M_row(double k, const vec3 &p, const mat3 &qas, const mat3 &qbs, const mat3 &qcs, const mat3 &qns) {
    Eigen::RowVectorXcd result(qas.rows());
    for (int i = 0; i < qas.rows(); i++) {
        result[i] = compute_M(k, p, qas.row(i), qbs.row(i), qcs.row(i), qns.row(i));
    }
    return result;
}

cvec compute_M_col(double k, const mat3 &ps, const vec3 &qa, const vec3 &qb, const vec3 &qc, const vec3 &qn) {
    cvec result(ps.rows());
    for (int i = 0; i < ps.rows(); i++) {
        result[i] = compute_M(k, ps.row(i), qa, qb, qc, qn);
    }
    return result;
}

cmat compute_boundary_matrix(double k, int Ne, const mat3 &Ps, const imat3 &Es, const mat3 &Cs, const mat3 &Ns) {
    cmat M(Ne, Ne);
    for (int i = 0; i < Ne; i++) {
        auto qa = Ps(Es.row(i)[0], all);
        auto qb = Ps(Es.row(i)[1], all);
        auto qc = Ps(Es.row(i)[2], all);
        auto qn = Ns.row(i);

        M.col(i) = compute_M_col(k, Cs, qa, qb, qc, qn);
    }
    M -= 0.5 * cmat::Identity(Ne, Ne);
    return M;
}

cmat compute_listener_matrix(double k, int Ne, int L, const mat3 &Ls, const mat3 &Ps, const imat3 &Es, const mat3 &Cs, const mat3 &Ns) {
    cmat Q(L, Ne);
    for (int i = 0; i < Ne; i++) {
        auto qa = Ps(Es.row(i)[0], all);
        auto qb = Ps(Es.row(i)[1], all);
        auto qc = Ps(Es.row(i)[2], all);
        auto qn = Ns.row(i);

        Q.col(i) = compute_M_col(k, Ls, qa, qb, qc, qn);
    }
    return Q;
}

Eigen::RowVectorXcd compute_ACA_row(int r, int cs, int ce, double k, const mat3 &Ps, const imat3 &Es, const mat3 &Cs, const mat3 &Ns) {
    Eigen::RowVectorXcd result(ce - cs);
    auto p = Cs.row(r);
    for (int i = cs; i < ce; i++) {
        auto qa = Ps(Es.row(i)[0], all);
        auto qb = Ps(Es.row(i)[1], all);
        auto qc = Ps(Es.row(i)[2], all);
        auto qn = Ns.row(i);

        result[i - cs] = compute_M(k, p, qa, qb, qc, qn);
    }
    return result;
}

cvec compute_ACA_col(int rs, int re, int c, double k, const mat3 &Ps, const imat3 &Es, const mat3 &Cs, const mat3 &Ns) {
    cvec result(re - rs);

    auto qa = Ps(Es.row(c)[0], all);
    auto qb = Ps(Es.row(c)[1], all);
    auto qc = Ps(Es.row(c)[2], all);
    auto qn = Ns.row(c);

    for (int i = rs; i < re; i++) {
        result[i - rs] = compute_M(k, Cs.row(i), qa, qb, qc, qn);
    }
    return result;
}

cmat compute_block(int rs, int re, int cs, int ce, double k, const mat3 &Ps, const imat3 &Es, const mat3 &Cs, const mat3 &Ns) {
    cmat block(re - rs, ce - cs);
    for (int i = cs; i < ce; i++) {
        auto qa = Ps(Es.row(i)[0], all);
        auto qb = Ps(Es.row(i)[1], all);
        auto qc = Ps(Es.row(i)[2], all);
        auto qn = Ns.row(i);

        auto ps = Cs(Eigen::seq(rs, re - 1), all);

        block.col(i - cs) = compute_M_col(k, ps, qa, qb, qc, qn);
    }
    return block;
}

} // namespace bem3d