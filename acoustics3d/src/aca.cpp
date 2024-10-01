#include "aca.h"

#include <algorithm>
#include <iostream>

#include "integrate.h"

namespace bem3d {
int argmax_abs_masked(const cvec &v, const std::unordered_set<int> &used_indices) {
    int idx = 0, max_idx = 0;
    double max_val = -1;
    for (auto val : v.cwiseAbs2()) {
        // if the current index isn't in used_indices
        if (used_indices.find(idx) == used_indices.end()) {
            if (val > max_val) {
                max_val = val;
                max_idx = idx;
            }
        }
        idx++;
    }
    return max_idx;
}

int argmax_abs(const cvec &v) {
    int idx = 0, max_idx = 0;
    double max_val = -1;
    for (auto val : v.cwiseAbs2()) {
        if (val > max_val) {
            max_val = val;
            max_idx = idx;
        }
        idx++;
    }
    return max_idx;
}

std::pair<cmat, cmat> compute_ACA_block(int rs, int re, int cs, int ce, double k,
                                        const mat3 &Ps, const imat3 &Es,
                                        const mat3 &Cs, const mat3 &Ns,
                                        double eps, int max_iter) {
    int M = re - rs;
    int N = ce - cs;
    int iter_lim = std::min(max_iter, std::min(M - 1, N - 1));
    cmat U = cmat::Zero(M, iter_lim);
    cmat V = cmat::Zero(iter_lim, N);
    int i = 0, iter = 0;
    std::unordered_set<int> used_I{0};
    double err = 0;
    double un_sq = 0, vn_sq = 0, dots = 0;

    while (iter < iter_lim) {
        auto r_v = compute_ACA_row(i + rs, cs, ce, k, Ps, Es, Cs, Ns);
        auto v = r_v - U.row(i) * V;
        int j = argmax_abs(v);
        if (v[j] == 0.0) {
            V.row(iter) = v;
        } else {
            V.row(iter) = v / v[j];
        }
        auto c_v = compute_ACA_col(rs, re, j + cs, k, Ps, Es, Cs, Ns);
        auto u = c_v - U * V.col(j);
        U.col(iter) = u;
        vn_sq = V.row(iter).cwiseAbs2().sum();
        un_sq = U.col(iter).cwiseAbs2().sum();
        auto Udots = (U.col(iter).transpose() * U(all, Eigen::seqN(0, iter))).cwiseAbs();
        auto Vdots = (V.row(iter) * V(Eigen::seqN(0, iter), all).transpose()).cwiseAbs();
        dots = Udots.dot(Vdots);
        err += un_sq * vn_sq + 2 * dots;

        if (sqrt(un_sq * vn_sq / err) <= eps) break;

        i = argmax_abs_masked(U.col(iter), used_I);
        used_I.insert(i);
        iter++;
    }

    cmat U_ret = U(all, Eigen::seqN(0, iter));
    cmat V_ret = V(Eigen::seqN(0, iter), all);

    return std::make_pair(U_ret, V_ret);
}

std::pair<cmat, cmat> compute_ACA_listener(int L, int Ne, double k,
                                           const mat3 &Ls, const mat3 &Ps, const imat3 &Es,
                                           const mat3 &Cs, const mat3 &Ns,
                                           double eps, int max_iter, bool verbose) {
    int iter_lim = std::min(max_iter, std::min(L - 1, Ne - 1));
    cmat U = cmat::Zero(L, iter_lim);
    cmat V = cmat::Zero(iter_lim, Ne);
    int i = 0, iter = 0;
    std::unordered_set<int> used_I{0};
    double err = 0;
    double un_sq = 0, vn_sq = 0, dots = 0;

    auto qas = Ps(Es.col(0), all);
    auto qbs = Ps(Es.col(1), all);
    auto qcs = Ps(Es.col(2), all);

    while (iter < iter_lim) {
        auto r_v = compute_M_row(k, Ls.row(i), qas, qbs, qcs, Ns);
        auto v = r_v - U.row(i) * V;
        int j = argmax_abs(v);
        V.row(iter) = v / v[j];

        auto qa = Ps(Es.row(j)[0], all);
        auto qb = Ps(Es.row(j)[1], all);
        auto qc = Ps(Es.row(j)[2], all);
        auto qn = Ns.row(j);

        auto c_v = compute_M_col(k, Ls, qa, qb, qc, qn);
        auto u = c_v - U * V.col(j);
        U.col(iter) = u;
        vn_sq = V.row(iter).cwiseAbs2().sum();
        un_sq = U.col(iter).cwiseAbs2().sum();
        auto Udots = (U.col(iter).transpose() * U(all, Eigen::seqN(0, iter))).cwiseAbs();
        auto Vdots = (V.row(iter) * V(Eigen::seqN(0, iter), all).transpose()).cwiseAbs();
        dots = Udots.dot(Vdots);
        err += un_sq * vn_sq + 2 * dots;

        if (sqrt(un_sq * vn_sq / err) <= eps) break;

        i = argmax_abs_masked(U.col(iter), used_I);
        used_I.insert(i);
        iter++;
        if (verbose) std::cout << "\33[2K\riter " << iter << " out of " << max_iter << ", err: " << sqrt(un_sq * vn_sq / err) << std::flush;
    }
    if (verbose) std::cout << std::endl;

    cmat U_ret = U(all, Eigen::seqN(0, iter));
    cmat V_ret = V(Eigen::seqN(0, iter), all);

    return std::make_pair(U_ret, V_ret);
}
} // namespace bem3d