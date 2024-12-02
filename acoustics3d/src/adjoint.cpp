#include "adjoint.h"

#include <fstream>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

// iterative solvers
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include "aca.h"
#include "clustertree.h"
#include "constants.h"
#include "derivatives.h"
#include "helpers.h"
#include "hmatrix.h"
#include "integrate.h"

namespace bem3d {

double gradient(int Ne, int HH, const std::vector<std::vector<pii>> &elements,
                const mat3 &Ps, const imat3 &Es, const Eigen::RowVector3d &src_pt,
                const mat3 &Cs, const mat3 &Ns, const mat3 &Ls,
                bool forward_only, double freq_band, int n_freqs, bool actual,
                std::vector<cvec> &xs, std::vector<cvec> &ys, std::vector<cvec> &lmbs, std::vector<cvec> &adj_bs, vec &dcdh,
                double approx_ACA_tol, double Q_aca_tol, double solver_tol, bool recompute_matrix, bool silent,
                const std::vector<std::pair<pii, pii>> *direct, const std::vector<std::pair<pii, pii>> *approx) {

    if (!silent) {
        if (actual) {
            std::cout << "Computing actual values..." << std::endl;
        } else {
            std::cout << "Computing approx values..." << std::endl;
        }
    }

    // reset dcdh
    dcdh.setZero(HH);

    auto frequencies = get_frequencies(freq_band, n_freqs);
    // auto frequencies = get_frequencies_12(freq_band, n_freqs);

    int LL = Ls.rows();
    // int LL = compute_listener_pts(Ls, args.listener_ds);

    vec yt = Eigen::VectorXd::Zero(LL);

    // store the matries
    std::vector<HMatrix *> save_M;
    std::vector<std::pair<cmat, cmat>> save_Q;

    std::vector<cmat> save_M_act, save_Q_act;

    // forward pass
    if (!silent)
        std::cout << "===== Forward pass =====" << std::endl;
    for (int ff = 0; ff < n_freqs; ff++) {
        if (!silent)
            std::cout << "  ~ " << frequencies[ff] << " Hz (" << ff + 1 << " / " << n_freqs << ") ~  " << std::endl;
        double k = freq_to_wavenumber(frequencies[ff]);
        cvec G_r;
        compute_G_r(Cs, src_pt, k, G_r);
        cvec x, y_cmplx;

        if (actual) {
            if (!silent)
                std::cout << "Computing boundary matrix..." << std::endl;
            auto M = compute_boundary_matrix(k, Ne, Ps, Es, Cs, Ns);
            if (!silent)
                std::cout << "Linear solve..." << std::endl;
            x = M.colPivHouseholderQr().solve(-G_r);
            if (!silent)
                std::cout << "Computing listener matrix..." << std::endl;
            auto Q = compute_listener_matrix(k, Ne, LL, Ls, Ps, Es, Cs, Ns);
            y_cmplx = Q * x;

            if (!recompute_matrix) {
                save_M_act.push_back(M);
                save_Q_act.push_back(Q);
            }
        } else {
            if (!silent)
                std::cout << "Computing direct/approx blocks..." << std::endl;
            HMatrix *hmat = new HMatrix(Ne, !silent);
            hmat->compute_direct_and_approx_blocks_cpu(*direct, *approx, k, Ps, Es, Cs, Ns, approx_ACA_tol);

            if (!silent)
                std::cout << "\nSolving linear system..." << std::endl;
            Eigen::BiCGSTAB<HMatrix, Eigen::IdentityPreconditioner> bicgstab;
            bicgstab.setTolerance(solver_tol);
            bicgstab.compute(*hmat);
            x = bicgstab.solve(-G_r);
            if (!silent)
                std::cout << "\33[2K\r" << bem3d::mat_mults << " matrix mults, " << bicgstab.iterations() << " iters, " << bicgstab.error() << " error" << std::endl;

            if (!silent)
                std::cout << "\nComputing Q matrix approx - " << LL << " listener points..." << std::endl;
            auto Q_appx = compute_ACA_listener(LL, Ne, k, Ls, Ps, Es, Cs, Ns, Q_aca_tol, Q_ACA_MAX_K, !silent);
            y_cmplx = Q_appx.first * (Q_appx.second * x);

            if (forward_only || recompute_matrix) {
                delete hmat;
            } else {
                save_M.push_back(hmat);
                save_Q.push_back(Q_appx);
            }
        }

        vec y = y_cmplx.cwiseAbs();
        yt += y;

        // store intermediate variables
        xs.push_back(x);
        ys.push_back(y_cmplx);

        if (!silent)
            std::cout << std::endl;
    }

    // compute L = yt^2, initialize u
    Eigen::ArrayXd L = yt.array().square(), u;

    // compute u = dc/dyt = dc/dL * dL/dyt (1 x LL)
    double c = compute_u(LL, HH, L, yt, u);

    if (forward_only)
        return c;

    // backward pass
    if (!silent)
        std::cout << "===== Backward pass =====" << std::endl;
    for (int ff = 0; ff < n_freqs; ff++) {
        if (!silent)
            std::cout << "  ~ " << frequencies[ff] << " Hz (" << ff + 1 << " / " << n_freqs << ") ~  " << std::endl;
        double k = freq_to_wavenumber(frequencies[ff]);

        cvec x = xs[ff];
        cvec y_cmplx = ys[ff];

        // compute vf_real = u * dy/dycmplx, vf = vf_real[:LL] - i vf_real[LL:]
        // (- instead of + because we are now using complex numbers instead of reals)
        cvec vf = u * y_cmplx.conjugate().array() / y_cmplx.cwiseAbs().array();

        cvec adj_b, lmb;
        if (actual) {

            if (recompute_matrix) {
                // compute adjoint RHS = adj_b = Q * vf
                if (!silent)
                    std::cout << "Computing listener matrix..." << std::endl;
                auto Q = compute_listener_matrix(k, Ne, LL, Ls, Ps, Es, Cs, Ns);
                adj_b = Q.transpose() * vf;

                // solve adjoint var M lmb = adj_b
                if (!silent)
                    std::cout << "Computing boundary matrix..." << std::endl;
                auto M = compute_boundary_matrix(k, Ne, Ps, Es, Cs, Ns);

                if (!silent)
                    std::cout << "Linear solve..." << std::endl;
                lmb = M.transpose().colPivHouseholderQr().solve(adj_b);
            } else {
                adj_b = save_Q_act[ff].transpose() * vf;
                if (!silent)
                    std::cout << "Linear solve..." << std::endl;
                lmb = save_M_act[ff].transpose().colPivHouseholderQr().solve(adj_b);
            }

        } else {
            if (recompute_matrix) {
                // compute adjoint RHS = adj_b = Q * vf
                if (!silent)
                    std::cout << "Computing Q matrix approx - " << LL << " listener points..." << std::endl;
                auto Q_appx = compute_ACA_listener(LL, Ne, k, Ls, Ps, Es, Cs, Ns, Q_aca_tol, Q_ACA_MAX_K, !silent);
                adj_b = Q_appx.second.transpose() * (Q_appx.first.transpose() * vf);

                // recompute M
                if (!silent)
                    std::cout << "\nComputing direct/approx blocks..." << std::endl;
                HMatrix hmat(Ne, !silent);
                hmat.compute_direct_and_approx_blocks_cpu(*direct, *approx, k, Ps, Es, Cs, Ns, approx_ACA_tol);
                hmat.set_transposed(true);

                // solve adjoint var M lmb = adj_b
                if (!silent)
                    std::cout << "\nSolving adjoint system..." << std::endl;
                Eigen::BiCGSTAB<HMatrix, Eigen::IdentityPreconditioner> bicgstab;
                bicgstab.setTolerance(solver_tol);
                bicgstab.compute(hmat);
                lmb = bicgstab.solve(adj_b);
                if (!silent)
                    std::cout << "\33[2K\r" << bem3d::mat_mults << " matrix mults, " << bicgstab.iterations() << " iters, " << bicgstab.error() << " error" << std::endl;
            } else {
                // compute backward pass using saved matrices
                adj_b = save_Q[ff].second.transpose() * (save_Q[ff].first.transpose() * vf);

                // transpose saved M
                save_M[ff]->set_transposed(true);

                // solve adjoint var M lmb = adj_b
                if (!silent)
                    std::cout << "\nSolving adjoint system..." << std::endl;
                Eigen::BiCGSTAB<HMatrix, Eigen::IdentityPreconditioner> bicgstab;
                bicgstab.setTolerance(solver_tol);
                bicgstab.compute(*save_M[ff]);
                lmb = bicgstab.solve(adj_b);
                if (!silent)
                    std::cout << "\33[2K\r" << bem3d::mat_mults << " matrix mults, " << bicgstab.iterations() << " iters, " << bicgstab.error() << " error" << std::endl;

                // make sure to delete the HMatrices
                delete save_M[ff];
                save_M[ff] = nullptr;
            }
        }

        vec der_A = Eigen::VectorXd::Zero(HH), der_B = Eigen::VectorXd::Zero(HH), der_C = Eigen::VectorXd::Zero(HH);

        // compute A = vf dQ/dh x = sum_i^LL vf[i] * (sum_j^Ne x[j] * dQij/dh)
        compute_der_A(k, LL, HH, Ps, Es, Ls, elements, vf, x, der_A, silent);

        // compute B = lmb dM/dh x = sum_i^Ne lmb[i] * (sum_j^Ne x[j] * dMij/dh)
        compute_der_B(k, Ne, HH, Ps, Es, elements, lmb, x, der_B, silent);

        // compute C = lmb.T * db/dh
        compute_der_C(k, HH, Ps, Es, src_pt, elements, lmb, der_C, silent);

        // dc/dh = sum_F A - B + C
        dcdh += (der_A - der_B + der_C);

        // store intermediate variables
        lmbs.push_back(lmb);
        adj_bs.push_back(adj_b);

        if (!silent)
            std::cout << std::endl;
    }

    return c;
}

void compute_der_A(double k, int LL, int HH,
                   const mat3 &Ps, const imat3 &Es, const mat3 &Ls,
                   const std::vector<std::vector<pii>> &elements, const cvec &vf, const cvec &x, vec &der_A, bool silent) {

    der_A.setZero(HH);

    int done = 0;

#pragma omp parallel for
    for (int h = 0; h < HH; h++) {

        // for each element that contains this point
        for (auto ji : elements[h]) {
            // get the element index, and the pos within the element
            int j = ji.first;
            int order = ji.second;

            auto ej = Es.row(j);
            vec3 q1 = Ps.row(ej[0]);
            vec3 q2 = Ps.row(ej[1]);
            vec3 q3 = Ps.row(ej[2]);

            std::complex<double> res = 0;
            // we need to sum over all the receiver positions
            for (int i = 0; i < LL; i++) {
                if (order == 0) {
                    res += vf[i] * dMij_dq1y(Ls.row(i), Ls.row(i), Ls.row(i), q1, q2, q3, k);
                } else if (order == 1) {
                    res += vf[i] * dMij_dq2y(Ls.row(i), Ls.row(i), Ls.row(i), q1, q2, q3, k);
                } else if (order == 2) {
                    res += vf[i] * dMij_dq3y(Ls.row(i), Ls.row(i), Ls.row(i), q1, q2, q3, k);
                }
            }
            der_A[h] += (res * x[j]).real();
        }

#pragma omp critical
        {
            done++;
            if (!silent)
                std::cout << "\33[2K\rvf  * dQ/dh * x : " << done << " out of " << HH << std::flush;
        }
    }
    if (!silent)
        std::cout << std::endl;
}

void compute_der_B(double k, int Ne, int HH,
                   const mat3 &Ps, const imat3 &Es,
                   const std::vector<std::vector<pii>> &elements, const cvec &lmb, const cvec &x, vec &der_B, bool silent) {

    der_B.setZero(HH);

    int done = 0;

#pragma omp parallel for
    for (int h = 0; h < HH; h++) {

        for (auto io : elements[h]) {
            int i = io.first;
            int order = io.second;

            auto ei = Es.row(io.first);
            vec3 p1 = Ps.row(ei[0]);
            vec3 p2 = Ps.row(ei[1]);
            vec3 p3 = Ps.row(ei[2]);

            // std::complex<double> s1 = 0, s2 = 0;
            double s1r = 0, s1i = 0, s2r = 0, s2i = 0;
#pragma omp parallel for reduction(+ : s1r, s1i, s2r, s2i)
            for (int j = 0; j < Ne; j++) {
                auto ej = Es.row(j);
                vec3 q1 = Ps.row(ej[0]);
                vec3 q2 = Ps.row(ej[1]);
                vec3 q3 = Ps.row(ej[2]);

                // elements which contain h as a target (Ci)
                auto tmp1 = x[j] * dMij_dpy(p1, p2, p3, q1, q2, q3, k);
                s1r += tmp1.real();
                s1i += tmp1.imag();
                // s1 += x[j] * dMij_dpy(p1, p2, p3, q1, q2, q3, k);

                // elements which contain h as a source (e_i)
                std::complex<double> tmp2;
                if (order == 0) {
                    tmp2 = lmb[j] * dMij_dq1y(q1, q2, q3, p1, p2, p3, k);
                } else if (order == 1) {
                    tmp2 = lmb[j] * dMij_dq2y(q1, q2, q3, p1, p2, p3, k);
                } else if (order == 2) {
                    tmp2 = lmb[j] * dMij_dq3y(q1, q2, q3, p1, p2, p3, k);
                }

                s2r += tmp2.real();
                s2i += tmp2.imag();
            }
            der_B[h] += (std::complex<double>(s1r, s1i) * lmb[i]).real() + (std::complex<double>(s2r, s2i) * x[i]).real();
        }

#pragma omp critical
        {
            done++;

            if (!silent)
                std::cout << "\33[2K\rlmb * dM/dh * x : " << done << " out of " << HH << std::flush;
        }
    }
    if (!silent)
        std::cout << std::endl;
}

void compute_der_C(double k, int HH,
                   const mat3 &Ps, const imat3 &Es, const Eigen::RowVector3d &src_pt,
                   const std::vector<std::vector<pii>> &elements, const cvec &lmb, vec &der_C, bool silent) {

    der_C.setZero(HH);

    int done = 0;

#pragma omp parallel for
    for (int h = 0; h < HH; h++) {

        for (auto io : elements[h]) {
            // get the element index, and the pos within the element
            int i = io.first;
            auto ei = Es.row(i);
            vec3 p1 = Ps.row(ei[0]);
            vec3 p2 = Ps.row(ei[1]);
            vec3 p3 = Ps.row(ei[2]);

            der_C[h] -= (dbi_dpy(p1, p2, p3, src_pt, k) * lmb[i]).real();
        }

#pragma omp critical
        {
            done++;

            if (!silent)
                std::cout << "\33[2K\rlmb * db/dh     : " << done << " out of " << HH << std::flush;
        }
    }
    if (!silent)
        std::cout << std::endl;
}

double compute_u(int LL, int HH, const Eigen::ArrayXd &L, const vec &yt, Eigen::ArrayXd &u) {
    double L_sum = L.sum();
    double L_sqr_of_sum = pow(L_sum, 2);
    double L_sum_of_sqr = L.square().sum();
    double c = (L_sqr_of_sum - L_sum_of_sqr) / ((LL - 1) * L_sum_of_sqr);

    // compute u = dc/dyt = dc/dL * dL/dyt (1 x LL)
    u = 4 * (L_sum * L_sum_of_sqr - L * L_sqr_of_sum) / ((LL - 1) * L_sum_of_sqr * L_sum_of_sqr) * yt.array();
    return c;
}

} // namespace bem3d