#include <fstream>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include "aca.h"
#include "adjoint.h"
#include "clustertree.h"
#include "constants.h"
#include "helpers.h"
#include "hmatrix.h"
#include "integrate.h"
#include "read_obj.h"

#include <nanobind/stl/string.h>

namespace bem3d::sphere_test {
void compute_b(const mat3 &Cs, double k, cvec &b) {
    b.setZero(Cs.rows());
    auto x = Cs.col(0).array();
    b = (-1.0i * k * x).exp();
}

bem3d::cvec sphere(std::string sphere_mesh, double freq, int LL, double lrad, bool actual) {
    using namespace bem3d;

    double k = freq_to_wavenumber(freq);
    double tols = 1e-10;

    mat3 Ps, Cs, Ns;
    imat3 Es_tmp, Es;

    std::vector<std::pair<pii, pii>> direct, approx;

    int Ne = read_obj(sphere_mesh, Ps, Es_tmp);
    std::cout << Ps.rows() << " vertices, " << Ne << " elements" << std::endl;
    compute_centers(Ps, Es_tmp, Cs);

    ClusterTree tree(128, Cs);
    Es = Es_tmp(tree.reordering, all);
    compute_intermediates(Ps, Es, Cs, Ns);

    auto tree_ok = (tree.check(Cs) ? "tree ok" : "tree bad");
    std::cout << tree_ok << std::endl;

    std::cout << "Finding direct/approx pairs..." << std::endl;
    tree.get_direct_and_approx_blocks(1.5, direct, approx);
    std::cout << direct.size() << " direct, " << approx.size() << " approx\n"
              << std::endl;

    cvec b, y_cmplx;
    compute_b(Cs, k, b);

    mat3 Ls = mat3::Zero(LL, 3);
    for (int l = 0; l < LL; l++) {
        double theta = (double)l / (double)LL * 2 * PI;
        Ls(l, 0) = lrad * std::cos(theta);
        Ls(l, 2) = -lrad * std::sin(theta);
    }

    if (actual) {
        auto M = compute_boundary_matrix(k, Ne, Ps, Es, Cs, Ns);
        cvec x = M.colPivHouseholderQr().solve(-b);
        auto Q = compute_listener_matrix(k, Ne, LL, Ls, Ps, Es, Cs, Ns);
        y_cmplx = Q * x;
    } else {
        HMatrix hmat(Ne, true);
        hmat.compute_direct_and_approx_blocks_cpu(direct, approx, k, Ps, Es, Cs, Ns, tols);

        Eigen::BiCGSTAB<HMatrix, Eigen::IdentityPreconditioner> bicgstab;
        bicgstab.setTolerance(tols);
        bicgstab.compute(hmat);
        cvec x = bicgstab.solve(-b);
        std::cout << "\33[2K\r" << bem3d::mat_mults << " matrix mults, " << bicgstab.iterations() << " iters, " << bicgstab.error() << " error" << std::endl;

        auto Q_appx = compute_ACA_listener(LL, Ne, k, Ls, Ps, Es, Cs, Ns, tols, Q_ACA_MAX_K, true);
        y_cmplx = Q_appx.first * (Q_appx.second * x);
    }

    return y_cmplx;
}
} // namespace bem3d::sphere_test