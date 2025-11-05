#include "acoustics3d.h"

#include <iostream>

#include <nanobind/nanobind.h>

#include <nanobind/eigen/dense.h>

#include <nanobind/stl/pair.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>

#include "adjoint.h"
#include "clustertree.h"
#include "helpers.h"
#include "sphere.h"

namespace nb = nanobind;
using namespace nb::literals;

DiffBEM::DiffBEM(int cluster_size, double radius_factor, const std::vector<double> &freq_bands, int n_freqs, double approx_ACA_tol, double Q_ACA_tol, double solver_tol, const Eigen::RowVector3d &src_pt, double listener_radius, double listener_ds, bool recompute_matrices) {
    _cluster_size = cluster_size;
    _radius_factor = radius_factor;
    _freq_bands = freq_bands;
    _n_freqs = n_freqs;
    _approx_ACA_tol = approx_ACA_tol;
    _Q_ACA_tol = Q_ACA_tol;
    _solver_tol = solver_tol;
    _src_pt = src_pt;
    _listener_radius = listener_radius;
    _listener_ds = listener_ds;
    _recompute_matrices = recompute_matrices;
}

std::pair<bem3d::mat3, bem3d::imat3> DiffBEM::set_mesh(const bem3d::mat3 &Ps, const bem3d::imat3 &Es) {
    bem3d::imat3 Es_tmp = Es;
    _Ps = Ps;
    _Ne = Es.rows();

    std::cout << _Ps.rows() << " vertices, " << _Ne << " elements" << std::endl;

    bem3d::mat3 Cs;
    bem3d::compute_centers(_Ps, Es_tmp, Cs);

    // construct tree
    bem3d::ClusterTree tree(_cluster_size, Cs);
    _Es = Es_tmp(tree.reordering, Eigen::placeholders::all);
    bem3d::compute_centers(_Ps, _Es, Cs);

    // check tree to make sure it is valid
    auto tree_ok = (tree.check(Cs) ? "tree ok" : "tree bad");
    std::cout << tree_ok << std::endl;
    // compute direct/approx pairs
    _direct.clear();
    _approx.clear();
    std::cout << "Finding direct/approx pairs..." << std::endl;
    tree.get_direct_and_approx_blocks(_radius_factor, _direct, _approx);
    std::cout << _direct.size() << " direct, " << _approx.size() << " approx\n"
              << std::endl;

    // compute listener points
    bem3d::compute_listener_pts(_Ls, _listener_radius, _listener_ds);

    _mesh_set = true;
    return std::make_pair(_Ps, _Es);
}

void DiffBEM::set_diff_points(const std::vector<int> &diff_pts) {
    // set up the differentiable points
    _Hs.clear();

    _HH = 0;
    for (auto i : diff_pts) {
        _Hs[i] = _HH;
        _HH++;
    }

    // determine, for each differentiable point, the elements that contain that point
    _elements.clear();
    _elements.resize(_HH);

    for (int j = 0; j < _Ne; j++) {
        auto ej = _Es.row(j);
        int qi1 = ej[0], qi2 = ej[1], qi3 = ej[2];

        if (_Hs.find(qi1) != _Hs.end()) {
            _elements[_Hs.at(qi1)].push_back({j, 0});
        }

        if (_Hs.find(qi2) != _Hs.end()) {
            _elements[_Hs.at(qi2)].push_back({j, 1});
        }

        if (_Hs.find(qi3) != _Hs.end()) {
            _elements[_Hs.at(qi3)].push_back({j, 2});
        }
    }

    _diff_set = true;
}

std::pair<bem3d::mat3, bem3d::imat3> DiffBEM::precompute(const bem3d::mat3 &Ps, const bem3d::imat3 &Es, const std::vector<int> &diff_pts) {
    set_mesh(Ps, Es);
    set_diff_points(diff_pts);
    return std::make_pair(_Ps, _Es);
}

double DiffBEM::value() {
    if (!_mesh_set) {
        std::cerr << "Error! No mesh set. Set one using set_mesh or precompute." << std::endl;
        return -1;
    }

    // compute intermediate values
    bem3d::mat3 Cs, Ns;
    bem3d::compute_intermediates(_Ps, _Es, Cs, Ns);

    double c = 0;
    bem3d::vec dcdh;
    std::vector<bem3d::cvec> xs, ys, lmbs, adj_bs;

    for (double freq_band : _freq_bands) {
        c += bem3d::gradient(_Ne, _HH, _elements, _Ps, _Es, _src_pt, Cs, Ns, _Ls, true, freq_band, _n_freqs, use_actual, xs, ys, lmbs, adj_bs, dcdh, _approx_ACA_tol, _Q_ACA_tol, _solver_tol, _recompute_matrices, silent, &_direct, &_approx);
    }

    // average gradients over freq bands
    c /= _freq_bands.size();
    return c;
}

double DiffBEM::value(const bem3d::vec &x) {
    if (!_mesh_set) {
        std::cerr << "Error! No mesh set. Set one using set_mesh or precompute." << std::endl;
        return 0;
    }

    // compute actual Ps (x is a heightfield over the differentiable points)
    bem3d::mat3 Ps = _Ps;
    for (const auto &hh : _Hs) {
        Ps(hh.first, 1) = x[hh.second] + _Ps(hh.first, 1);
    }

    // compute intermediate values
    bem3d::mat3 Cs, Ns;
    bem3d::compute_intermediates(Ps, _Es, Cs, Ns);

    double c = 0;
    bem3d::vec dcdh;
    std::vector<bem3d::cvec> xs, ys, lmbs, adj_bs;

    for (double freq_band : _freq_bands) {
        c += bem3d::gradient(_Ne, _HH, _elements, Ps, _Es, _src_pt, Cs, Ns, _Ls, true, freq_band, _n_freqs, use_actual, xs, ys, lmbs, adj_bs, dcdh, _approx_ACA_tol, _Q_ACA_tol, _solver_tol, _recompute_matrices, silent, &_direct, &_approx);
    }

    // average gradients over freq bands
    c /= _freq_bands.size();

    return c;
}

double DiffBEM::band_value(double freq_band) {
    if (!_mesh_set) {
        std::cerr << "Error! No mesh set. Set one using set_mesh or precompute." << std::endl;
        return -1;
    }

    bem3d::mat3 Cs, Ns;
    bem3d::compute_intermediates(_Ps, _Es, Cs, Ns);

    bem3d::vec dcdh;
    std::vector<bem3d::cvec> xs, ys, lmbs, adj_bs;

    double c = bem3d::gradient(_Ne, _HH, _elements, _Ps, _Es, _src_pt, Cs, Ns, _Ls, true, freq_band, _n_freqs, use_actual, xs, ys, lmbs, adj_bs, dcdh, _approx_ACA_tol, _Q_ACA_tol, _solver_tol, _recompute_matrices, silent, &_direct, &_approx);
    return c;
}

std::vector<double> DiffBEM::values() {
    if (!_mesh_set) {
        std::cerr << "Error! No mesh set. Set one using set_mesh or precompute." << std::endl;
        return std::vector<double>();
    }

    bem3d::mat3 Cs, Ns;
    bem3d::compute_intermediates(_Ps, _Es, Cs, Ns);

    bem3d::vec dcdh;
    std::vector<bem3d::cvec> xs, ys, lmbs, adj_bs;
    std::vector<double> coeffs;

    for (double freq_band : _freq_bands) {
        double c = bem3d::gradient(_Ne, _HH, _elements, _Ps, _Es, _src_pt, Cs, Ns, _Ls, true, freq_band, _n_freqs, use_actual, xs, ys, lmbs, adj_bs, dcdh, _approx_ACA_tol, _Q_ACA_tol, _solver_tol, _recompute_matrices, silent, &_direct, &_approx);
        coeffs.push_back(c);
    }

    return coeffs;
}

std::pair<double, bem3d::vec> DiffBEM::gradient(const bem3d::vec &x) {
    if (!_mesh_set) {
        std::cerr << "Error! No mesh set. Set one using set_mesh or precompute." << std::endl;
        return std::pair<double, bem3d::vec>();
    }
    if (!_diff_set) {
        std::cerr << "Error! No differentiable points set. Set them using set_diff_pts or precompute." << std::endl;
        return std::make_pair(-1, bem3d::vec());
    }

    // compute actual Ps (x is a heightfield over the differentiable points)
    bem3d::mat3 Ps = _Ps;
    for (const auto &hh : _Hs) {
        Ps(hh.first, 1) = x[hh.second] + _Ps(hh.first, 1);
    }

    // compute intermediate values
    bem3d::mat3 Cs, Ns;
    bem3d::compute_intermediates(Ps, _Es, Cs, Ns);

    double c = 0;
    bem3d::vec d_grad, dcdh;
    std::vector<bem3d::cvec> xs, ys, lmbs, adj_bs;

    d_grad.setZero(_HH);
    for (double freq_band : _freq_bands) {
        c += bem3d::gradient(_Ne, _HH, _elements, Ps, _Es, _src_pt, Cs, Ns, _Ls, false, freq_band, _n_freqs, use_actual, xs, ys, lmbs, adj_bs, dcdh, _approx_ACA_tol, _Q_ACA_tol, _solver_tol, _recompute_matrices, silent, &_direct, &_approx);
        d_grad += dcdh;
    }

    // average gradients over freq bands
    // c /= _freq_bands.size();
    // d_grad /= _freq_bands.size();

    return std::make_pair(c, d_grad);
}

bem3d::cvec DiffBEM::pvals(double frequency) {
    return pvals(frequency, _Ls);
}

bem3d::cvec DiffBEM::pvals(double frequency, const bem3d::mat3 &listeners) {
    if (!_mesh_set) {
        std::cerr << "Error! No mesh set. Set one using set_mesh or precompute." << std::endl;
        return bem3d::cvec();
    }

    int LL = listeners.rows();

    bem3d::mat3 Cs, Ns;
    bem3d::compute_intermediates(_Ps, _Es, Cs, Ns);

    double k = freq_to_wavenumber(frequency);
    bem3d::cvec G_r;
    compute_G_r(Cs, _src_pt, k, G_r);
    bem3d::cvec x, y_cmplx;

    if (!silent) {
        std::cout << "===== Forward pass =====" << std::endl;
        std::cout << "Computing direct/approx blocks..." << std::endl;
    }

    HMatrix *hmat = new HMatrix(_Ne, !silent);
    hmat->compute_direct_and_approx_blocks_cpu(_direct, _approx, k, _Ps, _Es, Cs, Ns, _approx_ACA_tol);

    if (!silent)
        std::cout << "\nSolving linear system..." << std::endl;
    Eigen::BiCGSTAB<HMatrix, Eigen::IdentityPreconditioner> bicgstab;
    bicgstab.setTolerance(_solver_tol);
    bicgstab.compute(*hmat);
    x = bicgstab.solve(-G_r);
    if (!silent)
        std::cout << "\33[2K\r" << bem3d::mat_mults << " matrix mults, " << bicgstab.iterations() << " iters, " << bicgstab.error() << " error" << std::endl;

    if (!silent)
        std::cout << "\nComputing Q matrix approx - " << LL << " listener points..." << std::endl;
    auto Q_appx = compute_ACA_listener(LL, _Ne, k, listeners, _Ps, _Es, Cs, Ns, _Q_ACA_tol, Q_ACA_MAX_K, !silent);
    y_cmplx = Q_appx.first * (Q_appx.second * x);

    delete hmat;
    return y_cmplx;
}

bem3d::cvec DiffBEM::surface_vals(double frequency) {
    if (!_mesh_set) {
        std::cerr << "Error! No mesh set. Set one using set_mesh or precompute." << std::endl;
        return bem3d::cvec();
    }

    bem3d::mat3 Cs, Ns;
    bem3d::compute_intermediates(_Ps, _Es, Cs, Ns);

    double k = freq_to_wavenumber(frequency);
    bem3d::cvec G_r;
    compute_G_r(Cs, _src_pt, k, G_r);
    bem3d::cvec x, y_cmplx;

    if (!silent) {
        std::cout << "===== Forward pass =====" << std::endl;
        std::cout << "Computing direct/approx blocks..." << std::endl;
    }

    HMatrix *hmat = new HMatrix(_Ne, !silent);
    hmat->compute_direct_and_approx_blocks_cpu(_direct, _approx, k, _Ps, _Es, Cs, Ns, _approx_ACA_tol);

    if (!silent)
        std::cout << "\nSolving linear system..." << std::endl;
    Eigen::BiCGSTAB<HMatrix, Eigen::IdentityPreconditioner> bicgstab;
    bicgstab.setTolerance(_solver_tol);
    bicgstab.compute(*hmat);
    x = bicgstab.solve(-G_r);
    if (!silent)
        std::cout << "\33[2K\r" << bem3d::mat_mults << " matrix mults, " << bicgstab.iterations() << " iters, " << bicgstab.error() << " error" << std::endl;

    delete hmat;
    return x;
}

bem3d::mat3 DiffBEM::get_listeners() {
    return _Ls;
}

std::pair<bem3d::mat3, bem3d::imat3> DiffBEM::get_mesh() {
    if (!_mesh_set) {
        std::cout << "Warning! No mesh set. Behavior might not be as expected." << std::endl;
    }

    return std::make_pair(_Ps, _Es);
}

std::pair<bem3d::mat3, bem3d::imat3> DiffBEM::get_mesh(const bem3d::vec &x) {
    if (!_mesh_set) {
        std::cout << "Warning! No mesh set. Behavior might not be as expected." << std::endl;
    }
    if (!_diff_set) {
        std::cout << "Warning! No diff points set. Behavior might not be as expected." << std::endl;
    }

    bem3d::mat3 Ps = _Ps;
    for (const auto &hh : _Hs) {
        Ps(hh.first, 1) = x[hh.second] + _Ps(hh.first, 1);
    }
    return std::make_pair(Ps, _Es);
}

std::unordered_map<int, int> DiffBEM::get_Hs() {
    return _Hs;
}

void DiffBEM::set_band(double freq_band) {
    std::vector<double>().swap(_freq_bands);
    _freq_bands.push_back(freq_band);
}

NB_MODULE(acoustics3d, m) {
    nb::class_<DiffBEM>(m, "DiffBEM")
        .def(nb::init<int, double, const std::vector<double> &, int, double, double, double, const Eigen::RowVector3d &, double, double, bool>(),
             "cluster_size"_a = 64, "radius_factor"_a = 1.5, "freq_bands"_a = std::vector<int>{1000}, "n_freqs"_a = 1,
             "approx_ACA_tol"_a = 1e-3, "Q_ACA_tol"_a = 1e-3, "solver_tol"_a = 1e-3, "src_pt"_a = Eigen::RowVector3d(0, 100, 0), "listener_radius"_a = 50, "listener_ds"_a = 5, "recompute_matrices"_a = false)
        .def("set_mesh", &DiffBEM::set_mesh, "Ps"_a, "Es"_a)
        .def("set_diff_points", &DiffBEM::set_diff_points, "diff_pts"_a)
        .def("precompute", &DiffBEM::precompute, "Ps"_a, "Es"_a, "diff_pts"_a)

        .def("value", nb::overload_cast<>(&DiffBEM::value))
        .def("value", nb::overload_cast<const bem3d::vec &>(&DiffBEM::value), "x"_a)
        .def("band_value", &DiffBEM::band_value, "freq_band"_a)
        .def("values", &DiffBEM::values)
        .def("gradient", &DiffBEM::gradient, "x"_a)

        .def("pvals", nb::overload_cast<double>(&DiffBEM::pvals), "frequency"_a)
        .def("pvals", nb::overload_cast<double, const bem3d::mat3 &>(&DiffBEM::pvals), "frequency"_a, "listeners"_a)
        .def("get_listeners", &DiffBEM::get_listeners)
        .def("surface_vals", &DiffBEM::surface_vals, "frequency"_a)

        .def("get_mesh", nb::overload_cast<>(&DiffBEM::get_mesh))
        .def("get_mesh", nb::overload_cast<const bem3d::vec &>(&DiffBEM::get_mesh), "x"_a)

        .def("get_Hs", &DiffBEM::get_Hs)
        .def("set_band", &DiffBEM::set_band, "freq_band"_a)

        .def_rw("silent", &DiffBEM::silent)
        .def_rw("use_actual", &DiffBEM::use_actual);

    m.def("sphere", &bem3d::sphere_test::sphere, "sphere_mesh"_a, "freq"_a, "LL"_a = 100, "lrad"_a = 10, "actual"_a = false);
}
