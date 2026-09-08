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
#include "profiler.h"
#include "sphere.h"

#ifdef BEM_USE_CUDA
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include <Eigen/IterativeLinearSolvers>

#include "cuda/adjoint_gpu.h"
#include "cuda/gpu_common.h"
#include "derivatives.h"
#include "hmatrix.h"
#include "integrate.h"
#endif

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

DiffBEM::~DiffBEM() {
#ifdef BEM_USE_CUDA
    if (_dev)
        bem3d::cuda::device_state_destroy(_dev);
#endif
}

// Flatten the static geometry + incidence "task list" and upload to the device.
// No-op unless built with BEM_USE_CUDA. Called at the end of precompute().
void DiffBEM::upload_device_static() {
#ifdef BEM_USE_CUDA
    if (_dev) {
        bem3d::cuda::device_state_destroy(_dev);
        _dev = nullptr;
    }
    if (!_mesh_set || !_diff_set)
        return;

    int Ne = _Ne;
    int LL = static_cast<int>(_Ls.rows());
    int Nv = static_cast<int>(_Ps.rows());
    int HH = _HH;

    // flatten _elements (per diff-point incident (element, local-order)) into tasks
    std::vector<int> task_h, task_elem, task_order;
    for (int h = 0; h < HH; h++) {
        for (const auto &io : _elements[h]) {
            task_h.push_back(h);
            task_elem.push_back(io.first);
            task_order.push_back(io.second);
        }
    }
    int T = static_cast<int>(task_h.size());

    _dev = bem3d::cuda::device_state_create();
    // _Es (imat3) and _Ls (mat3) are RowMajor and contiguous -> .data() is flat.
    bem3d::cuda::device_set_static(_dev, _Es.data(), Ne, _Ls.data(), LL,
                                   task_h.data(), task_elem.data(), task_order.data(),
                                   T, HH, Nv);

    // GPU-resident H-matrix assembly metadata (block dims + per-block max ranks).
    // Block index lists (_direct/_approx) are static; ranks are adaptive (<=maxrank).
    int nD = static_cast<int>(_direct.size());
    std::vector<int> Dmeta(nD * 4);
    std::vector<long> Doff(nD);
    long Dtot = 0;
    for (int i = 0; i < nD; i++) {
        int rs = _direct[i].first.first, re = _direct[i].first.second;
        int cs = _direct[i].second.first, ce = _direct[i].second.second;
        Dmeta[4 * i + 0] = rs;
        Dmeta[4 * i + 1] = re;
        Dmeta[4 * i + 2] = cs;
        Dmeta[4 * i + 3] = ce;
        Doff[i] = Dtot;
        Dtot += (long)(re - rs) * (ce - cs);
    }
    int nA = static_cast<int>(_approx.size());
    std::vector<int> Ameta(nA * 5);
    std::vector<long> Uoff(nA), Voff(nA);
    long Utot = 0, Vtot = 0;
    int gmaxrank = 0;
    for (int i = 0; i < nA; i++) {
        int rs = _approx[i].first.first, re = _approx[i].first.second;
        int cs = _approx[i].second.first, ce = _approx[i].second.second;
        int M = re - rs, N = ce - cs;
        int mr = std::min(APPROX_ACA_MAX_K, std::min(M - 1, N - 1));
        if (mr < 1)
            mr = 1;
        gmaxrank = std::max(gmaxrank, mr);
        Ameta[5 * i + 0] = rs;
        Ameta[5 * i + 1] = re;
        Ameta[5 * i + 2] = cs;
        Ameta[5 * i + 3] = ce;
        Ameta[5 * i + 4] = mr; // iter_lim / max rank for this block
        Uoff[i] = Utot;
        Voff[i] = Vtot;
        Utot += (long)M * mr;
        Vtot += (long)mr * N;
    }
    bem3d::cuda::device_setup_assembly(_dev, Dmeta.data(), Doff.data(), Dtot, nD,
                                       Ameta.data(), Uoff.data(), Voff.data(), Utot, Vtot, nA, gmaxrank);
#endif
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
    upload_device_static();
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
#ifdef BEM_USE_CUDA
    if (_dev) {
        bem3d::cuda::device_set_fp32(_dev, gpu_fp32);
        bem3d::cuda::device_set_warm(_dev, warm_start);
        bem3d::cuda::device_set_precond(_dev, use_precond);
    }
#endif

    for (double freq_band : _freq_bands) {
        c += bem3d::gradient(_Ne, _HH, _elements, _Ps, _Es, _src_pt, Cs, Ns, _Ls, true, freq_band, _n_freqs, use_actual, xs, ys, lmbs, adj_bs, dcdh, _approx_ACA_tol, _Q_ACA_tol, _solver_tol, _recompute_matrices, silent, &_direct, &_approx, use_gpu, _dev);
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
#ifdef BEM_USE_CUDA
    if (_dev) {
        bem3d::cuda::device_set_fp32(_dev, gpu_fp32);
        bem3d::cuda::device_set_warm(_dev, warm_start);
        bem3d::cuda::device_set_precond(_dev, use_precond);
    }
#endif

    for (double freq_band : _freq_bands) {
        c += bem3d::gradient(_Ne, _HH, _elements, Ps, _Es, _src_pt, Cs, Ns, _Ls, true, freq_band, _n_freqs, use_actual, xs, ys, lmbs, adj_bs, dcdh, _approx_ACA_tol, _Q_ACA_tol, _solver_tol, _recompute_matrices, silent, &_direct, &_approx, use_gpu, _dev);
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
#ifdef BEM_USE_CUDA
    if (_dev) {
        bem3d::cuda::device_set_fp32(_dev, gpu_fp32);
        bem3d::cuda::device_set_warm(_dev, warm_start);
        bem3d::cuda::device_set_precond(_dev, use_precond);
    }
#endif

    double c = bem3d::gradient(_Ne, _HH, _elements, _Ps, _Es, _src_pt, Cs, Ns, _Ls, true, freq_band, _n_freqs, use_actual, xs, ys, lmbs, adj_bs, dcdh, _approx_ACA_tol, _Q_ACA_tol, _solver_tol, _recompute_matrices, silent, &_direct, &_approx, use_gpu, _dev);
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
#ifdef BEM_USE_CUDA
    if (_dev) {
        bem3d::cuda::device_set_fp32(_dev, gpu_fp32);
        bem3d::cuda::device_set_warm(_dev, warm_start);
        bem3d::cuda::device_set_precond(_dev, use_precond);
    }
#endif
    std::vector<double> coeffs;

    for (double freq_band : _freq_bands) {
        double c = bem3d::gradient(_Ne, _HH, _elements, _Ps, _Es, _src_pt, Cs, Ns, _Ls, true, freq_band, _n_freqs, use_actual, xs, ys, lmbs, adj_bs, dcdh, _approx_ACA_tol, _Q_ACA_tol, _solver_tol, _recompute_matrices, silent, &_direct, &_approx, use_gpu, _dev);
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
    {
        BEM_SCOPED_TIMER("outer.vertex_displace");
        for (const auto &hh : _Hs) {
            Ps(hh.first, 1) = x[hh.second] + _Ps(hh.first, 1);
        }
    }

    // compute intermediate values
    bem3d::mat3 Cs, Ns;
    {
        BEM_SCOPED_TIMER("outer.compute_intermediates");
        bem3d::compute_intermediates(Ps, _Es, Cs, Ns);
    }

    double c = 0;
    bem3d::vec d_grad, dcdh;
    std::vector<bem3d::cvec> xs, ys, lmbs, adj_bs;
#ifdef BEM_USE_CUDA
    if (_dev) {
        bem3d::cuda::device_set_fp32(_dev, gpu_fp32);
        bem3d::cuda::device_set_warm(_dev, warm_start);
        bem3d::cuda::device_set_precond(_dev, use_precond);
    }
#endif

    d_grad.setZero(_HH);
    {
        BEM_SCOPED_TIMER("outer.freq_bands_total");
        for (double freq_band : _freq_bands) {
            c += bem3d::gradient(_Ne, _HH, _elements, Ps, _Es, _src_pt, Cs, Ns, _Ls, false, freq_band, _n_freqs, use_actual, xs, ys, lmbs, adj_bs, dcdh, _approx_ACA_tol, _Q_ACA_tol, _solver_tol, _recompute_matrices, silent, &_direct, &_approx, use_gpu, _dev);
            d_grad += dcdh;
        }
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

void DiffBEM::reset_warm_start() {
#ifdef BEM_USE_CUDA
    if (_dev)
        bem3d::cuda::device_reset_warm(_dev);
#endif
}

#ifdef BEM_USE_CUDA
std::pair<double, double> DiffBEM::_debug_hmatvec(double frequency) {
    bem3d::mat3 Cs, Ns;
    bem3d::compute_intermediates(_Ps, _Es, Cs, Ns);
    double k = freq_to_wavenumber(frequency);

    bem3d::HMatrix hmat(_Ne, false);
    hmat.compute_direct_and_approx_blocks_cpu(_direct, _approx, k, _Ps, _Es, Cs, Ns, _approx_ACA_tol);

    bem3d::cvec rhs(_Ne);
    for (int i = 0; i < _Ne; i++)
        rhs[i] = std::complex<double>(std::sin(0.1 * i), std::cos(0.07 * i));

    bem3d::cvec y_cpu = hmat * rhs;
    hmat.set_transposed(true);
    bem3d::cvec y_cpuT = hmat * rhs;
    hmat.set_transposed(false);

    auto *H = bem3d::cuda_upload_hmatrix(hmat);
    bem3d::cvec y_gpu(_Ne), y_gpuT(_Ne);
    bem3d::cuda::hmat_matvec_gpu(H, rhs.data(), y_gpu.data(), false);
    bem3d::cuda::hmat_matvec_gpu(H, rhs.data(), y_gpuT.data(), true);
    bem3d::cuda::hmat_free(H);

    double e = (y_cpu - y_gpu).norm() / y_cpu.norm();
    double eT = (y_cpuT - y_gpuT).norm() / y_cpuT.norm();
    return {e, eT};
}

std::pair<double, int> DiffBEM::_debug_solve(double frequency) {
    bem3d::mat3 Cs, Ns;
    bem3d::compute_intermediates(_Ps, _Es, Cs, Ns);
    double k = freq_to_wavenumber(frequency);

    bem3d::HMatrix hmat(_Ne, false);
    hmat.compute_direct_and_approx_blocks_cpu(_direct, _approx, k, _Ps, _Es, Cs, Ns, _approx_ACA_tol);

    bem3d::cvec G_r;
    compute_G_r(Cs, _src_pt, k, G_r);
    bem3d::cvec b = -G_r;

    // reference: Eigen BiCGSTAB (same solver as the CPU path)
    Eigen::BiCGSTAB<bem3d::HMatrix, Eigen::IdentityPreconditioner> solver;
    solver.setTolerance(_solver_tol);
    solver.compute(hmat);
    bem3d::cvec x_eigen = solver.solve(b);

    // GPU solve
    auto *H = bem3d::cuda_upload_hmatrix(hmat);
    bem3d::cvec x_gpu(_Ne);
    int it_gpu = bem3d::cuda::solve_gpu(_dev, H, b.data(), x_gpu.data(), false, _solver_tol, _Ne);
    bem3d::cuda::hmat_free(H);

    double e = (x_eigen - x_gpu).norm() / x_eigen.norm();
    return {e, it_gpu};
}

std::pair<double, double> DiffBEM::_debug_gpu_assembly(double frequency) {
    bem3d::mat3 Cs, Ns;
    bem3d::compute_intermediates(_Ps, _Es, Cs, Ns);
    double k = freq_to_wavenumber(frequency);

    // CPU reference H-matrix
    bem3d::HMatrix hmat(_Ne, false);
    hmat.compute_direct_and_approx_blocks_cpu(_direct, _approx, k, _Ps, _Es, Cs, Ns, _approx_ACA_tol);
    bem3d::cvec rhs(_Ne);
    for (int i = 0; i < _Ne; i++)
        rhs[i] = std::complex<double>(std::sin(0.1 * i), std::cos(0.07 * i));
    bem3d::cvec y_cpu = hmat * rhs;
    hmat.set_transposed(true);
    bem3d::cvec y_cpuT = hmat * rhs;
    hmat.set_transposed(false);

    // GPU-assembled H-matrix
    bem3d::cuda::device_set_geometry(_dev, _Ps.data(), Cs.data(), Ns.data());
    auto *H = bem3d::cuda::device_assemble(_dev, k, _approx_ACA_tol);
    bem3d::cvec y_gpu(_Ne), y_gpuT(_Ne);
    bem3d::cuda::hmat_matvec_gpu(H, rhs.data(), y_gpu.data(), false);
    bem3d::cuda::hmat_matvec_gpu(H, rhs.data(), y_gpuT.data(), true);

    double e = (y_cpu - y_gpu).norm() / y_cpu.norm();
    double eT = (y_cpuT - y_gpuT).norm() / y_cpuT.norm();
    return {e, eT};
}
#endif

NB_MODULE(acoustics3d_impl, m) {
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
        .def_rw("use_actual", &DiffBEM::use_actual)
        .def_rw("use_gpu", &DiffBEM::use_gpu)
        .def_rw("gpu_fp32", &DiffBEM::gpu_fp32)
        .def_rw("warm_start", &DiffBEM::warm_start)
        .def_rw("use_precond", &DiffBEM::use_precond)
        .def("reset_warm_start", &DiffBEM::reset_warm_start)
#ifdef BEM_USE_CUDA
        .def("_debug_hmatvec", &DiffBEM::_debug_hmatvec, "frequency"_a)
        .def("_debug_solve", &DiffBEM::_debug_solve, "frequency"_a)
        .def("_debug_gpu_assembly", &DiffBEM::_debug_gpu_assembly, "frequency"_a)
#endif
        ;

    m.def("sphere", &bem3d::sphere_test::sphere, "sphere_mesh"_a, "freq"_a, "LL"_a = 100, "lrad"_a = 10, "actual"_a = false);

    // profiling controls (gated at runtime by the BEM_PROFILE env var)
    m.def("profile_summary", [] { return bem3d::Profiler::instance().summary(); });
    m.def("profile_reset", [] { bem3d::Profiler::instance().reset(); });
    m.def("profile_enabled", [] { return bem3d::Profiler::enabled(); });

    // CUDA backend availability + Phase 0 smoke test
#ifdef BEM_USE_CUDA
    m.attr("cuda_available") = true;
    m.def("cuda_selftest", [](int n) { return bem3d::cuda::device_selftest(n); }, "n"_a = 1);
    m.def("cuda_device_count", [] { return bem3d::cuda::device_count(); });

    // Phase 1: validate the ported device kernels against the CPU golden
    // (src/derivatives.cpp, src/integrate.cpp). Returns max relative error over
    // a handful of fixed geometric inputs; caller asserts it is within tol.
    m.def("_validate_device_math", [] {
        struct Case {
            double p[9], q[9], qn[3], src[3], k;
        };
        std::vector<Case> cases = {
            {{-0.30, 0.15, -0.30, -0.28, 0.15, -0.30, -0.30, 0.15, -0.28},
             {0.10, 0.12, 0.05, 0.13, 0.11, 0.06, 0.11, 0.14, 0.04},
             {0.0, 0.02, 0.0},
             {0.0, 100.0, 0.0},
             36.5},
            {{0.05, 0.10, 0.20, 0.07, 0.09, 0.19, 0.06, 0.12, 0.21},
             {-0.15, -0.05, -0.10, -0.13, -0.06, -0.12, -0.14, -0.07, -0.09},
             {0.01, -0.03, 0.02},
             {0.0, 100.0, 0.0},
             91.3},
            {{0.20, 0.00, -0.10, 0.22, 0.01, -0.11, 0.21, 0.02, -0.09},
             {-0.20, 0.05, 0.15, -0.19, 0.04, 0.16, -0.21, 0.06, 0.14},
             {-0.02, 0.03, 0.01},
             {0.0, 100.0, 0.0},
             18.25},
        };

        auto v3 = [](const double *a) { return bem3d::vec3(a[0], a[1], a[2]); };
        std::vector<double> per_kernel(6, 0.0); // [compute_M, dpy, dq1y, dq2y, dq3y, dbi]
        for (auto &c : cases) {
            std::complex<double> gpu[6];
            bem3d::cuda::eval_kernels_gpu(c.p, c.q, c.qn, c.src, c.k, gpu);

            bem3d::vec3 p1 = v3(c.p), p2 = v3(c.p + 3), p3 = v3(c.p + 6);
            bem3d::vec3 q1 = v3(c.q), q2 = v3(c.q + 3), q3 = v3(c.q + 6);
            bem3d::vec3 qn = v3(c.qn), src = v3(c.src);
            bem3d::vec3 cen = (p1 + p2 + p3) / 3.0;

            std::complex<double> cpu[6];
            cpu[0] = bem3d::compute_M(c.k, cen, q1, q2, q3, qn);
            cpu[1] = bem3d::dMij_dpy(p1, p2, p3, q1, q2, q3, c.k);
            cpu[2] = bem3d::dMij_dq1y(p1, p2, p3, q1, q2, q3, c.k);
            cpu[3] = bem3d::dMij_dq2y(p1, p2, p3, q1, q2, q3, c.k);
            cpu[4] = bem3d::dMij_dq3y(p1, p2, p3, q1, q2, q3, c.k);
            cpu[5] = bem3d::dbi_dpy(p1, p2, p3, src, c.k);

            for (int i = 0; i < 6; i++) {
                // allclose-style normalized error: |diff| / (atol + rtol*|cpu|).
                // <= 1 means within tolerance. atol handles outputs where the
                // derivative genuinely vanishes (~1e-19); rtol handles the rest.
                const double atol = 1e-12, rtol = 1e-11;
                double ratio = std::abs(gpu[i] - cpu[i]) / (atol + rtol * std::abs(cpu[i]));
                per_kernel[i] = std::max(per_kernel[i], ratio);
            }
        }
        return per_kernel;
    });
#else
    m.attr("cuda_available") = false;
#endif
}
