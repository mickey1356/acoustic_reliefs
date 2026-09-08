#pragma once

#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "constants.h"

namespace bem3d {
namespace cuda {
struct DeviceState; // opaque; only used behind a pointer (defined in the CUDA TU)
}
} // namespace bem3d

// wrapper to compute the value/gradient of BEM
class DiffBEM {
public:
    // set initial parameters for the problem instance
    //   - clustertree params (leaf size and radius factor)
    //   - freq bands and no. freqs per band
    //   - estimation tolerances and whether or not to save the matrices
    DiffBEM(int cluster_size, double radius_factor, const std::vector<double> &freq_bands, int n_freqs, double approx_ACA_tol, double Q_ACA_tol, double solver_tol, const Eigen::RowVector3d &src_pt, double listener_radius, double listener_ds, bool recompute_matrices);
    ~DiffBEM();

    // precompute the direct and approx blocks, listener positions, and Hs/elements (diff points)
    std::pair<bem3d::mat3, bem3d::imat3> set_mesh(const bem3d::mat3 &Ps, const bem3d::imat3 &Es);
    void set_diff_points(const std::vector<int> &diff_pts);
    std::pair<bem3d::mat3, bem3d::imat3> precompute(const bem3d::mat3 &Ps, const bem3d::imat3 &Es, const std::vector<int> &diff_pts);

    // returns the averaged diffusion coefficient of the current mesh over all frequency bands
    double value();
    double value(const bem3d::vec &x);
    double band_value(double freq_band);
    // returns the diffusion coefficients of the current mesh for all frequency bands
    std::vector<double> values();
    // given an initial x, compute the value and the gradient
    std::pair<double, bem3d::vec> gradient(const bem3d::vec &x);
    // assuming the mesh is set, given a frequency (not band), and a list of listener points, compute the complex pressure at said listener points
    bem3d::cvec pvals(double frequency);
    bem3d::cvec pvals(double frequency, const bem3d::mat3 &listeners);

    bem3d::cvec surface_vals(double frequency);

    bem3d::mat3 get_listeners();

    std::pair<bem3d::mat3, bem3d::imat3> get_mesh();
    std::pair<bem3d::mat3, bem3d::imat3> get_mesh(const bem3d::vec &x);

    std::unordered_map<int, int> get_Hs();

    void set_band(double freq_band);

#ifdef BEM_USE_CUDA
    // Validate the device H-matvec against the CPU HMatrix (returns {normal, transposed} rel err).
    std::pair<double, double> _debug_hmatvec(double frequency);
    // Validate the GPU BiCGSTAB against Eigen's (returns {rel err vs Eigen x, gpu iters}).
    std::pair<double, int> _debug_solve(double frequency);
    // Validate the GPU-assembled H-matrix (device assembly -> matvec) vs CPU.
    std::pair<double, double> _debug_gpu_assembly(double frequency);
#endif

    bool silent = false;
    bool use_actual = false;
    bool use_gpu = false;   // route the GPU-able phases (der_A/B/C, ACA) through CUDA
    bool gpu_fp32 = false;  // FP32 compute on the GPU (speed lever; slightly less accurate)
    bool warm_start = false; // warm-start the GPU BiCGSTAB from the previous solution
    bool use_precond = false; // block-Jacobi preconditioner for the GPU BiCGSTAB

    void reset_warm_start(); // zero the warm-start caches (call before a new optimization)

private:
    // CUDA persistent device state (null unless built with BEM_USE_CUDA and precomputed)
    bem3d::cuda::DeviceState *_dev = nullptr;
    void upload_device_static();

    bem3d::mat3 _Ps, _Ls;
    bem3d::imat3 _Es;
    std::vector<std::pair<pii, pii>> _direct, _approx;

    std::unordered_map<int, int> _Hs;
    std::vector<std::vector<pii>> _elements;

    int _Ne = 0, _HH = 0;

    // problem parameters
    int _cluster_size;
    double _radius_factor;
    std::vector<double> _freq_bands;
    int _n_freqs;
    double _approx_ACA_tol;
    double _Q_ACA_tol;
    double _solver_tol;
    Eigen::RowVector3d _src_pt = Eigen::RowVector3d(0, 100, 0);
    double _listener_radius, _listener_ds;
    bool _recompute_matrices;

    // make sure pre-steps have been completed
    bool _mesh_set = false, _diff_set = false;
};