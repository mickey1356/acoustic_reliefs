#pragma once

#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "constants.h"

// wrapper to compute the value/gradient of BEM
class DiffBEM {
public:
    // set initial parameters for the problem instance
    //   - clustertree params (leaf size and radius factor)
    //   - freq bands and no. freqs per band
    //   - estimation tolerances and whether or not to save the matrices
    DiffBEM(int cluster_size, double radius_factor, const std::vector<double> &freq_bands, int n_freqs, double approx_ACA_tol, double Q_ACA_tol, double solver_tol, double listener_ds, bool recompute_matrices);

    // precompute the direct and approx blocks, listener positions, and Hs/elements (diff points)
    std::pair<bem3d::mat3, bem3d::imat3> set_mesh(const bem3d::mat3 &Ps, const bem3d::imat3 &Es);
    void set_diff_points(const std::vector<int> &diff_pts);
    std::pair<bem3d::mat3, bem3d::imat3> precompute(const bem3d::mat3 &Ps, const bem3d::imat3 &Es, const std::vector<int> &diff_pts);

    // returns the averaged diffusion coefficient of the current mesh over all frequency bands
    double value();
    double band_value(double freq_band);
    // returns the diffusion coefficients of the current mesh for all frequency bands
    std::vector<double> values();
    // given an initial x, compute the value and the gradient
    std::pair<double, bem3d::vec> gradient(const bem3d::vec &x);

    std::pair<bem3d::mat3, bem3d::imat3> get_mesh();
    std::pair<bem3d::mat3, bem3d::imat3> get_mesh(const bem3d::vec &x);

    std::unordered_map<int, int> get_Hs();

    bool silent = false;

private:
    bem3d::mat3 _Ps, _Ls;
    bem3d::imat3 _Es;
    std::vector<std::pair<pii, pii>> _direct, _approx;

    std::unordered_map<int, int> _Hs;
    std::vector<std::vector<pii>> _elements;

    int _Ne, _HH;

    // problem parameters
    int _cluster_size;
    double _radius_factor;
    std::vector<double> _freq_bands;
    int _n_freqs;
    double _approx_ACA_tol;
    double _Q_ACA_tol;
    double _solver_tol;
    double _listener_ds;
    bool _recompute_matrices;

    // make sure pre-steps have been completed
    bool _mesh_set = false, _diff_set = false, _raster_mesh_set = false;
};