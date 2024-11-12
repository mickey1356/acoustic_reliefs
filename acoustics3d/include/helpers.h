#pragma once

#include "constants.h"

namespace bem3d {

inline double freq_to_wavenumber(double freq) {
    return 2 * PI * freq / SPEED_OF_SOUND;
}

void compute_centers(const mat3 &Ps, const imat3 &Es, mat3 &Cs);

void compute_intermediates(const mat3 &Ps, const imat3 &Es, mat3 &Cs, mat3 &Ns);

void spherical_to_cartesian(double radius, const Eigen::ArrayXd &elevations, const Eigen::ArrayXd &azimuths, mat3 &pts);

void compute_G_r(const mat3 &Cs, const Eigen::RowVector3d &src_pt, double k, cvec &G_r);

int compute_listener_pts(mat3 &Ls, double lr, double ds);

std::vector<double> get_frequencies(double freq_band, int n_freqs);

} // namespace bem3d