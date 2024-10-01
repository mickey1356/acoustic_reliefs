#include "helpers.h"

#include <iostream>

namespace bem3d {

void compute_centers(const mat3 &Ps, const imat3 &Es, mat3 &Cs) {
    Cs.setZero(Es.rows(), 3);
    auto pta = Ps(Es.col(0), all);
    auto ptb = Ps(Es.col(1), all);
    auto ptc = Ps(Es.col(2), all);

    Cs = (pta + ptb + ptc) / 3;
}

void compute_intermediates(const mat3 &Ps, const imat3 &Es, mat3 &Cs, mat3 &Ns) {
    Cs.setZero(Es.rows(), 3);
    Ns.setZero(Es.rows(), 3);

    auto pta = Ps(Es.col(0), all);
    auto ptb = Ps(Es.col(1), all);
    auto ptc = Ps(Es.col(2), all);

    Cs = (pta + ptb + ptc) / 3;

    auto ab = ptb - pta;
    auto ac = ptc - pta;

    Ns.col(0) = ac.col(2).array() * ab.col(1).array() - ac.col(1).array() * ab.col(2).array();
    Ns.col(1) = ac.col(0).array() * ab.col(2).array() - ac.col(2).array() * ab.col(0).array();
    Ns.col(2) = ac.col(1).array() * ab.col(0).array() - ac.col(0).array() * ab.col(1).array();
}

void spherical_to_cartesian(double radius, const Eigen::ArrayXd &elevations, const Eigen::ArrayXd &azimuths, mat3 &pts) {
    auto theta_rad = elevations * PI / 180.0;
    auto phi_rad = azimuths * PI / 180.0;
    pts.setZero(elevations.size(), 3);
    // pts.col(0) = radius * theta_rad.sin() * phi_rad.sin();
    // pts.col(1) = radius * theta_rad.sin() * phi_rad.cos();
    // pts.col(2) = radius * theta_rad.cos();
    pts.col(0) = radius * theta_rad.sin() * phi_rad.cos();
    pts.col(1) = radius * theta_rad.cos();
    pts.col(2) = -radius * theta_rad.sin() * phi_rad.sin();
}

void compute_G_r(const mat3 &Cs, double k, cvec &G_r) {
    G_r.setZero(Cs.rows());
    auto r = (Cs.rowwise() - SRC_PT).rowwise().norm().array();
    G_r = (1.0i * k * r).exp() / (4 * PI * r);
}

int compute_listener_pts(mat3 &Ls, double ds) {
    std::vector<double> elevations{0}, azimuths{0};
    for (double elev = ds; elev < 91; elev += ds) {
        for (double azim = 0; azim < 360; azim += ds) {
            elevations.push_back(elev);
            azimuths.push_back(azim);
        }
    }
    Eigen::ArrayXd elev_vec = Eigen::ArrayXd::Map(elevations.data(), elevations.size());
    Eigen::ArrayXd azim_vec = Eigen::ArrayXd::Map(azimuths.data(), azimuths.size());
    spherical_to_cartesian(LISTENER_RADIUS, elev_vec, azim_vec, Ls);
    return static_cast<int>(Ls.rows());
}

std::vector<double> get_frequencies(double freq_band, int n_freqs) {
    if (n_freqs == 1) {
        return std::vector<double>{freq_band};
    }

    double band_width = std::pow(2, (1.0 / 6.0));
    double lower = std::log(freq_band / band_width);
    double higher = std::log(freq_band * band_width);

    double delta = (higher - lower) / (n_freqs - 1.0);

    std::vector<double> frequencies;
    for (int i = 0; i < n_freqs; i++) {
        frequencies.push_back(std::exp(lower + i * delta));
    }
    return frequencies;
}

} // namespace bem3d