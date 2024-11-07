#pragma once

#include "constants.h"

namespace bem3d {

std::complex<double> dMij_dpy(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k, double mk, double nk);

std::complex<double> dMij_dq1y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k, double mk, double nk);

std::complex<double> dMij_dq2y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k, double mk, double nk);

std::complex<double> dMij_dq3y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k, double mk, double nk);

std::complex<double> dbi_dpy(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &src, double k);

std::complex<double> dMij_dpy(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k);

std::complex<double> dMij_dq1y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k);

std::complex<double> dMij_dq2y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k);

std::complex<double> dMij_dq3y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k);

} // namespace bem3d