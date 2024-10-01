#include "derivatives.h"

namespace bem3d {

std::complex<double> dMij_dpy(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k, double mk, double nk) {
    auto p1x = p1[0], p1y = p1[1], p1z = p1[2];
    auto p2x = p2[0], p2y = p2[1], p2z = p2[2];
    auto p3x = p3[0], p3y = p3[1], p3z = p3[2];
    auto q1x = q1[0], q1y = q1[1], q1z = q1[2];
    auto q2x = q2[0], q2y = q2[1], q2z = q2[2];
    auto q3x = q3[0], q3y = q3[1], q3z = q3[2];

    auto x0 = q1x - q2x;
    auto x1 = q1x - q3x;
    auto x2 = -mk * x0 - nk * x1 - p1x / 3.0 - p2x / 3.0 - p3x / 3.0 + q1x;
    auto x3 = q1y - q2y;
    auto x4 = -mk * x3;
    auto x5 = q1y - q3y;
    auto x6 = -nk * x5;
    auto x7 = -p1y / 3.0 - p2y / 3.0 - p3y / 3.0 + q1y + x4 + x6;
    auto x8 = q1z - q2z;
    auto x9 = q1z - q3z;
    auto x10 = -mk * x8 - nk * x9 - p1z / 3.0 - p2z / 3.0 - p3z / 3.0 + q1z;
    auto x11 = x10 * x10 + x2 * x2 + x7 * x7;
    auto x12 = x0 * x9;
    auto x13 = k * 1.0i;
    auto x14 = std::sqrt(x11) * x13;
    auto x15 = 0.25 * std::exp(x14) / PI;
    auto x16 = x15 * (x14 - 1.0);
    auto x17 = x10 * (x0 * x5 - x1 * x3) + x2 * (x3 * x9 - x5 * x8) + x7 * (x1 * x8 - x12);
    auto x18 = x13 * (p1y / 9.0 + p2y / 9.0 + p3y / 9.0 - q1y / 3.0 - x4 / 3.0 - x6 / 3.0) / (x11 * x11);
    auto x19 = x16 * x17;
    return x15 * x17 * x18 + x18 * x19 + x16 * (-1.0 / 3.0 * x1 * x8 + (1.0 / 3.0) * x12) / std::pow(x11, 1.5) + x19 * x7 / std::pow(x11, 2.5);
}

std::complex<double> dMij_dq1y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k, double mk, double nk) {
    auto p1x = p1[0], p1y = p1[1], p1z = p1[2];
    auto p2x = p2[0], p2y = p2[1], p2z = p2[2];
    auto p3x = p3[0], p3y = p3[1], p3z = p3[2];
    auto q1x = q1[0], q1y = q1[1], q1z = q1[2];
    auto q2x = q2[0], q2y = q2[1], q2z = q2[2];
    auto q3x = q3[0], q3y = q3[1], q3z = q3[2];

    auto x0 = q1x - q2x;
    auto x1 = -q3x;
    auto x2 = q1x + x1;
    auto x3 = -mk * x0 - nk * x2 - p1x / 3.0 - p2x / 3.0 - p3x / 3.0 + q1x;
    auto x4 = q1y - q2y;
    auto x5 = q1y - q3y;
    auto x6 = -mk * x4 - nk * x5 - p1y / 3.0 - p2y / 3.0 - p3y / 3.0 + q1y;
    auto x7 = q1z - q2z;
    auto x8 = -q3z;
    auto x9 = q1z + x8;
    auto x10 = -mk * x7 - nk * x9 - p1z / 3.0 - p2z / 3.0 - p3z / 3.0 + q1z;
    auto x11 = x10 * x10 + x3 * x3 + x6 * x6;
    auto x12 = -x0 * x9 + x2 * x7;
    auto x13 = 1.0i * k;
    auto x14 = std::sqrt(x11) * x13;
    auto x15 = std::exp(x14) / PI;
    auto x16 = x15 * (x14 - 1.0);
    auto x17 = x6 * (-2 * mk - 2 * nk + 2) * (x10 * (x0 * x5 - x2 * x4) + x12 * x6 + x3 * (x4 * x9 - x5 * x7));
    auto x18 = (1.0 / 8.0) * x13 / (x11 * x11);
    auto x19 = x16 * x17;
    return x15 * x17 * x18 + x18 * x19 + 0.25 * x16 * (x10 * (-q2x - x1) + x12 * (-mk - nk + 1.0) + x3 * (q2z + x8)) / std::pow(x11, 1.5) - 3.0 / 8.0 * x19 / std::pow(x11, 2.5);
}

std::complex<double> dMij_dq2y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k, double mk, double nk) {
    auto p1x = p1[0], p1y = p1[1], p1z = p1[2];
    auto p2x = p2[0], p2y = p2[1], p2z = p2[2];
    auto p3x = p3[0], p3y = p3[1], p3z = p3[2];
    auto q1x = q1[0], q1y = q1[1], q1z = q1[2];
    auto q2x = q2[0], q2y = q2[1], q2z = q2[2];
    auto q3x = q3[0], q3y = q3[1], q3z = q3[2];

    auto x0 = q1x - q2x;
    auto x1 = q1x - q3x;
    auto x2 = -mk * x0 - nk * x1 - p1x / 3.0 - p2x / 3.0 - p3x / 3.0 + q1x;
    auto x3 = q1y - q2y;
    auto x4 = q1y - q3y;
    auto x5 = -mk * x3 - nk * x4 - p1y / 3.0 - p2y / 3.0 - p3y / 3.0 + q1y;
    auto x6 = q1z - q2z;
    auto x7 = q1z - q3z;
    auto x8 = -x7;
    auto x9 = -mk * x6 + nk * x8 - p1z / 3.0 - p2z / 3.0 - p3z / 3.0 + q1z;
    auto x10 = x2 * x2 + x5 * x5 + x9 * x9;
    auto x11 = -x0 * x7 + x1 * x6;
    auto x12 = 1.0 / PI;
    auto x13 = 1.0i * k;
    auto x14 = std::sqrt(x10) * x13;
    auto x15 = std::exp(x14);
    auto x16 = x12 * x15 * (x14 - 1.0);
    auto x17 = 0.25 * x16;
    auto x18 = mk * x5 * (x11 * x5 + x2 * (x3 * x7 - x4 * x6) + x9 * (x0 * x4 - x1 * x3));
    auto x19 = x13 * x18 / (x10 * x10);
    return 0.25 * x12 * x15 * x19 + x17 * x19 + x17 * (mk * x11 + x1 * x9 + x2 * x8) / std::pow(x10, 1.5) - 0.75 * x16 * x18 / std::pow(x10, 2.5);
}

std::complex<double> dMij_dq3y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k, double mk, double nk) {
    auto p1x = p1[0], p1y = p1[1], p1z = p1[2];
    auto p2x = p2[0], p2y = p2[1], p2z = p2[2];
    auto p3x = p3[0], p3y = p3[1], p3z = p3[2];
    auto q1x = q1[0], q1y = q1[1], q1z = q1[2];
    auto q2x = q2[0], q2y = q2[1], q2z = q2[2];
    auto q3x = q3[0], q3y = q3[1], q3z = q3[2];

    auto x0 = q1x - q2x;
    auto x1 = -x0;
    auto x2 = q1x - q3x;
    auto x3 = mk * x1 - nk * x2 - p1x / 3.0 - p2x / 3.0 - p3x / 3.0 + q1x;
    auto x4 = q1y - q2y;
    auto x5 = q1y - q3y;
    auto x6 = -mk * x4 - nk * x5 - p1y / 3.0 - p2y / 3.0 - p3y / 3.0 + q1y;
    auto x7 = q1z - q2z;
    auto x8 = q1z - q3z;
    auto x9 = -mk * x7 - nk * x8 - p1z / 3.0 - p2z / 3.0 - p3z / 3.0 + q1z;
    auto x10 = x3 * x3 + x6 * x6 + x9 * x9;
    auto x11 = -x0 * x8 + x2 * x7;
    auto x12 = 1 / PI;
    auto x13 = 1.0i * k;
    auto x14 = std::sqrt(x10) * x13;
    auto x15 = std::exp(x14);
    auto x16 = x12 * x15 * (x14 - 1.0);
    auto x17 = 0.25 * x16;
    auto x18 = nk * x6 * (x11 * x6 + x3 * (x4 * x8 - x5 * x7) + x9 * (x0 * x5 - x2 * x4));
    auto x19 = x13 * x18 / (x10 * x10);
    return 0.25 * x12 * x15 * x19 + x17 * x19 + x17 * (nk * x11 + x1 * x9 + x3 * x7) / std::pow(x10, 1.5) - 0.75 * x16 * x18 / std::pow(x10, 2.5);
}

std::complex<double> dbi_dpy(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &src, double k) {
    auto p1x = p1[0], p1y = p1[1], p1z = p1[2];
    auto p2x = p2[0], p2y = p2[1], p2z = p2[2];
    auto p3x = p3[0], p3y = p3[1], p3z = p3[2];
    auto sx = src[0], sy = src[1], sz = src[2];

    auto x0 = std::pow((p1x / 3.0 + p2x / 3.0 + p3x / 3.0 - sx), 2) + std::pow((p1y / 3.0 + p2y / 3.0 + p3y / 3.0 - sy), 2) + std::pow((p1z / 3.0 + p2z / 3.0 + p3z / 3.0 - sz), 2);
    auto x1 = p1y / 9.0 + p2y / 9.0 + p3y / 9.0 - sy / 3.0;
    auto x2 = 1.0i * k;
    auto x3 = 0.25 * std::exp(std::sqrt(x0) * x2) / PI;
    return x1 * x2 * x3 / x0 - x1 * x3 / std::pow(x0, 1.5);
}

std::complex<double> dMij_dpy(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k) {
    std::complex<double> result = 0;
    for (int i = 0; i < 6; i++) {
        result += QUAD_WEIGHTS_AND_POINTS[i][0] * dMij_dpy(p1, p2, p3, q1, q2, q3, k, QUAD_WEIGHTS_AND_POINTS[i][1], QUAD_WEIGHTS_AND_POINTS[i][2]);
    }
    return result * 0.5;
}

std::complex<double> dMij_dq1y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k) {
    std::complex<double> result = 0;
    for (int i = 0; i < 6; i++) {
        result += QUAD_WEIGHTS_AND_POINTS[i][0] * dMij_dq1y(p1, p2, p3, q1, q2, q3, k, QUAD_WEIGHTS_AND_POINTS[i][1], QUAD_WEIGHTS_AND_POINTS[i][2]);
    }
    return result * 0.5;
}

std::complex<double> dMij_dq2y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k) {
    std::complex<double> result = 0;
    for (int i = 0; i < 6; i++) {
        result += QUAD_WEIGHTS_AND_POINTS[i][0] * dMij_dq2y(p1, p2, p3, q1, q2, q3, k, QUAD_WEIGHTS_AND_POINTS[i][1], QUAD_WEIGHTS_AND_POINTS[i][2]);
    }
    return result * 0.5;
}

std::complex<double> dMij_dq3y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &q1, const vec3 &q2, const vec3 &q3, double k) {
    std::complex<double> result = 0;
    for (int i = 0; i < 6; i++) {
        result += QUAD_WEIGHTS_AND_POINTS[i][0] * dMij_dq3y(p1, p2, p3, q1, q2, q3, k, QUAD_WEIGHTS_AND_POINTS[i][1], QUAD_WEIGHTS_AND_POINTS[i][2]);
    }
    return result * 0.5;
}

double dC_dp1y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &l) {
    auto ax = p1[0], ay = p1[1], az = p1[2];
    auto bx = p2[0], by = p2[1], bz = p2[2];
    auto cx = p3[0], cy = p3[1], cz = p3[2];
    auto lx = l[0], ly = l[1], lz = l[2];

    double x0 = -cz;
    double x1 = -ax + bx;
    double x2 = -ay + cy;
    double x3 = -cx;
    double x4 = -ax - x3;
    double x5 = -ay + by;
    double x6 = x1 * x2 - x4 * x5;
    double x7 = -az - x0;
    double x8 = -az + bz;
    double x9 = -x1 * x7 + x4 * x8;
    double x10 = -x2 * x8 + x5 * x7;
    double x11 = x10 * x10 + x6 * x6 + x9 * x9;
    double x12 = 1 / std::sqrt(x11);
    double x13 = (-1.0 / 2.0 * x10 * (2 * bz - 2 * cz) - 1.0 / 2.0 * x6 * (-2 * bx + 2 * cx)) / std::pow(x11, 1.5);
    return lx * x10 * x13 + lx * x12 * (bz + x0) + ly * x13 * x9 + lz * x12 * (-bx - x3) + lz * x13 * x6;
}

double dC_dp2y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &l) {
    auto ax = p1[0], ay = p1[1], az = p1[2];
    auto bx = p2[0], by = p2[1], bz = p2[2];
    auto cx = p3[0], cy = p3[1], cz = p3[2];
    auto lx = l[0], ly = l[1], lz = l[2];

    double x0 = -az + cz;
    double x1 = -ax + bx;
    double x2 = -ay + cy;
    double x3 = ax - cx;
    double x4 = -x3;
    double x5 = -ay + by;
    double x6 = x1 * x2 - x4 * x5;
    double x7 = -az + bz;
    double x8 = -x0 * x1 + x4 * x7;
    double x9 = x0 * x5 - x2 * x7;
    double x10 = x6 * x6 + x8 * x8 + x9 * x9;
    double x11 = 1 / std::sqrt(x10);
    double x12 = (-1.0 / 2.0 * x6 * (2 * ax - 2 * cx) - 1.0 / 2.0 * x9 * (-2 * az + 2 * cz)) / std::pow(x10, 1.5);
    return lx * x0 * x11 + lx * x12 * x9 + ly * x12 * x8 + lz * x11 * x3 + lz * x12 * x6;
}

double dC_dp3y(const vec3 &p1, const vec3 &p2, const vec3 &p3, const vec3 &l) {
    auto ax = p1[0], ay = p1[1], az = p1[2];
    auto bx = p2[0], by = p2[1], bz = p2[2];
    auto cx = p3[0], cy = p3[1], cz = p3[2];
    auto lx = l[0], ly = l[1], lz = l[2];

    double x0 = az - bz;
    double x1 = -ax + bx;
    double x2 = -ay + cy;
    double x3 = -ax + cx;
    double x4 = -ay + by;
    double x5 = x1 * x2 - x3 * x4;
    double x6 = -az + cz;
    double x7 = -x0;
    double x8 = -x1 * x6 + x3 * x7;
    double x9 = -x2 * x7 + x4 * x6;
    double x10 = x5 * x5 + x8 * x8 + x9 * x9;
    double x11 = 1 / std::sqrt(x10);
    double x12 = (-1.0 / 2.0 * x5 * (-2 * ax + 2 * bx) - 1.0 / 2.0 * x9 * (2 * az - 2 * bz)) / std::pow(x10, 1.5);
    return lx * x0 * x11 + lx * x12 * x9 + ly * x12 * x8 + lz * x1 * x11 + lz * x12 * x5;
}

} // namespace bem3d