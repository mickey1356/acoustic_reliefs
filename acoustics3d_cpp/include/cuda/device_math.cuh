#pragma once

// Device (and host) port of the BEM boundary/derivative kernels.
// Single source of truth for the GPU path; validated against the untouched CPU
// reference in src/derivatives.cpp / src/integrate.cpp.
//
// Templated on Scalar S (double for the correctness baseline, float for the
// speed lever). All functions are __host__ __device__ and take raw S* points so
// that NO Eigen type ever enters device code.
//
// Idiom mapping from the CPU code:
//   std::complex<double>  -> thrust::complex<S>
//   1.0i literal          -> thrust::complex<S>(0,1)
//   std::pow(x,1.5|2.5)   -> x*sqrt(x) / x*x*sqrt(x)  (faster and closer to CPU)
//   return fin.real() - fin.imag()*1.0i -> complex(fin.real(), -fin.imag())

#include <thrust/complex.h>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

namespace bem3d {
namespace cuda {

// PI matches constants.h:12 exactly.
template <class S>
__host__ __device__ inline S dm_pi() {
    return S(3.14159265358979323846);
}

template <class S>
__host__ __device__ inline S dm_sqrt(S x) {
    return sqrt(x);
}

// 6-point triangle quadrature: {weight, mk, nk}, matches constants.h:39-46.
template <class S>
__host__ __device__ inline void dm_quad(int i, S &w, S &mk, S &nk) {
    // stored as double literals; cast to S on read
    const double Q[6][3] = {
        {0.223381589678011, 0.445948490915965, 0.108103018168070},
        {0.223381589678011, 0.445948490915965, 0.445948490915965},
        {0.223381589678011, 0.108103018168070, 0.445948490915965},
        {0.109951743655322, 0.091576213509771, 0.816847572980459},
        {0.109951743655322, 0.091576213509771, 0.091576213509771},
        {0.109951743655322, 0.816847572980459, 0.091576213509771},
    };
    w = S(Q[i][0]);
    mk = S(Q[i][1]);
    nk = S(Q[i][2]);
}

// ---- boundary kernel (integrate.h) ----

template <class S>
__host__ __device__ inline thrust::complex<S> dm_dGdr(S k, S r) {
    thrust::complex<S> I(S(0), S(1));
    return thrust::exp(-I * k * r) * (-I * k * r - S(1)) / (S(4) * dm_pi<S>() * r * r);
}

template <class S>
__host__ __device__ inline thrust::complex<S>
dm_kernel(S k, const S p[3], const S pt[3], const S normal[3]) {
    S rv[3] = {p[0] - pt[0], p[1] - pt[1], p[2] - pt[2]};
    S rn = dm_sqrt<S>(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
    S dot = rv[0] * normal[0] + rv[1] * normal[1] + rv[2] * normal[2];
    return dm_dGdr<S>(k, rn) * (-dot / rn);
}

// compute_M: double-layer entry, 6-pt quadrature over the source triangle.
template <class S>
__host__ __device__ inline thrust::complex<S>
dm_compute_M(S k, const S p[3], const S qa[3], const S qb[3], const S qc[3], const S qn[3]) {
    S ab[3] = {qb[0] - qa[0], qb[1] - qa[1], qb[2] - qa[2]};
    S ac[3] = {qc[0] - qa[0], qc[1] - qa[1], qc[2] - qa[2]};
    thrust::complex<S> res(S(0), S(0));
    for (int i = 0; i < 6; i++) {
        S w, mk, nk;
        dm_quad<S>(i, w, mk, nk);
        S pt[3] = {qa[0] + mk * ab[0] + nk * ac[0],
                   qa[1] + mk * ab[1] + nk * ac[1],
                   qa[2] + mk * ab[2] + nk * ac[2]};
        res += w * dm_kernel<S>(k, p, pt, qn);
    }
    return res * S(0.5);
}

// ---- derivative kernels (per quadrature point; derivatives.cpp) ----
// Each mirrors the CPU function line-by-line.

template <class S>
__host__ __device__ inline thrust::complex<S>
dm_dMij_dpy_q(const S p1[3], const S p2[3], const S p3[3],
              const S q1[3], const S q2[3], const S q3[3], S k, S mk, S nk) {
    S p1x = p1[0], p1y = p1[1], p1z = p1[2];
    S p2x = p2[0], p2y = p2[1], p2z = p2[2];
    S p3x = p3[0], p3y = p3[1], p3z = p3[2];
    S q1x = q1[0], q1y = q1[1], q1z = q1[2];
    S q2x = q2[0], q2y = q2[1], q2z = q2[2];
    S q3x = q3[0], q3y = q3[1], q3z = q3[2];
    thrust::complex<S> I(S(0), S(1));

    S x0 = q1x - q2x;
    S x1 = q1x - q3x;
    S x2 = -mk * x0 - nk * x1 - p1x / S(3) - p2x / S(3) - p3x / S(3) + q1x;
    S x3 = q1y - q2y;
    S x4 = -mk * x3;
    S x5 = q1y - q3y;
    S x6 = -nk * x5;
    S x7 = -p1y / S(3) - p2y / S(3) - p3y / S(3) + q1y + x4 + x6;
    S x8 = q1z - q2z;
    S x9 = q1z - q3z;
    S x10 = -mk * x8 - nk * x9 - p1z / S(3) - p2z / S(3) - p3z / S(3) + q1z;
    S x11 = x10 * x10 + x2 * x2 + x7 * x7;
    S x12 = x0 * x9;
    thrust::complex<S> x13 = k * I;
    thrust::complex<S> x14 = dm_sqrt<S>(x11) * x13;
    thrust::complex<S> x15 = S(0.25) * thrust::exp(x14) / dm_pi<S>();
    thrust::complex<S> x16 = x15 * (x14 - S(1));
    S x17 = x10 * (x0 * x5 - x1 * x3) + x2 * (x3 * x9 - x5 * x8) + x7 * (x1 * x8 - x12);
    thrust::complex<S> x18 = x13 * (p1y / S(9) + p2y / S(9) + p3y / S(9) - q1y / S(3) - x4 / S(3) - x6 / S(3)) / (x11 * x11);
    thrust::complex<S> x19 = x16 * x17;
    S sx11 = dm_sqrt<S>(x11);
    S x11_1p5 = x11 * sx11;       // pow(x11, 1.5)
    S x11_2p5 = x11 * x11 * sx11; // pow(x11, 2.5)
    thrust::complex<S> fin = x15 * x17 * x18 + x18 * x19 +
                             x16 * (-S(1) / S(3) * x1 * x8 + (S(1) / S(3)) * x12) / x11_1p5 +
                             x19 * x7 / x11_2p5;
    return thrust::complex<S>(fin.real(), -fin.imag());
}

template <class S>
__host__ __device__ inline thrust::complex<S>
dm_dMij_dq1y_q(const S p1[3], const S p2[3], const S p3[3],
               const S q1[3], const S q2[3], const S q3[3], S k, S mk, S nk) {
    S p1x = p1[0], p1y = p1[1], p1z = p1[2];
    S p2x = p2[0], p2y = p2[1], p2z = p2[2];
    S p3x = p3[0], p3y = p3[1], p3z = p3[2];
    S q1x = q1[0], q1y = q1[1], q1z = q1[2];
    S q2x = q2[0], q2y = q2[1], q2z = q2[2];
    S q3x = q3[0], q3y = q3[1], q3z = q3[2];
    thrust::complex<S> I(S(0), S(1));

    S x0 = q1x - q2x;
    S x1 = -q3x;
    S x2 = q1x + x1;
    S x3 = -mk * x0 - nk * x2 - p1x / S(3) - p2x / S(3) - p3x / S(3) + q1x;
    S x4 = q1y - q2y;
    S x5 = q1y - q3y;
    S x6 = -mk * x4 - nk * x5 - p1y / S(3) - p2y / S(3) - p3y / S(3) + q1y;
    S x7 = q1z - q2z;
    S x8 = -q3z;
    S x9 = q1z + x8;
    S x10 = -mk * x7 - nk * x9 - p1z / S(3) - p2z / S(3) - p3z / S(3) + q1z;
    S x11 = x10 * x10 + x3 * x3 + x6 * x6;
    S x12 = -x0 * x9 + x2 * x7;
    thrust::complex<S> x13 = I * k;
    thrust::complex<S> x14 = dm_sqrt<S>(x11) * x13;
    thrust::complex<S> x15 = thrust::exp(x14) / dm_pi<S>();
    thrust::complex<S> x16 = x15 * (x14 - S(1));
    S x17 = x6 * (-S(2) * mk - S(2) * nk + S(2)) * (x10 * (x0 * x5 - x2 * x4) + x12 * x6 + x3 * (x4 * x9 - x5 * x7));
    thrust::complex<S> x18 = (S(1) / S(8)) * x13 / (x11 * x11);
    thrust::complex<S> x19 = x16 * x17;
    S sx11 = dm_sqrt<S>(x11);
    S x11_1p5 = x11 * sx11;
    S x11_2p5 = x11 * x11 * sx11;
    thrust::complex<S> fin = x15 * x17 * x18 + x18 * x19 +
                             S(0.25) * x16 * (x10 * (-q2x - x1) + x12 * (-mk - nk + S(1)) + x3 * (q2z + x8)) / x11_1p5 -
                             S(3) / S(8) * x19 / x11_2p5;
    return thrust::complex<S>(fin.real(), -fin.imag());
}

template <class S>
__host__ __device__ inline thrust::complex<S>
dm_dMij_dq2y_q(const S p1[3], const S p2[3], const S p3[3],
               const S q1[3], const S q2[3], const S q3[3], S k, S mk, S nk) {
    S p1x = p1[0], p1y = p1[1], p1z = p1[2];
    S p2x = p2[0], p2y = p2[1], p2z = p2[2];
    S p3x = p3[0], p3y = p3[1], p3z = p3[2];
    S q1x = q1[0], q1y = q1[1], q1z = q1[2];
    S q2x = q2[0], q2y = q2[1], q2z = q2[2];
    S q3x = q3[0], q3y = q3[1], q3z = q3[2];
    thrust::complex<S> I(S(0), S(1));

    S x0 = q1x - q2x;
    S x1 = q1x - q3x;
    S x2 = -mk * x0 - nk * x1 - p1x / S(3) - p2x / S(3) - p3x / S(3) + q1x;
    S x3 = q1y - q2y;
    S x4 = q1y - q3y;
    S x5 = -mk * x3 - nk * x4 - p1y / S(3) - p2y / S(3) - p3y / S(3) + q1y;
    S x6 = q1z - q2z;
    S x7 = q1z - q3z;
    S x8 = -x7;
    S x9 = -mk * x6 + nk * x8 - p1z / S(3) - p2z / S(3) - p3z / S(3) + q1z;
    S x10 = x2 * x2 + x5 * x5 + x9 * x9;
    S x11 = -x0 * x7 + x1 * x6;
    S x12 = S(1) / dm_pi<S>();
    thrust::complex<S> x13 = I * k;
    thrust::complex<S> x14 = dm_sqrt<S>(x10) * x13;
    thrust::complex<S> x15 = thrust::exp(x14);
    thrust::complex<S> x16 = x12 * x15 * (x14 - S(1));
    thrust::complex<S> x17 = S(0.25) * x16;
    S x18 = mk * x5 * (x11 * x5 + x2 * (x3 * x7 - x4 * x6) + x9 * (x0 * x4 - x1 * x3));
    thrust::complex<S> x19 = x13 * x18 / (x10 * x10);
    S sx10 = dm_sqrt<S>(x10);
    S x10_1p5 = x10 * sx10;
    S x10_2p5 = x10 * x10 * sx10;
    thrust::complex<S> fin = S(0.25) * x12 * x15 * x19 + x17 * x19 +
                             x17 * (mk * x11 + x1 * x9 + x2 * x8) / x10_1p5 -
                             S(0.75) * x16 * x18 / x10_2p5;
    return thrust::complex<S>(fin.real(), -fin.imag());
}

template <class S>
__host__ __device__ inline thrust::complex<S>
dm_dMij_dq3y_q(const S p1[3], const S p2[3], const S p3[3],
               const S q1[3], const S q2[3], const S q3[3], S k, S mk, S nk) {
    S p1x = p1[0], p1y = p1[1], p1z = p1[2];
    S p2x = p2[0], p2y = p2[1], p2z = p2[2];
    S p3x = p3[0], p3y = p3[1], p3z = p3[2];
    S q1x = q1[0], q1y = q1[1], q1z = q1[2];
    S q2x = q2[0], q2y = q2[1], q2z = q2[2];
    S q3x = q3[0], q3y = q3[1], q3z = q3[2];
    thrust::complex<S> I(S(0), S(1));

    S x0 = q1x - q2x;
    S x1 = -x0;
    S x2 = q1x - q3x;
    S x3 = mk * x1 - nk * x2 - p1x / S(3) - p2x / S(3) - p3x / S(3) + q1x;
    S x4 = q1y - q2y;
    S x5 = q1y - q3y;
    S x6 = -mk * x4 - nk * x5 - p1y / S(3) - p2y / S(3) - p3y / S(3) + q1y;
    S x7 = q1z - q2z;
    S x8 = q1z - q3z;
    S x9 = -mk * x7 - nk * x8 - p1z / S(3) - p2z / S(3) - p3z / S(3) + q1z;
    S x10 = x3 * x3 + x6 * x6 + x9 * x9;
    S x11 = -x0 * x8 + x2 * x7;
    S x12 = S(1) / dm_pi<S>();
    thrust::complex<S> x13 = I * k;
    thrust::complex<S> x14 = dm_sqrt<S>(x10) * x13;
    thrust::complex<S> x15 = thrust::exp(x14);
    thrust::complex<S> x16 = x12 * x15 * (x14 - S(1));
    thrust::complex<S> x17 = S(0.25) * x16;
    S x18 = nk * x6 * (x11 * x6 + x3 * (x4 * x8 - x5 * x7) + x9 * (x0 * x5 - x2 * x4));
    thrust::complex<S> x19 = x13 * x18 / (x10 * x10);
    S sx10 = dm_sqrt<S>(x10);
    S x10_1p5 = x10 * sx10;
    S x10_2p5 = x10 * x10 * sx10;
    thrust::complex<S> fin = S(0.25) * x12 * x15 * x19 + x17 * x19 +
                             x17 * (nk * x11 + x1 * x9 + x3 * x7) / x10_1p5 -
                             S(0.75) * x16 * x18 / x10_2p5;
    return thrust::complex<S>(fin.real(), -fin.imag());
}

// b-vector source derivative (no quadrature; derivatives.cpp:133).
template <class S>
__host__ __device__ inline thrust::complex<S>
dm_dbi_dpy(const S p1[3], const S p2[3], const S p3[3], const S src[3], S k) {
    S p1x = p1[0], p1y = p1[1], p1z = p1[2];
    S p2x = p2[0], p2y = p2[1], p2z = p2[2];
    S p3x = p3[0], p3y = p3[1], p3z = p3[2];
    S sx = src[0], sy = src[1], sz = src[2];
    thrust::complex<S> I(S(0), S(1));

    S cx = p1x / S(3) + p2x / S(3) + p3x / S(3) - sx;
    S cy = p1y / S(3) + p2y / S(3) + p3y / S(3) - sy;
    S cz = p1z / S(3) + p2z / S(3) + p3z / S(3) - sz;
    S x0 = cx * cx + cy * cy + cz * cz;
    S x1 = p1y / S(9) + p2y / S(9) + p3y / S(9) - sy / S(3);
    thrust::complex<S> x2 = I * k;
    thrust::complex<S> x3 = S(0.25) * thrust::exp(dm_sqrt<S>(x0) * x2) / dm_pi<S>();
    S x0_1p5 = x0 * dm_sqrt<S>(x0);
    thrust::complex<S> fin = x1 * x2 * x3 / x0 - x1 * x3 / x0_1p5;
    return thrust::complex<S>(fin.real(), -fin.imag());
}

// ---- 6-point-quadrature wrappers (derivatives.cpp:147-177) ----

#define BEM_DM_QUAD_WRAP(NAME, PERPT)                                                 \
    template <class S>                                                                \
    __host__ __device__ inline thrust::complex<S> NAME(                               \
        const S p1[3], const S p2[3], const S p3[3],                                  \
        const S q1[3], const S q2[3], const S q3[3], S k) {                           \
        thrust::complex<S> res(S(0), S(0));                                           \
        for (int i = 0; i < 6; i++) {                                                 \
            S w, mk, nk;                                                              \
            dm_quad<S>(i, w, mk, nk);                                                 \
            res += w * PERPT<S>(p1, p2, p3, q1, q2, q3, k, mk, nk);                   \
        }                                                                            \
        return res * S(0.5);                                                          \
    }

BEM_DM_QUAD_WRAP(dm_dMij_dpy, dm_dMij_dpy_q)
BEM_DM_QUAD_WRAP(dm_dMij_dq1y, dm_dMij_dq1y_q)
BEM_DM_QUAD_WRAP(dm_dMij_dq2y, dm_dMij_dq2y_q)
BEM_DM_QUAD_WRAP(dm_dMij_dq3y, dm_dMij_dq3y_q)

#undef BEM_DM_QUAD_WRAP

// dispatch dq{1,2,3}y by local vertex order (0/1/2)
template <class S>
__host__ __device__ inline thrust::complex<S>
dm_dMij_dqy(int order, const S p1[3], const S p2[3], const S p3[3],
            const S q1[3], const S q2[3], const S q3[3], S k) {
    if (order == 0)
        return dm_dMij_dq1y<S>(p1, p2, p3, q1, q2, q3, k);
    else if (order == 1)
        return dm_dMij_dq2y<S>(p1, p2, p3, q1, q2, q3, k);
    else
        return dm_dMij_dq3y<S>(p1, p2, p3, q1, q2, q3, k);
}

} // namespace cuda
} // namespace bem3d
