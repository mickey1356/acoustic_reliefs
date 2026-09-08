#pragma once

// Host-callable declarations for the CUDA backend launchers.
// Included only under #ifdef BEM_USE_CUDA. No CUDA headers pulled in here, so it
// is safe to include from plain C++ TUs (adjoint.cpp / acoustics3d.cpp).

#include <complex>
#include <vector>

namespace bem3d {
namespace cuda {

// Phase 1 validation: evaluate the 6 boundary/derivative kernels for a single
// (p-triangle, q-triangle, normal, source) input on the device, in FP64.
// out order: [compute_M, dMij_dpy, dMij_dq1y, dMij_dq2y, dMij_dq3y, dbi_dpy].
// compute_M's evaluation point is the centroid of the p-triangle.
void eval_kernels_gpu(const double p[9], const double q[9], const double qn[3],
                      const double src[3], double k, std::complex<double> out[6]);

// ---- Phase 2/3: persistent device state + derivative-assembly launchers ----
//
// DeviceState is an opaque per-DiffBEM handle holding device buffers that
// persist across gradient() calls. STATIC data (connectivity, listeners, the
// flattened incidence "task list") is uploaded once via device_set_static;
// per-iteration geometry (Ps) and solution vectors (x/lmb/vf) are uploaded by
// the launchers. Task t = (task_h[t], task_elem[t], task_order[t]) is one
// (diff-point, incident-element, local-vertex-order) triple, flattened from the
// CPU `elements` structure.
struct DeviceState;

DeviceState *device_state_create();
void device_state_destroy(DeviceState *);
void device_set_fp32(DeviceState *dev, bool fp32);
void device_set_warm(DeviceState *dev, bool warm);
void device_reset_warm(DeviceState *dev);
void device_set_precond(DeviceState *dev, bool precond);

void device_set_static(DeviceState *dev, const int *Es, int Ne, const double *Ls, int LL,
                       const int *task_h, const int *task_elem, const int *task_order,
                       int T, int HH, int Nv);

// Upload per-iteration geometry (Ps: Nv*3, Cs/Ns: Ne*3). Call once per gradient()
// call, before the ACA/der/assembly launchers, which reuse the cached geometry.
void device_set_geometry(DeviceState *dev, const double *Ps, const double *Cs, const double *Ns);

// ACA entry evaluations (reuse cached Ps/Es/Ns/Ls). aca_row fills out[Ne] with
// compute_M(k, Ls[listener_i], element_j) over all j; aca_col fills out[LL] with
// compute_M(k, Ls[l], element elem_j) over all listeners l.
void aca_row_gpu(DeviceState *dev, double k, int listener_i, std::complex<double> *out);
void aca_col_gpu(DeviceState *dev, double k, int elem_j, std::complex<double> *out);

// device-resident ACA: runs the whole pivot loop on-device; returns rank and
// fills hU (LL*rank, colmajor L x rank) and hV (rank*Ne, rowmajor rank x Ne).
int aca_resident_gpu(DeviceState *dev, int LL, int Ne, double k, double eps, int max_iter,
                     std::vector<std::complex<double>> &hU, std::vector<std::complex<double>> &hV);

// der_B[HH] += sum over tasks; uploads Ps(Nv*3), x(Ne), lmb(Ne).
void compute_der_B_gpu(DeviceState *dev, double k, const double *Ps,
                       const std::complex<double> *x, const std::complex<double> *lmb,
                       double *der_B_out);

// der_A[HH] += sum over tasks; uploads Ps(Nv*3), vf(LL), x(Ne).
void compute_der_A_gpu(DeviceState *dev, double k, const double *Ps,
                       const std::complex<double> *vf, const std::complex<double> *x,
                       double *der_A_out);

// der_C[HH] += sum over tasks; uploads Ps(Nv*3), lmb(Ne), src(3).
void compute_der_C_gpu(DeviceState *dev, double k, const double *Ps, const double src[3],
                       const std::complex<double> *lmb, double *der_C_out);

// ---- Solve port: device H-matrix + BiCGSTAB ----
// DeviceHMat holds one flattened H-matrix on the device (direct dense blocks +
// low-rank approx blocks). Built once per solve via hmat_upload; reused (with the
// transposed matvec) for the adjoint solve.
struct DeviceHMat;

DeviceHMat *hmat_upload(int Ne,
                        int nD, const int *Dmeta, const long *Doff, const std::complex<double> *Ddata, long Dtotal,
                        int nA, const int *Ameta, const long *Uoff, const long *Voff,
                        const std::complex<double> *Udata, long Utotal, const std::complex<double> *Vdata, long Vtotal,
                        int maxrank);
void hmat_free(DeviceHMat *H);

// y = M*rhs (transposed: M^T*rhs); host rhs/out. For validation.
void hmat_matvec_gpu(DeviceHMat *H, const std::complex<double> *rhs, std::complex<double> *y_out, bool transposed);

// Solve M*x = b (transposed: M^T*x = b) with device BiCGSTAB (identity precond),
// matching Eigen's BiCGSTAB. Returns iteration count; writes x_out (host).
int solve_gpu(DeviceState *dev, DeviceHMat *H, const std::complex<double> *b,
              std::complex<double> *x_out, bool transposed, double tol, int max_iter);

// ---- GPU-resident H-matrix assembly ----
// Allocate the persistent assembled H-matrix (dev->asm_hmat) from static block
// metadata (block dims + per-block max ranks). Called once at precompute.
void device_setup_assembly(DeviceState *dev,
                           const int *Dmeta, const long *Doff, long Dtot, int nD,
                           const int *Ameta, const long *Uoff, const long *Voff, long Utot, long Vtot, int nA, int gmaxrank);
// Assemble the H-matrix on device (dense + approx-ACA) for wavenumber k; reuses
// the cached geometry. Returns dev->asm_hmat.
DeviceHMat *device_assemble(DeviceState *dev, double k, double aca_tol);

} // namespace cuda
} // namespace bem3d
