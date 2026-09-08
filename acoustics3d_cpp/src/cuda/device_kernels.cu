// CUDA backend kernels + host launchers.
// Phase 1: single-input kernel evaluation for validation against the CPU golden.

#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <vector>

#include "cuda/adjoint_gpu.h"
#include "cuda/device_math.cuh"
#include "cuda/gpu_common.h"

namespace bem3d {
namespace cuda {

namespace {

__global__ void eval_kernels_kernel(const double *p, const double *q, const double *qn,
                                    const double *src, double k, thrust::complex<double> *out) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    const double *p1 = p + 0, *p2 = p + 3, *p3 = p + 6;
    const double *q1 = q + 0, *q2 = q + 3, *q3 = q + 6;
    double c[3] = {(p1[0] + p2[0] + p3[0]) / 3.0,
                   (p1[1] + p2[1] + p3[1]) / 3.0,
                   (p1[2] + p2[2] + p3[2]) / 3.0};
    out[0] = dm_compute_M<double>(k, c, q1, q2, q3, qn);
    out[1] = dm_dMij_dpy<double>(p1, p2, p3, q1, q2, q3, k);
    out[2] = dm_dMij_dq1y<double>(p1, p2, p3, q1, q2, q3, k);
    out[3] = dm_dMij_dq2y<double>(p1, p2, p3, q1, q2, q3, k);
    out[4] = dm_dMij_dq3y<double>(p1, p2, p3, q1, q2, q3, k);
    out[5] = dm_dbi_dpy<double>(p1, p2, p3, src, k);
}

} // namespace

// ======================= Phase 2/3: DeviceState + der kernels =======================

using cplx = thrust::complex<double>;

struct DeviceState {
    // dims
    int Ne = 0, LL = 0, HH = 0, T = 0, Nv = 0;
    bool fp32 = false; // FP32 compute lever (transcendentals in float, accum in double)
    // static buffers (uploaded once)
    int *d_Es = nullptr;         // Ne*3
    double *d_Ls = nullptr;      // LL*3
    int *d_task_h = nullptr;     // T
    int *d_task_elem = nullptr;  // T
    int *d_task_order = nullptr; // T
    // per-iteration buffers (lazily allocated, reused)
    double *d_Ps = nullptr;      // Nv*3
    double *d_Cs = nullptr;      // Ne*3 (element centroids, per geometry)
    double *d_Ns = nullptr;      // Ne*3 (element normals, per geometry)
    cplx *d_x = nullptr;         // Ne
    cplx *d_lmb = nullptr;       // Ne
    cplx *d_vf = nullptr;        // LL
    double *d_src = nullptr;     // 3
    double *d_der = nullptr;     // HH scratch (reused for A/B/C)
    cplx *d_aca_row = nullptr;   // Ne scratch (ACA row)
    cplx *d_aca_col = nullptr;   // LL scratch (ACA col)
    // BiCGSTAB solve scratch (Ne each) + cuBLAS handle
    cublasHandle_t cublas = nullptr;
    cplx *d_r = nullptr, *d_r0 = nullptr, *d_p = nullptr, *d_v = nullptr;
    cplx *d_s = nullptr, *d_t = nullptr, *d_xsol = nullptr, *d_b = nullptr;
    // warm-start caches: previous forward / adjoint solutions (initial guesses)
    bool warm_start = false;
    cplx *d_warm_fwd = nullptr, *d_warm_bwd = nullptr;
    // block-Jacobi preconditioner (pre-inverted diagonal near-blocks)
    bool use_precond = false;
    int nJ = 0, jpad = 0;      // #diagonal blocks, padded block dim
    int *d_Jmeta = nullptr;    // nJ*2: (rs, n)
    long *d_Joff = nullptr;    // nJ: offset of this block in H.d_Ddata (colmajor n x n)
    cplx *d_Pinv = nullptr;    // nJ*jpad*jpad: inverses (colmajor, ld=jpad)
    cplx *d_Jwork = nullptr;   // nJ*jpad*jpad: factorization workspace
    cplx **d_Jptr = nullptr;   // nJ pointers into d_Jwork (cublas batched)
    cplx **d_Cptr = nullptr;   // nJ pointers into d_Pinv
    int *d_Jpiv = nullptr;     // nJ*jpad pivots
    int *d_Jinfo = nullptr;    // nJ info flags
    cplx *d_phat = nullptr, *d_shat = nullptr; // Ne: preconditioned directions
    // device-resident ACA: U (LL x cap, colmajor), V (cap x Ne, rowmajor) + scratch
    int aca_cap = 0;
    cplx *d_U = nullptr, *d_V = nullptr, *d_cu = nullptr, *d_cv = nullptr;
    char *d_mask = nullptr;      // used-listener mask (LL)
    double *d_bval = nullptr;    // argmax block values (ARGMAX_NB)
    int *d_bidx = nullptr;       // argmax block indices (ARGMAX_NB)
    // persistent GPU-assembled H-matrix (built on device each iteration)
    struct DeviceHMat *asm_hmat = nullptr;
};

static constexpr int ARGMAX_NB = 256;

// cuBLAS complex helpers (thrust::complex<double> and cuDoubleComplex share layout)
static inline cuDoubleComplex *cud(cplx *p) { return reinterpret_cast<cuDoubleComplex *>(p); }
static inline const cuDoubleComplex *cud(const cplx *p) { return reinterpret_cast<const cuDoubleComplex *>(p); }
static inline cuDoubleComplex toC(std::complex<double> z) { return make_cuDoubleComplex(z.real(), z.imag()); }
static inline std::complex<double> fromC(cuDoubleComplex z) { return {cuCreal(z), cuCimag(z)}; }

static constexpr int BLOCK = 256;

// Load a double[3] point and cast to compute type CT (float for the FP32 lever).
template <class CT>
__device__ inline void ld3(CT o[3], const double *p) {
    o[0] = (CT)p[0];
    o[1] = (CT)p[1];
    o[2] = (CT)p[2];
}

// Reduce four per-thread partials across a block into thread 0.
__device__ inline void block_reduce4(double &a, double &b, double &c, double &d) {
    __shared__ double sa[BLOCK], sb[BLOCK], sc[BLOCK], sd[BLOCK];
    int tid = threadIdx.x;
    sa[tid] = a;
    sb[tid] = b;
    sc[tid] = c;
    sd[tid] = d;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sa[tid] += sa[tid + s];
            sb[tid] += sb[tid + s];
            sc[tid] += sc[tid + s];
            sd[tid] += sd[tid + s];
        }
        __syncthreads();
    }
    a = sa[0];
    b = sb[0];
    c = sc[0];
    d = sd[0];
}

// der_B: one block per task, threads stride-reduce over all Ne elements.
// Compute type CT (double or float); accumulation is always double (mixed precision).
template <class CT>
__global__ void der_B_kernel_t(double k, int Ne, const double *Ps, const int *Es,
                               const int *task_h, const int *task_elem, const int *task_order, int T,
                               const cplx *x, const cplx *lmb, double *der_B) {
    int t = blockIdx.x;
    if (t >= T)
        return;
    int h = task_h[t], ei = task_elem[t], order = task_order[t];
    CT p1[3], p2[3], p3[3];
    ld3<CT>(p1, Ps + 3 * Es[3 * ei + 0]);
    ld3<CT>(p2, Ps + 3 * Es[3 * ei + 1]);
    ld3<CT>(p3, Ps + 3 * Es[3 * ei + 2]);
    CT kk = (CT)k;

    double s1r = 0, s1i = 0, s2r = 0, s2i = 0;
    for (int j = threadIdx.x; j < Ne; j += blockDim.x) {
        CT q1[3], q2[3], q3[3];
        ld3<CT>(q1, Ps + 3 * Es[3 * j + 0]);
        ld3<CT>(q2, Ps + 3 * Es[3 * j + 1]);
        ld3<CT>(q3, Ps + 3 * Es[3 * j + 2]);
        // s1 += x[j] * dMij_dpy(p_i, q_j)
        thrust::complex<CT> xj((CT)x[j].real(), (CT)x[j].imag());
        thrust::complex<CT> t1 = xj * dm_dMij_dpy<CT>(p1, p2, p3, q1, q2, q3, kk);
        s1r += (double)t1.real();
        s1i += (double)t1.imag();
        // s2 += lmb[j] * dMij_dq{order}y(q_j, p_i)   (note arg order matches CPU)
        thrust::complex<CT> lj((CT)lmb[j].real(), (CT)lmb[j].imag());
        thrust::complex<CT> t2 = lj * dm_dMij_dqy<CT>(order, q1, q2, q3, p1, p2, p3, kk);
        s2r += (double)t2.real();
        s2i += (double)t2.imag();
    }
    block_reduce4(s1r, s1i, s2r, s2i);
    if (threadIdx.x == 0) {
        cplx s1(s1r, s1i), s2(s2r, s2i);
        double contrib = (s1 * lmb[ei]).real() + (s2 * x[ei]).real();
        atomicAdd(&der_B[h], contrib);
    }
}

// der_A: one block per task, threads stride-reduce over LL listeners.
template <class CT>
__global__ void der_A_kernel_t(double k, int LL, const double *Ps, const int *Es, const double *Ls,
                               const int *task_h, const int *task_elem, const int *task_order, int T,
                               const cplx *vf, const cplx *x, double *der_A) {
    int t = blockIdx.x;
    if (t >= T)
        return;
    int h = task_h[t], j = task_elem[t], order = task_order[t];
    CT q1[3], q2[3], q3[3];
    ld3<CT>(q1, Ps + 3 * Es[3 * j + 0]);
    ld3<CT>(q2, Ps + 3 * Es[3 * j + 1]);
    ld3<CT>(q3, Ps + 3 * Es[3 * j + 2]);
    CT kk = (CT)k;

    double rr = 0, ri = 0, z0 = 0, z1 = 0;
    for (int i = threadIdx.x; i < LL; i += blockDim.x) {
        CT L[3];
        ld3<CT>(L, Ls + 3 * i);
        thrust::complex<CT> vfi((CT)vf[i].real(), (CT)vf[i].imag());
        thrust::complex<CT> tv = vfi * dm_dMij_dqy<CT>(order, L, L, L, q1, q2, q3, kk);
        rr += (double)tv.real();
        ri += (double)tv.imag();
    }
    block_reduce4(rr, ri, z0, z1);
    if (threadIdx.x == 0) {
        cplx res(rr, ri);
        atomicAdd(&der_A[h], (res * x[j]).real());
    }
}

// ACA row: for a fixed listener, compute_M against every element -> out[Ne].
// compute in CT, store as complex<double> (ACA host algebra stays double).
template <class CT>
__global__ void aca_row_kernel_t(double k, int listener_i, int Ne,
                                 const double *Ps, const int *Es, const double *Ns, const double *Ls,
                                 cplx *out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Ne)
        return;
    CT p[3], qa[3], qb[3], qc[3], qn[3];
    ld3<CT>(p, Ls + 3 * listener_i);
    ld3<CT>(qa, Ps + 3 * Es[3 * j + 0]);
    ld3<CT>(qb, Ps + 3 * Es[3 * j + 1]);
    ld3<CT>(qc, Ps + 3 * Es[3 * j + 2]);
    ld3<CT>(qn, Ns + 3 * j);
    thrust::complex<CT> m = dm_compute_M<CT>((CT)k, p, qa, qb, qc, qn);
    out[j] = cplx((double)m.real(), (double)m.imag());
}

// ACA col: for a fixed element, compute_M against every listener -> out[LL].
template <class CT>
__global__ void aca_col_kernel_t(double k, int elem_j, int LL,
                                 const double *Ps, const int *Es, const double *Ns, const double *Ls,
                                 cplx *out) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= LL)
        return;
    CT p[3], qa[3], qb[3], qc[3], qn[3];
    ld3<CT>(p, Ls + 3 * l);
    ld3<CT>(qa, Ps + 3 * Es[3 * elem_j + 0]);
    ld3<CT>(qb, Ps + 3 * Es[3 * elem_j + 1]);
    ld3<CT>(qc, Ps + 3 * Es[3 * elem_j + 2]);
    ld3<CT>(qn, Ns + 3 * elem_j);
    thrust::complex<CT> m = dm_compute_M<CT>((CT)k, p, qa, qb, qc, qn);
    out[l] = cplx((double)m.real(), (double)m.imag());
}

// der_C: one thread per task (no inner reduction).
template <class CT>
__global__ void der_C_kernel_t(double k, const double *Ps, const int *Es, const double *src,
                               const int *task_h, const int *task_elem, int T,
                               const cplx *lmb, double *der_C) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T)
        return;
    int h = task_h[t], i = task_elem[t];
    CT p1[3], p2[3], p3[3], s[3];
    ld3<CT>(p1, Ps + 3 * Es[3 * i + 0]);
    ld3<CT>(p2, Ps + 3 * Es[3 * i + 1]);
    ld3<CT>(p3, Ps + 3 * Es[3 * i + 2]);
    ld3<CT>(s, src);
    thrust::complex<CT> v = dm_dbi_dpy<CT>(p1, p2, p3, s, (CT)k);
    cplx vd((double)v.real(), (double)v.imag());
    atomicAdd(&der_C[h], -(vd * lmb[i]).real());
}

DeviceState *device_state_create() { return new DeviceState(); }

void device_state_destroy(DeviceState *dev) {
    if (!dev)
        return;
    cudaFree(dev->d_Es);
    cudaFree(dev->d_Ls);
    cudaFree(dev->d_task_h);
    cudaFree(dev->d_task_elem);
    cudaFree(dev->d_task_order);
    cudaFree(dev->d_Ps);
    cudaFree(dev->d_Cs);
    cudaFree(dev->d_Ns);
    cudaFree(dev->d_x);
    cudaFree(dev->d_lmb);
    cudaFree(dev->d_vf);
    cudaFree(dev->d_src);
    cudaFree(dev->d_der);
    cudaFree(dev->d_aca_row);
    cudaFree(dev->d_aca_col);
    cudaFree(dev->d_r);
    cudaFree(dev->d_r0);
    cudaFree(dev->d_p);
    cudaFree(dev->d_v);
    cudaFree(dev->d_s);
    cudaFree(dev->d_t);
    cudaFree(dev->d_xsol);
    cudaFree(dev->d_b);
    cudaFree(dev->d_warm_fwd);
    cudaFree(dev->d_warm_bwd);
    cudaFree(dev->d_phat);
    cudaFree(dev->d_shat);
    cudaFree(dev->d_Jmeta);
    cudaFree(dev->d_Joff);
    cudaFree(dev->d_Pinv);
    cudaFree(dev->d_Jwork);
    cudaFree(dev->d_Jptr);
    cudaFree(dev->d_Cptr);
    cudaFree(dev->d_Jpiv);
    cudaFree(dev->d_Jinfo);
    cudaFree(dev->d_U);
    cudaFree(dev->d_V);
    cudaFree(dev->d_cu);
    cudaFree(dev->d_cv);
    cudaFree(dev->d_mask);
    cudaFree(dev->d_bval);
    cudaFree(dev->d_bidx);
    if (dev->asm_hmat)
        hmat_free(dev->asm_hmat);
    if (dev->cublas)
        cublasDestroy(dev->cublas);
    delete dev;
}

void device_set_static(DeviceState *dev, const int *Es, int Ne, const double *Ls, int LL,
                       const int *task_h, const int *task_elem, const int *task_order,
                       int T, int HH, int Nv) {
    dev->Ne = Ne;
    dev->LL = LL;
    dev->HH = HH;
    dev->T = T;
    dev->Nv = Nv;

    BEM_CUDA_CHECK(cudaMalloc(&dev->d_Es, Ne * 3 * sizeof(int)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_Ls, LL * 3 * sizeof(double)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_task_h, T * sizeof(int)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_task_elem, T * sizeof(int)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_task_order, T * sizeof(int)));
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_Es, Es, Ne * 3 * sizeof(int), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_Ls, Ls, LL * 3 * sizeof(double), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_task_h, task_h, T * sizeof(int), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_task_elem, task_elem, T * sizeof(int), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_task_order, task_order, T * sizeof(int), cudaMemcpyHostToDevice));

    // per-iter buffers allocated once here (sizes are fixed across iterations)
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_Ps, Nv * 3 * sizeof(double)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_Cs, Ne * 3 * sizeof(double)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_Ns, Ne * 3 * sizeof(double)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_x, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_lmb, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_vf, LL * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_src, 3 * sizeof(double)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_der, HH * sizeof(double)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_aca_row, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_aca_col, LL * sizeof(cplx)));

    // BiCGSTAB scratch + cuBLAS handle
    if (!dev->cublas)
        cublasCreate(&dev->cublas);
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_r, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_r0, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_p, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_v, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_s, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_t, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_xsol, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_b, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_warm_fwd, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_warm_bwd, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMemset(dev->d_warm_fwd, 0, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMemset(dev->d_warm_bwd, 0, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_phat, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_shat, Ne * sizeof(cplx)));

    // device-resident ACA buffers. Cap the stored rank (ACA converges at rank
    // << LL for these problems); bounds V to cap*Ne.
    dev->aca_cap = std::min(512, std::min(LL - 1, Ne - 1));
    if (dev->aca_cap < 1)
        dev->aca_cap = 1;
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_U, (size_t)LL * dev->aca_cap * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_V, (size_t)dev->aca_cap * Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_cu, dev->aca_cap * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_cv, dev->aca_cap * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_mask, LL * sizeof(char)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_bval, ARGMAX_NB * sizeof(double)));
    BEM_CUDA_CHECK(cudaMalloc(&dev->d_bidx, ARGMAX_NB * sizeof(int)));
}

void device_set_geometry(DeviceState *dev, const double *Ps, const double *Cs, const double *Ns) {
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_Ps, Ps, dev->Nv * 3 * sizeof(double), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_Cs, Cs, dev->Ne * 3 * sizeof(double), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_Ns, Ns, dev->Ne * 3 * sizeof(double), cudaMemcpyHostToDevice));
}

void device_set_fp32(DeviceState *dev, bool fp32) {
    if (dev)
        dev->fp32 = fp32;
}

void device_set_warm(DeviceState *dev, bool warm) {
    if (dev)
        dev->warm_start = warm;
}

void device_set_precond(DeviceState *dev, bool precond) {
    if (dev)
        dev->use_precond = precond;
}

// Reset the warm-start caches to zero (call at the start of a fresh optimization).
void device_reset_warm(DeviceState *dev) {
    if (dev && dev->d_warm_fwd) {
        BEM_CUDA_CHECK(cudaMemset(dev->d_warm_fwd, 0, dev->Ne * sizeof(cplx)));
        BEM_CUDA_CHECK(cudaMemset(dev->d_warm_bwd, 0, dev->Ne * sizeof(cplx)));
    }
}

void aca_row_gpu(DeviceState *dev, double k, int listener_i, std::complex<double> *out) {
    int Ne = dev->Ne;
    int grid = (Ne + BLOCK - 1) / BLOCK;
    if (dev->fp32)
        aca_row_kernel_t<float><<<grid, BLOCK>>>(k, listener_i, Ne, dev->d_Ps, dev->d_Es, dev->d_Ns, dev->d_Ls, dev->d_aca_row);
    else
        aca_row_kernel_t<double><<<grid, BLOCK>>>(k, listener_i, Ne, dev->d_Ps, dev->d_Es, dev->d_Ns, dev->d_Ls, dev->d_aca_row);
    BEM_CUDA_CHECK(cudaGetLastError());
    BEM_CUDA_CHECK(cudaMemcpy(out, dev->d_aca_row, Ne * sizeof(cplx), cudaMemcpyDeviceToHost));
}

void aca_col_gpu(DeviceState *dev, double k, int elem_j, std::complex<double> *out) {
    int LL = dev->LL;
    int grid = (LL + BLOCK - 1) / BLOCK;
    if (dev->fp32)
        aca_col_kernel_t<float><<<grid, BLOCK>>>(k, elem_j, LL, dev->d_Ps, dev->d_Es, dev->d_Ns, dev->d_Ls, dev->d_aca_col);
    else
        aca_col_kernel_t<double><<<grid, BLOCK>>>(k, elem_j, LL, dev->d_Ps, dev->d_Es, dev->d_Ns, dev->d_Ls, dev->d_aca_col);
    BEM_CUDA_CHECK(cudaGetLastError());
    BEM_CUDA_CHECK(cudaMemcpy(out, dev->d_aca_col, LL * sizeof(cplx), cudaMemcpyDeviceToHost));
}

void compute_der_B_gpu(DeviceState *dev, double k, const double * /*Ps set via device_set_geometry*/,
                       const std::complex<double> *x, const std::complex<double> *lmb,
                       double *der_B_out) {
    int Ne = dev->Ne, HH = dev->HH, T = dev->T;
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_x, x, Ne * sizeof(cplx), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_lmb, lmb, Ne * sizeof(cplx), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemset(dev->d_der, 0, HH * sizeof(double)));

    if (dev->fp32)
        der_B_kernel_t<float><<<T, BLOCK>>>(k, Ne, dev->d_Ps, dev->d_Es,
                                            dev->d_task_h, dev->d_task_elem, dev->d_task_order, T,
                                            dev->d_x, dev->d_lmb, dev->d_der);
    else
        der_B_kernel_t<double><<<T, BLOCK>>>(k, Ne, dev->d_Ps, dev->d_Es,
                                             dev->d_task_h, dev->d_task_elem, dev->d_task_order, T,
                                             dev->d_x, dev->d_lmb, dev->d_der);
    BEM_CUDA_CHECK(cudaGetLastError());
    BEM_CUDA_CHECK(cudaDeviceSynchronize());
    BEM_CUDA_CHECK(cudaMemcpy(der_B_out, dev->d_der, HH * sizeof(double), cudaMemcpyDeviceToHost));
}

void compute_der_A_gpu(DeviceState *dev, double k, const double * /*Ps set via device_set_geometry*/,
                       const std::complex<double> *vf, const std::complex<double> *x,
                       double *der_A_out) {
    int Ne = dev->Ne, LL = dev->LL, HH = dev->HH, T = dev->T;
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_vf, vf, LL * sizeof(cplx), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_x, x, Ne * sizeof(cplx), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemset(dev->d_der, 0, HH * sizeof(double)));

    if (dev->fp32)
        der_A_kernel_t<float><<<T, BLOCK>>>(k, LL, dev->d_Ps, dev->d_Es, dev->d_Ls,
                                            dev->d_task_h, dev->d_task_elem, dev->d_task_order, T,
                                            dev->d_vf, dev->d_x, dev->d_der);
    else
        der_A_kernel_t<double><<<T, BLOCK>>>(k, LL, dev->d_Ps, dev->d_Es, dev->d_Ls,
                                             dev->d_task_h, dev->d_task_elem, dev->d_task_order, T,
                                             dev->d_vf, dev->d_x, dev->d_der);
    BEM_CUDA_CHECK(cudaGetLastError());
    BEM_CUDA_CHECK(cudaDeviceSynchronize());
    BEM_CUDA_CHECK(cudaMemcpy(der_A_out, dev->d_der, HH * sizeof(double), cudaMemcpyDeviceToHost));
}

void compute_der_C_gpu(DeviceState *dev, double k, const double * /*Ps set via device_set_geometry*/,
                       const double src[3], const std::complex<double> *lmb, double *der_C_out) {
    int Ne = dev->Ne, HH = dev->HH, T = dev->T;
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_lmb, lmb, Ne * sizeof(cplx), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemcpy(dev->d_src, src, 3 * sizeof(double), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemset(dev->d_der, 0, HH * sizeof(double)));

    int grid = (T + BLOCK - 1) / BLOCK;
    if (dev->fp32)
        der_C_kernel_t<float><<<grid, BLOCK>>>(k, dev->d_Ps, dev->d_Es, dev->d_src,
                                               dev->d_task_h, dev->d_task_elem, T, dev->d_lmb, dev->d_der);
    else
        der_C_kernel_t<double><<<grid, BLOCK>>>(k, dev->d_Ps, dev->d_Es, dev->d_src,
                                                dev->d_task_h, dev->d_task_elem, T, dev->d_lmb, dev->d_der);
    BEM_CUDA_CHECK(cudaGetLastError());
    BEM_CUDA_CHECK(cudaDeviceSynchronize());
    BEM_CUDA_CHECK(cudaMemcpy(der_C_out, dev->d_der, HH * sizeof(double), cudaMemcpyDeviceToHost));
}

// ======================= Device-resident ACA =======================
// Keep U (LL x cap colmajor), V (cap x Ne rowmajor), residuals, pivots and the
// convergence bookkeeping all on device; only a few scalars cross per iteration.

// launch the M-row/col eval kernels into device scratch (no D2H)
static void aca_row_dev(DeviceState *dev, double k, int i) {
    int Ne = dev->Ne, grid = (Ne + BLOCK - 1) / BLOCK;
    if (dev->fp32)
        aca_row_kernel_t<float><<<grid, BLOCK>>>(k, i, Ne, dev->d_Ps, dev->d_Es, dev->d_Ns, dev->d_Ls, dev->d_aca_row);
    else
        aca_row_kernel_t<double><<<grid, BLOCK>>>(k, i, Ne, dev->d_Ps, dev->d_Es, dev->d_Ns, dev->d_Ls, dev->d_aca_row);
}
static void aca_col_dev(DeviceState *dev, double k, int j) {
    int LL = dev->LL, grid = (LL + BLOCK - 1) / BLOCK;
    if (dev->fp32)
        aca_col_kernel_t<float><<<grid, BLOCK>>>(k, j, LL, dev->d_Ps, dev->d_Es, dev->d_Ns, dev->d_Ls, dev->d_aca_col);
    else
        aca_col_kernel_t<double><<<grid, BLOCK>>>(k, j, LL, dev->d_Ps, dev->d_Es, dev->d_Ns, dev->d_Ls, dev->d_aca_col);
}

// v[j] -= sum_{t<iter} U(i,t)*V(t,j)   (v in place)
__global__ void aca_vres_kernel(int Ne, int L, int iter, int i, const cplx *U, const cplx *V, cplx *v) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Ne)
        return;
    cplx acc = v[j];
    for (int t = 0; t < iter; t++)
        acc -= U[(size_t)t * L + i] * V[(size_t)t * Ne + j];
    v[j] = acc;
}
// u[l] -= sum_{t<iter} U(l,t)*V(t,jpiv)   (u in place)
__global__ void aca_ures_kernel(int LL, int Ne, int iter, int jpiv, const cplx *U, const cplx *V, cplx *u) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= LL)
        return;
    cplx acc = u[l];
    for (int t = 0; t < iter; t++)
        acc -= U[(size_t)t * LL + l] * V[(size_t)t * Ne + jpiv];
    u[l] = acc;
}
// V.row(iter)[j] = v[j] / v[jpiv]
__global__ void aca_scaleV_kernel(int Ne, int iter, int jpiv, const cplx *v, cplx *V) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Ne)
        return;
    V[(size_t)iter * Ne + j] = v[j] / v[jpiv];
}
// U.col(iter)[l] = u[l]
__global__ void aca_storeU_kernel(int LL, int iter, const cplx *u, cplx *U) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= LL)
        return;
    U[(size_t)iter * LL + l] = u[l];
}
// argmax over |v|^2 (optionally masked), block-level
__global__ void argmax_blk_kernel(const cplx *v, int N, const char *mask, double *bval, int *bidx) {
    __shared__ double sv[256];
    __shared__ int si[256];
    int tid = threadIdx.x;
    double best = -1.0;
    int bi = 0;
    for (int idx = blockIdx.x * blockDim.x + tid; idx < N; idx += gridDim.x * blockDim.x) {
        if (mask && mask[idx])
            continue;
        double a = v[idx].real() * v[idx].real() + v[idx].imag() * v[idx].imag();
        if (a > best) {
            best = a;
            bi = idx;
        }
    }
    sv[tid] = best;
    si[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) {
            sv[tid] = sv[tid + s];
            si[tid] = si[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        bval[blockIdx.x] = sv[0];
        bidx[blockIdx.x] = si[0];
    }
}
// reduce block results -> bidx[0]/bval[0]
__global__ void argmax_final_kernel(double *bval, int *bidx, int nb) {
    __shared__ double sv[256];
    __shared__ int si[256];
    int tid = threadIdx.x;
    double best = -1.0;
    int bi = 0;
    for (int b = tid; b < nb; b += blockDim.x) {
        if (bval[b] > best) {
            best = bval[b];
            bi = bidx[b];
        }
    }
    sv[tid] = best;
    si[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) {
            sv[tid] = sv[tid + s];
            si[tid] = si[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        bval[0] = sv[0];
        bidx[0] = si[0];
    }
}
// dots = sum_t |cu[t]|*|cv[t]|
__global__ void aca_dots_kernel(const cplx *cu, const cplx *cv, int iter, double *out) {
    __shared__ double s[256];
    int tid = threadIdx.x;
    double acc = 0;
    for (int t = tid; t < iter; t += blockDim.x)
        acc += thrust::abs(cu[t]) * thrust::abs(cv[t]);
    s[tid] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st)
            s[tid] += s[tid + st];
        __syncthreads();
    }
    if (tid == 0)
        out[0] = s[0];
}
__global__ void set_mask_kernel(char *mask, int i) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        mask[i] = 1;
}

// returns rank; fills hU (LL*rank colmajor) and hV (rank*Ne rowmajor)
int aca_resident_gpu(DeviceState *dev, int LL, int Ne, double k, double eps, int max_iter,
                     std::vector<std::complex<double>> &hU, std::vector<std::complex<double>> &hV) {
    int iter_lim = std::min(dev->aca_cap, std::min(max_iter, std::min(LL - 1, Ne - 1)));
    cplx *U = dev->d_U, *V = dev->d_V;
    cplx *v = dev->d_aca_row; // Ne (r_v then residual v)
    cplx *u = dev->d_aca_col; // LL (c_v then residual u)
    cublasHandle_t h = dev->cublas;
    const cuDoubleComplex ONE = make_cuDoubleComplex(1, 0), ZERO = make_cuDoubleComplex(0, 0);

    BEM_CUDA_CHECK(cudaMemset(dev->d_mask, 0, LL * sizeof(char)));
    set_mask_kernel<<<1, 32>>>(dev->d_mask, 0); // listener 0 used

    int i = 0, iter = 0, host_j = 0, host_i = 0;
    double err = 0;
    int gN = (Ne + 255) / 256, gL = (LL + 255) / 256;

    while (iter < iter_lim) {
        aca_row_dev(dev, k, i); // v = compute_M(listener i, all elems)
        aca_vres_kernel<<<gN, 256>>>(Ne, LL, iter, i, U, V, v);
        argmax_blk_kernel<<<ARGMAX_NB, 256>>>(v, Ne, nullptr, dev->d_bval, dev->d_bidx);
        argmax_final_kernel<<<1, 256>>>(dev->d_bval, dev->d_bidx, ARGMAX_NB);
        BEM_CUDA_CHECK(cudaMemcpy(&host_j, dev->d_bidx, sizeof(int), cudaMemcpyDeviceToHost));

        aca_scaleV_kernel<<<gN, 256>>>(Ne, iter, host_j, v, V); // V.row(iter) = v/v[j]

        aca_col_dev(dev, k, host_j); // u = compute_M(all listeners, elem j)
        aca_ures_kernel<<<gL, 256>>>(LL, Ne, iter, host_j, U, V, u);
        aca_storeU_kernel<<<gL, 256>>>(LL, iter, u, U); // U.col(iter) = u

        double un = 0, vn = 0;
        cublasDznrm2(h, LL, cud(U + (size_t)iter * LL), 1, &un);
        cublasDznrm2(h, Ne, cud(V + (size_t)iter * Ne), 1, &vn);
        double un_sq = un * un, vn_sq = vn * vn, dots = 0;
        if (iter > 0) {
            cublasZgemv(h, CUBLAS_OP_T, LL, iter, &ONE, cud(U), LL, cud(U + (size_t)iter * LL), 1, &ZERO, cud(dev->d_cu), 1);
            cublasZgemv(h, CUBLAS_OP_T, Ne, iter, &ONE, cud(V), Ne, cud(V + (size_t)iter * Ne), 1, &ZERO, cud(dev->d_cv), 1);
            aca_dots_kernel<<<1, 256>>>(dev->d_cu, dev->d_cv, iter, dev->d_bval);
            BEM_CUDA_CHECK(cudaMemcpy(&dots, dev->d_bval, sizeof(double), cudaMemcpyDeviceToHost));
        }
        err += un_sq * vn_sq + 2 * dots;

        if (sqrt(un_sq * vn_sq / err) <= eps)
            break; // discard the just-computed column (matches CPU seqN(0,iter))

        argmax_blk_kernel<<<ARGMAX_NB, 256>>>(U + (size_t)iter * LL, LL, dev->d_mask, dev->d_bval, dev->d_bidx);
        argmax_final_kernel<<<1, 256>>>(dev->d_bval, dev->d_bidx, ARGMAX_NB);
        BEM_CUDA_CHECK(cudaMemcpy(&host_i, dev->d_bidx, sizeof(int), cudaMemcpyDeviceToHost));
        i = host_i;
        set_mask_kernel<<<1, 32>>>(dev->d_mask, i);
        iter++;
    }

    hU.resize((size_t)LL * iter);
    hV.resize((size_t)iter * Ne);
    BEM_CUDA_CHECK(cudaMemcpy(hU.data(), U, (size_t)LL * iter * sizeof(cplx), cudaMemcpyDeviceToHost));
    BEM_CUDA_CHECK(cudaMemcpy(hV.data(), V, (size_t)iter * Ne * sizeof(cplx), cudaMemcpyDeviceToHost));
    return iter;
}

// ======================= Solve port: device H-matrix + matvec =======================

__device__ inline void atomicAddC(cplx *addr, cplx val) {
    double *p = reinterpret_cast<double *>(addr); // thrust::complex is {re, im}
    atomicAdd(p, val.real());
    atomicAdd(p + 1, val.imag());
}

struct DeviceHMat {
    int Ne = 0, nD = 0, nA = 0, maxrank = 0;
    int *d_Dmeta = nullptr; // nD*4: rs,re,cs,ce
    long *d_Doff = nullptr; // nD
    cplx *d_Ddata = nullptr;
    int *d_Ameta = nullptr; // nA*5: rs,re,cs,ce,rank(=maxrank for assembled mode)
    long *d_Uoff = nullptr;  // nA
    long *d_Voff = nullptr;  // nA
    cplx *d_Udata = nullptr;
    cplx *d_Vdata = nullptr;
    int *d_Arank = nullptr;  // nA actual ranks (assembled mode); null for uploaded mode
};

// y = -0.5 * rhs (the +/-0.5 I term of the boundary operator)
__global__ void scal_half_kernel(int Ne, const cplx *rhs, cplx *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Ne)
        y[i] = -0.5 * rhs[i];
}

// direct (near) blocks; D is colmajor with (re-rs) rows
__global__ void mv_direct_kernel(const int *Dmeta, const long *Doff, const cplx *Ddata,
                                 const cplx *rhs, cplx *y, bool transposed) {
    int b = blockIdx.x;
    int rs = Dmeta[4 * b + 0], re = Dmeta[4 * b + 1], cs = Dmeta[4 * b + 2], ce = Dmeta[4 * b + 3];
    int nr = re - rs, nc = ce - cs;
    const cplx *D = Ddata + Doff[b];
    if (!transposed) {
        for (int r = threadIdx.x; r < nr; r += blockDim.x) {
            cplx acc(0, 0);
            for (int c = 0; c < nc; c++)
                acc += D[(long)c * nr + r] * rhs[cs + c];
            atomicAddC(&y[rs + r], acc);
        }
    } else {
        for (int c = threadIdx.x; c < nc; c += blockDim.x) {
            cplx acc(0, 0);
            for (int r = 0; r < nr; r++)
                acc += D[(long)c * nr + r] * rhs[rs + r];
            atomicAddC(&y[cs + c], acc);
        }
    }
}

// approx (far) low-rank blocks: y[rs:re] += U*(V*rhs[cs:ce]); U colmajor (re-rs)xrank, V colmajor rank x(ce-cs)
__global__ void mv_approx_kernel(const int *Ameta, const long *Uoff, const long *Voff,
                                 const cplx *Udata, const cplx *Vdata, const int *Arank,
                                 const cplx *rhs, cplx *y, bool transposed) {
    int b = blockIdx.x;
    int rs = Ameta[5 * b + 0], re = Ameta[5 * b + 1], cs = Ameta[5 * b + 2], ce = Ameta[5 * b + 3];
    int Vld = Ameta[5 * b + 4];                        // V colmajor leading dim (=exact rank uploaded, =maxrank assembled)
    int rank = Arank ? Arank[b] : Vld;                 // number of terms to actually sum
    int nr = re - rs, nc = ce - cs;
    const cplx *U = Udata + Uoff[b];
    const cplx *V = Vdata + Voff[b];
    extern __shared__ cplx tsh[]; // length >= rank
    if (!transposed) {
        for (int m = threadIdx.x; m < rank; m += blockDim.x) {
            cplx acc(0, 0);
            for (int c = 0; c < nc; c++)
                acc += V[(long)c * Vld + m] * rhs[cs + c];
            tsh[m] = acc;
        }
        __syncthreads();
        for (int r = threadIdx.x; r < nr; r += blockDim.x) {
            cplx acc(0, 0);
            for (int m = 0; m < rank; m++)
                acc += U[(long)m * nr + r] * tsh[m];
            atomicAddC(&y[rs + r], acc);
        }
    } else {
        for (int m = threadIdx.x; m < rank; m += blockDim.x) {
            cplx acc(0, 0);
            for (int r = 0; r < nr; r++)
                acc += U[(long)m * nr + r] * rhs[rs + r];
            tsh[m] = acc;
        }
        __syncthreads();
        for (int c = threadIdx.x; c < nc; c += blockDim.x) {
            cplx acc(0, 0);
            for (int m = 0; m < rank; m++)
                acc += V[(long)c * Vld + m] * tsh[m];
            atomicAddC(&y[cs + c], acc);
        }
    }
}

// y = M*rhs (or M^T*rhs). All buffers on device.
static void device_matvec(DeviceHMat *H, const cplx *d_rhs, cplx *d_y, bool transposed) {
    int Ne = H->Ne;
    scal_half_kernel<<<(Ne + 255) / 256, 256>>>(Ne, d_rhs, d_y);
    if (H->nD > 0)
        mv_direct_kernel<<<H->nD, 128>>>(H->d_Dmeta, H->d_Doff, H->d_Ddata, d_rhs, d_y, transposed);
    if (H->nA > 0) {
        size_t shmem = (size_t)H->maxrank * sizeof(cplx);
        mv_approx_kernel<<<H->nA, 128, shmem>>>(H->d_Ameta, H->d_Uoff, H->d_Voff,
                                                H->d_Udata, H->d_Vdata, H->d_Arank, d_rhs, d_y, transposed);
    }
    BEM_CUDA_CHECK(cudaGetLastError());
}

DeviceHMat *hmat_upload(int Ne,
                        int nD, const int *Dmeta, const long *Doff, const std::complex<double> *Ddata, long Dtotal,
                        int nA, const int *Ameta, const long *Uoff, const long *Voff,
                        const std::complex<double> *Udata, long Utotal, const std::complex<double> *Vdata, long Vtotal,
                        int maxrank) {
    DeviceHMat *H = new DeviceHMat();
    H->Ne = Ne;
    H->nD = nD;
    H->nA = nA;
    H->maxrank = maxrank;
    if (nD > 0) {
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Dmeta, nD * 4 * sizeof(int)));
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Doff, nD * sizeof(long)));
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Ddata, Dtotal * sizeof(cplx)));
        BEM_CUDA_CHECK(cudaMemcpy(H->d_Dmeta, Dmeta, nD * 4 * sizeof(int), cudaMemcpyHostToDevice));
        BEM_CUDA_CHECK(cudaMemcpy(H->d_Doff, Doff, nD * sizeof(long), cudaMemcpyHostToDevice));
        BEM_CUDA_CHECK(cudaMemcpy(H->d_Ddata, Ddata, Dtotal * sizeof(cplx), cudaMemcpyHostToDevice));
    }
    if (nA > 0) {
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Ameta, nA * 5 * sizeof(int)));
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Uoff, nA * sizeof(long)));
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Voff, nA * sizeof(long)));
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Udata, Utotal * sizeof(cplx)));
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Vdata, Vtotal * sizeof(cplx)));
        BEM_CUDA_CHECK(cudaMemcpy(H->d_Ameta, Ameta, nA * 5 * sizeof(int), cudaMemcpyHostToDevice));
        BEM_CUDA_CHECK(cudaMemcpy(H->d_Uoff, Uoff, nA * sizeof(long), cudaMemcpyHostToDevice));
        BEM_CUDA_CHECK(cudaMemcpy(H->d_Voff, Voff, nA * sizeof(long), cudaMemcpyHostToDevice));
        BEM_CUDA_CHECK(cudaMemcpy(H->d_Udata, Udata, Utotal * sizeof(cplx), cudaMemcpyHostToDevice));
        BEM_CUDA_CHECK(cudaMemcpy(H->d_Vdata, Vdata, Vtotal * sizeof(cplx), cudaMemcpyHostToDevice));
    }
    return H;
}

void hmat_free(DeviceHMat *H) {
    if (!H)
        return;
    cudaFree(H->d_Dmeta);
    cudaFree(H->d_Doff);
    cudaFree(H->d_Ddata);
    cudaFree(H->d_Ameta);
    cudaFree(H->d_Uoff);
    cudaFree(H->d_Voff);
    cudaFree(H->d_Udata);
    cudaFree(H->d_Vdata);
    cudaFree(H->d_Arank);
    delete H;
}

// Debug: y = M*rhs (or M^T) with host rhs/out, for validation against the CPU HMatrix.
void hmat_matvec_gpu(DeviceHMat *H, const std::complex<double> *rhs, std::complex<double> *y_out, bool transposed) {
    int Ne = H->Ne;
    cplx *d_rhs, *d_y;
    BEM_CUDA_CHECK(cudaMalloc(&d_rhs, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMalloc(&d_y, Ne * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMemcpy(d_rhs, rhs, Ne * sizeof(cplx), cudaMemcpyHostToDevice));
    device_matvec(H, d_rhs, d_y, transposed);
    BEM_CUDA_CHECK(cudaDeviceSynchronize());
    BEM_CUDA_CHECK(cudaMemcpy(y_out, d_y, Ne * sizeof(cplx), cudaMemcpyDeviceToHost));
    cudaFree(d_rhs);
    cudaFree(d_y);
}

// ---- block-Jacobi preconditioner (pre-inverted diagonal near-blocks) ----

// Fill the factorization workspace: pad each diagonal block to jpad, form
// M_ii = D_ii - 0.5 I on the top-left n x n, identity on the padding.
__global__ void fill_jwork_kernel(const int *Jmeta, const long *Joff, const cplx *Ddata, cplx *Jwork, int jpad) {
    int b = blockIdx.x;
    int n = Jmeta[2 * b + 1];
    const cplx *src = Ddata + Joff[b]; // colmajor n x n
    cplx *dst = Jwork + (size_t)b * jpad * jpad;
    for (int idx = threadIdx.x; idx < jpad * jpad; idx += blockDim.x) {
        int c = idx / jpad, r = idx % jpad;
        cplx val;
        if (r < n && c < n) {
            val = src[(long)c * n + r];
            if (r == c)
                val -= cplx(0.5, 0.0);
        } else {
            val = (r == c) ? cplx(1.0, 0.0) : cplx(0.0, 0.0);
        }
        dst[(long)c * jpad + r] = val;
    }
}

// w = blockdiag(Pinv) * v  (or Pinv^T for the transposed/adjoint solve)
__global__ void precond_apply_kernel(const int *Jmeta, const cplx *Pinv, int jpad,
                                     const cplx *v, cplx *w, bool transposed) {
    int b = blockIdx.x;
    int rs = Jmeta[2 * b + 0], n = Jmeta[2 * b + 1];
    const cplx *P = Pinv + (size_t)b * jpad * jpad; // colmajor ld=jpad
    for (int r = threadIdx.x; r < n; r += blockDim.x) {
        cplx acc(0, 0);
        for (int c = 0; c < n; c++) {
            cplx pval = transposed ? P[(long)r * jpad + c] : P[(long)c * jpad + r];
            acc += pval * v[rs + c];
        }
        w[rs + r] = acc;
    }
}

// factor + invert the diagonal blocks of the (freshly assembled) H-matrix
static void device_precond_setup(DeviceState *dev, DeviceHMat *H) {
    int nJ = dev->nJ, jpad = dev->jpad;
    if (nJ == 0)
        return;
    fill_jwork_kernel<<<nJ, 256>>>(dev->d_Jmeta, dev->d_Joff, H->d_Ddata, dev->d_Jwork, jpad);
    BEM_CUDA_CHECK(cudaGetLastError());
    cublasZgetrfBatched(dev->cublas, jpad, reinterpret_cast<cuDoubleComplex **>(dev->d_Jptr), jpad,
                        dev->d_Jpiv, dev->d_Jinfo, nJ);
    cublasZgetriBatched(dev->cublas, jpad, reinterpret_cast<cuDoubleComplex *const *>(dev->d_Jptr), jpad,
                        dev->d_Jpiv, reinterpret_cast<cuDoubleComplex **>(dev->d_Cptr), jpad, dev->d_Jinfo, nJ);
}

static void precond_apply(DeviceState *dev, const cplx *v, cplx *w, bool transposed) {
    precond_apply_kernel<<<dev->nJ, 128>>>(dev->d_Jmeta, dev->d_Pinv, dev->jpad, v, w, transposed);
}

// Device BiCGSTAB with identity or block-Jacobi preconditioner, replicating Eigen's BiCGSTAB
// (Eigen/src/IterativeLinearSolvers/BiCGSTAB.h). Solves M*x=b (or M^T*x=b), x0=0.
int solve_gpu(DeviceState *dev, DeviceHMat *H, const std::complex<double> *b,
              std::complex<double> *x_out, bool transposed, double tol, int max_iter) {
    cublasHandle_t h = dev->cublas;
    int n = H->Ne;
    cplx *r = dev->d_r, *r0 = dev->d_r0, *p = dev->d_p, *v = dev->d_v;
    cplx *s = dev->d_s, *t = dev->d_t, *x = dev->d_xsol, *db = dev->d_b;
    const cuDoubleComplex ONE = make_cuDoubleComplex(1, 0), MONE = make_cuDoubleComplex(-1, 0);

    // warm-start cache (previous fwd/adjoint solution); null => cold start (x0=0)
    cplx *warm = dev->warm_start ? (transposed ? dev->d_warm_bwd : dev->d_warm_fwd) : nullptr;

    BEM_CUDA_CHECK(cudaMemcpy(db, b, n * sizeof(cplx), cudaMemcpyHostToDevice));

    double rhs_norm = 0;
    cublasDznrm2(h, n, cud(db), 1, &rhs_norm);
    double rhs_sqnorm = rhs_norm * rhs_norm;
    if (rhs_sqnorm == 0.0) {
        for (int i = 0; i < n; i++)
            x_out[i] = 0.0;
        if (warm)
            BEM_CUDA_CHECK(cudaMemset(warm, 0, n * sizeof(cplx)));
        return 0;
    }

    if (warm) {
        // x0 = cached solution;  r = b - A*x0
        BEM_CUDA_CHECK(cudaMemcpy(x, warm, n * sizeof(cplx), cudaMemcpyDeviceToDevice));
        device_matvec(H, x, r, transposed);
        cublasZscal(h, n, &MONE, cud(r), 1);
        cublasZaxpy(h, n, &ONE, cud(db), 1, cud(r), 1);
    } else {
        BEM_CUDA_CHECK(cudaMemset(x, 0, n * sizeof(cplx)));
        BEM_CUDA_CHECK(cudaMemcpy(r, db, n * sizeof(cplx), cudaMemcpyDeviceToDevice)); // r = b
    }
    BEM_CUDA_CHECK(cudaMemcpy(r0, r, n * sizeof(cplx), cudaMemcpyDeviceToDevice));

    double r0_norm = 0;
    cublasDznrm2(h, n, cud(r0), 1, &r0_norm);
    double r0_sqnorm = r0_norm * r0_norm; // initial residual (== rhs_sqnorm for cold start)
    double tol2 = tol * tol * rhs_sqnorm; // convergence relative to ||b||
    double eps = 2.220446049250313e-16;
    double eps2 = eps * eps;

    std::complex<double> rho(1, 0), alpha(1, 0), w(1, 0);
    BEM_CUDA_CHECK(cudaMemset(v, 0, n * sizeof(cplx)));
    BEM_CUDA_CHECK(cudaMemset(p, 0, n * sizeof(cplx)));

    // block-Jacobi preconditioner (only for the assembled H-matrix)
    bool pc = dev->use_precond && dev->nJ > 0 && H == dev->asm_hmat;
    cplx *phat = dev->d_phat, *shat = dev->d_shat;
    if (pc)
        device_precond_setup(dev, H);

    cuDoubleComplex tmpc;
    int i = 0, restarts = 0;
    double r_sqnorm = r0_sqnorm;

    while (r_sqnorm > tol2 && i < max_iter) {
        std::complex<double> rho_old = rho;
        cublasZdotc(h, n, cud(r0), 1, cud(r), 1, &tmpc);
        rho = fromC(tmpc);

        if (std::abs(rho) < eps2 * r0_sqnorm) {
            // residual became ~orthogonal to r0: restart with r = b - A*x
            device_matvec(H, x, r, transposed);
            cublasZscal(h, n, &MONE, cud(r), 1);
            cublasZaxpy(h, n, &ONE, cud(db), 1, cud(r), 1);
            cublasZcopy(h, n, cud(r), 1, cud(r0), 1);
            double rn = 0;
            cublasDznrm2(h, n, cud(r), 1, &rn);
            r0_sqnorm = rn * rn;
            rho = std::complex<double>(r0_sqnorm, 0);
            if (restarts++ == 0)
                i = 0;
        }

        std::complex<double> beta = (rho / rho_old) * (alpha / w);
        // p = r + beta*(p - w*v)
        cuDoubleComplex nw = toC(-w);
        cublasZaxpy(h, n, &nw, cud(v), 1, cud(p), 1); // p -= w*v
        cuDoubleComplex betac = toC(beta);
        cublasZscal(h, n, &betac, cud(p), 1); // p *= beta
        cublasZaxpy(h, n, &ONE, cud(r), 1, cud(p), 1); // p += r

        // phat = K^-1 p ; v = A*phat  (phat == p when unpreconditioned)
        cplx *pv = p;
        if (pc) {
            precond_apply(dev, p, phat, transposed);
            pv = phat;
        }
        device_matvec(H, pv, v, transposed);

        cublasZdotc(h, n, cud(r0), 1, cud(v), 1, &tmpc);
        alpha = rho / fromC(tmpc);

        // s = r - alpha*v
        cublasZcopy(h, n, cud(r), 1, cud(s), 1);
        cuDoubleComplex nalpha = toC(-alpha);
        cublasZaxpy(h, n, &nalpha, cud(v), 1, cud(s), 1);

        // shat = K^-1 s ; t = A*shat
        cplx *sv = s;
        if (pc) {
            precond_apply(dev, s, shat, transposed);
            sv = shat;
        }
        device_matvec(H, sv, t, transposed);

        double tn = 0;
        cublasDznrm2(h, n, cud(t), 1, &tn);
        double tsq = tn * tn;
        if (tsq > 0.0) {
            cublasZdotc(h, n, cud(t), 1, cud(s), 1, &tmpc);
            w = fromC(tmpc) / tsq;
        } else {
            w = 0.0;
        }

        // x += alpha*phat + w*shat  (preconditioned directions)
        cuDoubleComplex alphac = toC(alpha), wc = toC(w);
        cublasZaxpy(h, n, &alphac, cud(pv), 1, cud(x), 1);
        cublasZaxpy(h, n, &wc, cud(sv), 1, cud(x), 1);

        // r = s - w*t
        cublasZcopy(h, n, cud(s), 1, cud(r), 1);
        cuDoubleComplex nwt = toC(-w);
        cublasZaxpy(h, n, &nwt, cud(t), 1, cud(r), 1);

        double rn2 = 0;
        cublasDznrm2(h, n, cud(r), 1, &rn2);
        r_sqnorm = rn2 * rn2;
        i++;
    }

    if (warm) // cache this solution as the next initial guess
        BEM_CUDA_CHECK(cudaMemcpy(warm, x, n * sizeof(cplx), cudaMemcpyDeviceToDevice));
    BEM_CUDA_CHECK(cudaMemcpy(x_out, x, n * sizeof(cplx), cudaMemcpyDeviceToHost));
    return i;
}

// ======================= GPU-resident H-matrix assembly =======================

// Dense (near) blocks: one CUDA block per H-block, one thread per matrix entry.
// D(r,c) = compute_M(k, Cs[rs+r], element(cs+c)).  D is colmajor (nr rows).
template <class CT>
__global__ void asm_dense_kernel(const int *Dmeta, const long *Doff, cplx *Ddata,
                                 double k, const double *Cs, const double *Ps, const int *Es, const double *Ns) {
    int b = blockIdx.x;
    int rs = Dmeta[4 * b + 0], re = Dmeta[4 * b + 1], cs = Dmeta[4 * b + 2], ce = Dmeta[4 * b + 3];
    int nr = re - rs, nc = ce - cs;
    cplx *D = Ddata + Doff[b];
    long total = (long)nr * nc;
    for (long idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int c = idx / nr, r = idx % nr;
        CT p[3];
        ld3<CT>(p, Cs + 3 * (rs + r));
        int e = cs + c;
        CT qa[3], qb[3], qc[3], qn[3];
        ld3<CT>(qa, Ps + 3 * Es[3 * e + 0]);
        ld3<CT>(qb, Ps + 3 * Es[3 * e + 1]);
        ld3<CT>(qc, Ps + 3 * Es[3 * e + 2]);
        ld3<CT>(qn, Ns + 3 * e);
        thrust::complex<CT> m = dm_compute_M<CT>((CT)k, p, qa, qb, qc, qn);
        D[(long)c * nr + r] = cplx((double)m.real(), (double)m.imag());
    }
}

// compute_M for the approx-block ACA (element index e, eval-point centroid index p_idx)
template <class CT>
__device__ inline cplx asm_M(double k, int p_idx, int e,
                             const double *Cs, const double *Ps, const int *Es, const double *Ns) {
    CT p[3];
    ld3<CT>(p, Cs + 3 * p_idx);
    CT qa[3], qb[3], qc[3], qn[3];
    ld3<CT>(qa, Ps + 3 * Es[3 * e + 0]);
    ld3<CT>(qb, Ps + 3 * Es[3 * e + 1]);
    ld3<CT>(qc, Ps + 3 * Es[3 * e + 2]);
    ld3<CT>(qn, Ns + 3 * e);
    thrust::complex<CT> m = dm_compute_M<CT>((CT)k, p, qa, qb, qc, qn);
    return cplx((double)m.real(), (double)m.imag());
}

// Approx (far) blocks: one CUDA block per H-block runs an ACA with adaptive rank.
// U is colmajor (nr x maxrank), V is rowmajor (maxrank x nc). Writes actual rank
// into Arank[b]. Mirrors compute_ACA_block, minus the non-orthogonality "dots"
// correction (a slightly more conservative but valid stopping rule).
template <class CT>
__global__ void asm_approx_kernel(const int *Ameta, const long *Uoff, const long *Voff,
                                  cplx *Udata, cplx *Vdata, int *Arank, double eps,
                                  double k, const double *Cs, const double *Ps, const int *Es, const double *Ns) {
    int b = blockIdx.x;
    int rs = Ameta[5 * b + 0], re = Ameta[5 * b + 1], cs = Ameta[5 * b + 2], ce = Ameta[5 * b + 3], iter_lim = Ameta[5 * b + 4];
    int M = re - rs, N = ce - cs;
    cplx *U = Udata + Uoff[b];
    cplx *V = Vdata + Voff[b];
    int tid = threadIdx.x, nt = blockDim.x;

    __shared__ double s_val[128];
    __shared__ int s_idx[128];
    __shared__ int used[128]; // pivot rows (iter_lim <= APPROX_ACA_MAX_K = 100)
    __shared__ int s_j, s_i, s_break;
    __shared__ cplx s_vj;
    __shared__ double s_un, s_vn, s_err;

    if (tid == 0) {
        used[0] = 0;
        s_i = 0;
        s_err = 0;
    }
    __syncthreads();

    int rank = iter_lim;
    for (int iter = 0; iter < iter_lim; iter++) {
        int i = s_i;
        // V.row(iter): r_v - U.row(i)*V
        for (int c = tid; c < N; c += nt) {
            cplx m = asm_M<CT>(k, rs + i, cs + c, Cs, Ps, Es, Ns);
            for (int t = 0; t < iter; t++)
                m -= U[(long)t * M + i] * V[(long)c * iter_lim + t];
            V[(long)c * iter_lim + iter] = m;
        }
        __syncthreads();
        // j = argmax |V.row(iter)|
        double best = -1;
        int bi = 0;
        for (int c = tid; c < N; c += nt) {
            cplx z = V[(long)c * iter_lim + iter];
            double a = z.real() * z.real() + z.imag() * z.imag();
            if (a > best) {
                best = a;
                bi = c;
            }
        }
        s_val[tid] = best;
        s_idx[tid] = bi;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s && s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            s_j = s_idx[0];
            s_vj = V[(long)s_j * iter_lim + iter];
        }
        __syncthreads();
        int j = s_j;
        cplx vj = s_vj;
        double vj2 = vj.real() * vj.real() + vj.imag() * vj.imag();
        if (vj2 > 0)
            for (int c = tid; c < N; c += nt)
                V[(long)c * iter_lim + iter] = V[(long)c * iter_lim + iter] / vj;
        __syncthreads();
        // U.col(iter): c_v - U*V.col(j)
        for (int r = tid; r < M; r += nt) {
            cplx m = asm_M<CT>(k, rs + r, cs + j, Cs, Ps, Es, Ns);
            for (int t = 0; t < iter; t++)
                m -= U[(long)t * M + r] * V[(long)j * iter_lim + t];
            U[(long)iter * M + r] = m;
        }
        __syncthreads();
        // norms
        double un = 0, vn = 0;
        for (int r = tid; r < M; r += nt) {
            cplx z = U[(long)iter * M + r];
            un += z.real() * z.real() + z.imag() * z.imag();
        }
        for (int c = tid; c < N; c += nt) {
            cplx z = V[(long)c * iter_lim + iter];
            vn += z.real() * z.real() + z.imag() * z.imag();
        }
        s_val[tid] = un;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s)
                s_val[tid] += s_val[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            s_un = s_val[0];
        __syncthreads();
        s_val[tid] = vn;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s)
                s_val[tid] += s_val[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            s_vn = s_val[0];
            s_err += s_un * s_vn;
            s_break = (sqrt(s_un * s_vn / s_err) <= eps) ? 1 : 0;
        }
        __syncthreads();
        if (s_break) {
            rank = iter; // discard the converging column (matches CPU seqN(0,iter))
            break;
        }
        // next pivot i = argmax_masked |U.col(iter)|
        best = -1;
        bi = 0;
        for (int r = tid; r < M; r += nt) {
            bool isused = false;
            for (int q = 0; q <= iter; q++)
                if (used[q] == r) {
                    isused = true;
                    break;
                }
            if (isused)
                continue;
            cplx z = U[(long)iter * M + r];
            double a = z.real() * z.real() + z.imag() * z.imag();
            if (a > best) {
                best = a;
                bi = r;
            }
        }
        s_val[tid] = best;
        s_idx[tid] = bi;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s && s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            s_i = s_idx[0];
            used[iter + 1] = s_i;
        }
        __syncthreads();
    }
    if (tid == 0)
        Arank[b] = rank < 1 ? 1 : rank;
}

void device_setup_assembly(DeviceState *dev,
                           const int *Dmeta, const long *Doff, long Dtot, int nD,
                           const int *Ameta, const long *Uoff, const long *Voff, long Utot, long Vtot, int nA, int gmaxrank) {
    DeviceHMat *H = new DeviceHMat();
    H->Ne = dev->Ne;
    H->nD = nD;
    H->nA = nA;
    H->maxrank = gmaxrank;
    if (nD > 0) {
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Dmeta, nD * 4 * sizeof(int)));
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Doff, nD * sizeof(long)));
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Ddata, Dtot * sizeof(cplx)));
        BEM_CUDA_CHECK(cudaMemcpy(H->d_Dmeta, Dmeta, nD * 4 * sizeof(int), cudaMemcpyHostToDevice));
        BEM_CUDA_CHECK(cudaMemcpy(H->d_Doff, Doff, nD * sizeof(long), cudaMemcpyHostToDevice));
    }
    if (nA > 0) {
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Ameta, nA * 5 * sizeof(int)));
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Uoff, nA * sizeof(long)));
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Voff, nA * sizeof(long)));
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Udata, Utot * sizeof(cplx)));
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Vdata, Vtot * sizeof(cplx)));
        BEM_CUDA_CHECK(cudaMalloc(&H->d_Arank, nA * sizeof(int)));
        BEM_CUDA_CHECK(cudaMemcpy(H->d_Ameta, Ameta, nA * 5 * sizeof(int), cudaMemcpyHostToDevice));
        BEM_CUDA_CHECK(cudaMemcpy(H->d_Uoff, Uoff, nA * sizeof(long), cudaMemcpyHostToDevice));
        BEM_CUDA_CHECK(cudaMemcpy(H->d_Voff, Voff, nA * sizeof(long), cudaMemcpyHostToDevice));
    }
    dev->asm_hmat = H;

    // block-Jacobi preconditioner metadata: diagonal near-blocks (row cluster == col cluster)
    std::vector<int> Jmeta;
    std::vector<long> Joff;
    int maxn = 0;
    for (int i = 0; i < nD; i++) {
        int rs = Dmeta[4 * i + 0], re = Dmeta[4 * i + 1], cs = Dmeta[4 * i + 2], ce = Dmeta[4 * i + 3];
        if (rs == cs && re == ce) {
            Jmeta.push_back(rs);
            Jmeta.push_back(re - rs);
            Joff.push_back(Doff[i]);
            maxn = std::max(maxn, re - rs);
        }
    }
    dev->nJ = (int)Joff.size();
    dev->jpad = maxn;
    if (dev->nJ > 0) {
        int nJ = dev->nJ;
        size_t bs = (size_t)maxn * maxn;
        BEM_CUDA_CHECK(cudaMalloc(&dev->d_Jmeta, nJ * 2 * sizeof(int)));
        BEM_CUDA_CHECK(cudaMalloc(&dev->d_Joff, nJ * sizeof(long)));
        BEM_CUDA_CHECK(cudaMalloc(&dev->d_Pinv, nJ * bs * sizeof(cplx)));
        BEM_CUDA_CHECK(cudaMalloc(&dev->d_Jwork, nJ * bs * sizeof(cplx)));
        BEM_CUDA_CHECK(cudaMalloc(&dev->d_Jpiv, (size_t)nJ * maxn * sizeof(int)));
        BEM_CUDA_CHECK(cudaMalloc(&dev->d_Jinfo, nJ * sizeof(int)));
        BEM_CUDA_CHECK(cudaMalloc(&dev->d_Jptr, nJ * sizeof(cplx *)));
        BEM_CUDA_CHECK(cudaMalloc(&dev->d_Cptr, nJ * sizeof(cplx *)));
        BEM_CUDA_CHECK(cudaMemcpy(dev->d_Jmeta, Jmeta.data(), nJ * 2 * sizeof(int), cudaMemcpyHostToDevice));
        BEM_CUDA_CHECK(cudaMemcpy(dev->d_Joff, Joff.data(), nJ * sizeof(long), cudaMemcpyHostToDevice));
        std::vector<cplx *> hJptr(nJ), hCptr(nJ);
        for (int i = 0; i < nJ; i++) {
            hJptr[i] = dev->d_Jwork + (size_t)i * bs;
            hCptr[i] = dev->d_Pinv + (size_t)i * bs;
        }
        BEM_CUDA_CHECK(cudaMemcpy(dev->d_Jptr, hJptr.data(), nJ * sizeof(cplx *), cudaMemcpyHostToDevice));
        BEM_CUDA_CHECK(cudaMemcpy(dev->d_Cptr, hCptr.data(), nJ * sizeof(cplx *), cudaMemcpyHostToDevice));
    }
}

DeviceHMat *device_assemble(DeviceState *dev, double k, double aca_tol) {
    DeviceHMat *H = dev->asm_hmat;
    if (H->nD > 0) {
        if (dev->fp32)
            asm_dense_kernel<float><<<H->nD, 128>>>(H->d_Dmeta, H->d_Doff, H->d_Ddata, k, dev->d_Cs, dev->d_Ps, dev->d_Es, dev->d_Ns);
        else
            asm_dense_kernel<double><<<H->nD, 128>>>(H->d_Dmeta, H->d_Doff, H->d_Ddata, k, dev->d_Cs, dev->d_Ps, dev->d_Es, dev->d_Ns);
    }
    if (H->nA > 0) {
        if (dev->fp32)
            asm_approx_kernel<float><<<H->nA, 128>>>(H->d_Ameta, H->d_Uoff, H->d_Voff, H->d_Udata, H->d_Vdata, H->d_Arank, aca_tol, k, dev->d_Cs, dev->d_Ps, dev->d_Es, dev->d_Ns);
        else
            asm_approx_kernel<double><<<H->nA, 128>>>(H->d_Ameta, H->d_Uoff, H->d_Voff, H->d_Udata, H->d_Vdata, H->d_Arank, aca_tol, k, dev->d_Cs, dev->d_Ps, dev->d_Es, dev->d_Ns);
    }
    BEM_CUDA_CHECK(cudaGetLastError());
    BEM_CUDA_CHECK(cudaDeviceSynchronize());
    return H;
}

void eval_kernels_gpu(const double p[9], const double q[9], const double qn[3],
                      const double src[3], double k, std::complex<double> out[6]) {
    double *d_p, *d_q, *d_qn, *d_src;
    thrust::complex<double> *d_out;
    BEM_CUDA_CHECK(cudaMalloc(&d_p, 9 * sizeof(double)));
    BEM_CUDA_CHECK(cudaMalloc(&d_q, 9 * sizeof(double)));
    BEM_CUDA_CHECK(cudaMalloc(&d_qn, 3 * sizeof(double)));
    BEM_CUDA_CHECK(cudaMalloc(&d_src, 3 * sizeof(double)));
    BEM_CUDA_CHECK(cudaMalloc(&d_out, 6 * sizeof(thrust::complex<double>)));

    BEM_CUDA_CHECK(cudaMemcpy(d_p, p, 9 * sizeof(double), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemcpy(d_q, q, 9 * sizeof(double), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemcpy(d_qn, qn, 3 * sizeof(double), cudaMemcpyHostToDevice));
    BEM_CUDA_CHECK(cudaMemcpy(d_src, src, 3 * sizeof(double), cudaMemcpyHostToDevice));

    eval_kernels_kernel<<<1, 1>>>(d_p, d_q, d_qn, d_src, k, d_out);
    BEM_CUDA_CHECK(cudaGetLastError());
    BEM_CUDA_CHECK(cudaDeviceSynchronize());

    // thrust::complex<double> and std::complex<double> share layout {re, im}.
    BEM_CUDA_CHECK(cudaMemcpy(out, d_out, 6 * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    BEM_CUDA_CHECK(cudaFree(d_p));
    BEM_CUDA_CHECK(cudaFree(d_q));
    BEM_CUDA_CHECK(cudaFree(d_qn));
    BEM_CUDA_CHECK(cudaFree(d_src));
    BEM_CUDA_CHECK(cudaFree(d_out));
}

} // namespace cuda
} // namespace bem3d
