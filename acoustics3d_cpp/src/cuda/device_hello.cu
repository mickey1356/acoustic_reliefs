// Phase 0 build de-risk: a trivial CUDA kernel proving nvcc + nanobind + cudart
// link and run. Not part of the math port; kept as a permanent smoke test.

#include <cuda_runtime.h>

#include "cuda/gpu_common.h"

namespace bem3d {
namespace cuda {

namespace {
__global__ void selftest_kernel(double *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(out, 1.0);
    }
}
} // namespace

// Launches a kernel that atomically adds 1.0 exactly `n` times and returns the
// sum (== n). Also exercises H<->D copy and error checking.
double device_selftest(int n) {
    double *d_out = nullptr;
    BEM_CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));
    BEM_CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));

    int block = 256;
    int grid = (n + block - 1) / block;
    selftest_kernel<<<grid, block>>>(d_out, n);
    BEM_CUDA_CHECK(cudaGetLastError());
    BEM_CUDA_CHECK(cudaDeviceSynchronize());

    double h_out = 0.0;
    BEM_CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
    BEM_CUDA_CHECK(cudaFree(d_out));
    return h_out;
}

int device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess)
        return 0;
    return count;
}

} // namespace cuda
} // namespace bem3d
