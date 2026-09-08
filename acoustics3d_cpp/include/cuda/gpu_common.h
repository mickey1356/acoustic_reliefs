#pragma once

// Shared host-side declarations + helpers for the CUDA backend.
// Safe to include from plain C++ TUs (no <cuda_runtime.h> pulled in here);
// the BEM_CUDA_CHECK macro is only ever *expanded* inside .cu files, which
// include <cuda_runtime.h> themselves.

#include <sstream>
#include <stdexcept>
#include <string>

// Wrap every CUDA runtime call; throws std::runtime_error with file:line on failure.
#define BEM_CUDA_CHECK(expr)                                                      \
    do {                                                                          \
        cudaError_t _bem_err = (expr);                                            \
        if (_bem_err != cudaSuccess) {                                            \
            std::ostringstream _bem_oss;                                          \
            _bem_oss << "CUDA error " << cudaGetErrorString(_bem_err) << " at "   \
                     << __FILE__ << ":" << __LINE__ << " (" #expr ")";            \
            throw std::runtime_error(_bem_oss.str());                            \
        }                                                                         \
    } while (0)

namespace bem3d {
namespace cuda {

// Phase 0 smoke test: returns (double)n by atomically adding 1.0 n times on device.
double device_selftest(int n = 1);

// Number of visible CUDA devices (0 if none / driver error).
int device_count();

} // namespace cuda
} // namespace bem3d
