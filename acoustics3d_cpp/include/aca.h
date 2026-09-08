#pragma once

#include <unordered_set>
#include <utility>

#include "constants.h"

using namespace bem3d;

namespace bem3d {
int argmax_abs_masked(const cvec &v, const std::unordered_set<int> &used_indices);

int argmax_abs(const cvec &v);

std::pair<cmat, cmat> compute_ACA_block(int rs, int re, int cs, int ce, double k,
                                        const mat3 &Ps, const imat3 &Es,
                                        const mat3 &Cs, const mat3 &Ns,
                                        double eps, int max_iter);

std::pair<cmat, cmat> compute_ACA_listener(int L, int Ne, double k,
                                           const mat3 &Ls, const mat3 &Ps, const imat3 &Es,
                                           const mat3 &Cs, const mat3 &Ns,
                                           double eps, int max_iter, bool verbose = false);

#ifdef BEM_USE_CUDA
namespace cuda {
struct DeviceState;
}
// Same ACA as compute_ACA_listener, but the row/col entry evaluations run on the
// GPU (reusing cached Ps/Es/Ns/Ls in dev). The serial pivot/rank-1 bookkeeping
// stays on host. Geometry must already be uploaded via device_set_geometry.
std::pair<cmat, cmat> compute_ACA_listener_gpu(cuda::DeviceState *dev, int L, int Ne, double k,
                                               double eps, int max_iter, bool verbose = false);
#endif
} // namespace bem3d