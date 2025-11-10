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
} // namespace bem3d