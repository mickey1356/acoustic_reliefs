#pragma once

#include <string>

#include "constants.h"

namespace bem3d {
int read_obj(const std::string &fname, mat3 &Ps, imat3 &Es);
void write_obj(const std::string &fname, const mat3 &Ps, const imat3 &Es);
} // namespace bem3d