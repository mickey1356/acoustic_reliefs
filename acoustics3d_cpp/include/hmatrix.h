#pragma once

#include <complex>
#include <iostream>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "constants.h"

namespace bem3d {
class HMatrix;
inline int mat_mults = 0;
} // namespace bem3d

namespace Eigen {
namespace internal {
template <>
struct traits<bem3d::HMatrix> : public Eigen::internal::traits<Eigen::SparseMatrix<std::complex<double>>> {};
} // namespace internal
} // namespace Eigen

namespace bem3d {

class HMatrix : public Eigen::EigenBase<bem3d::HMatrix> {
public:
    HMatrix(int Ne, bool verbose = false) : Ne(Ne), verbose(verbose) {
        mat_mults = 0;
    }

    void compute_direct_and_approx_blocks_cpu(const std::vector<std::pair<pii, pii>> &direct,
                                              const std::vector<std::pair<pii, pii>> &approx,
                                              double k, const mat3 &Ps, const imat3 &Es,
                                              const mat3 &Cs, const mat3 &Ns, double aca_tol);

    typedef std::complex<double> Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false,
    };

    Eigen::Index rows() const { return Ne; }
    Eigen::Index cols() const { return Ne; }

    // set transpose
    void set_transposed(bool transposed) { this->_transposed = transposed; }
    bool transposed() const { return _transposed; }

    template <typename Rhs>
    Eigen::Product<HMatrix, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const {
        mat_mults++;
        if (verbose)
            std::cout << "\33[2K\r" << mat_mults << " mat mults" << std::flush;
        return Eigen::Product<HMatrix, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    std::vector<cmat> direct_blocks;
    std::vector<std::pair<cmat, cmat>> approx_blocks;
    std::vector<std::pair<pii, pii>> direct;
    std::vector<std::pair<pii, pii>> approx;

private:
    int Ne;
    bool verbose;
    bool _transposed = false;
};

} // namespace bem3d

namespace Eigen {
namespace internal {

// openMP reduction declaration
#ifndef _WIN32
#pragma omp declare reduction(+: bem3d::cvec: omp_out += omp_in) \
    initializer(omp_priv=bem3d::cvec::Zero(omp_orig.size()))
#endif

template <typename Rhs>
struct generic_product_impl<bem3d::HMatrix, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
    : generic_product_impl_base<bem3d::HMatrix, Rhs, generic_product_impl<bem3d::HMatrix, Rhs>> {
    typedef typename Product<bem3d::HMatrix, Rhs>::Scalar Scalar;

    template <typename Dest>
    static void scaleAndAddTo(Dest &dst, const bem3d::HMatrix &lhs, const Rhs &rhs, const Scalar &alpha) {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
        // assert(alpha == Scalar(1) && "scaling is not implemented");
        // EIGEN_ONLY_USED_FOR_DEBUG(alpha);
        if (lhs.transposed()) {
            bem3d::cvec y_approx = bem3d::cvec::Zero(lhs.rows()) - 0.5 * rhs;

#ifndef _WIN32
#pragma omp parallel for reduction(+: y_approx)
#endif
            for (int i = 0; i < lhs.direct.size(); i++) {
                auto [rs, re] = lhs.direct[i].first;
                auto [cs, ce] = lhs.direct[i].second;
                y_approx(Eigen::seq(cs, ce - 1)) += lhs.direct_blocks[i].transpose() * rhs(Eigen::seq(rs, re - 1));
            }

#ifndef _WIN32
#pragma omp parallel for reduction(+: y_approx)
#endif
            for (int i = 0; i < lhs.approx.size(); i++) {
                auto [rs, re] = lhs.approx[i].first;
                auto [cs, ce] = lhs.approx[i].second;
                y_approx(Eigen::seq(cs, ce - 1)) += lhs.approx_blocks[i].second.transpose() * (lhs.approx_blocks[i].first.transpose() * rhs(Eigen::seq(rs, re - 1)));
            }
            dst += y_approx;

        } else {
            bem3d::cvec y_approx = bem3d::cvec::Zero(lhs.cols()) - 0.5 * rhs;

#ifndef _WIN32
#pragma omp parallel for reduction(+: y_approx)
#endif
            for (int i = 0; i < lhs.direct.size(); i++) {
                auto [rs, re] = lhs.direct[i].first;
                auto [cs, ce] = lhs.direct[i].second;
                y_approx(Eigen::seq(rs, re - 1)) += lhs.direct_blocks[i] * rhs(Eigen::seq(cs, ce - 1));
            }

#ifndef _WIN32
#pragma omp parallel for reduction(+: y_approx)
#endif
            for (int i = 0; i < lhs.approx.size(); i++) {
                auto [rs, re] = lhs.approx[i].first;
                auto [cs, ce] = lhs.approx[i].second;
                y_approx(Eigen::seq(rs, re - 1)) += lhs.approx_blocks[i].first * (lhs.approx_blocks[i].second * rhs(Eigen::seq(cs, ce - 1)));
            }
            dst += y_approx;
        }
    }
};

} // namespace internal
} // namespace Eigen