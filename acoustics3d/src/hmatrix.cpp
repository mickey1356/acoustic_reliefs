#include "hmatrix.h"

#include <iostream>

#include "aca.h"
#include "integrate.h"

namespace bem3d {

void HMatrix::compute_direct_and_approx_blocks_cpu(const std::vector<std::pair<pii, pii>> &direct,
                                                   const std::vector<std::pair<pii, pii>> &approx,
                                                   double k, const mat3 &Ps, const imat3 &Es,
                                                   const mat3 &Cs, const mat3 &Ns, double aca_tol) {

    int max_dx = 0, max_dy = 0;
    int max_ax = 0, max_ay = 0, max_r = 0;

    int done = 0;

#pragma omp parallel for
    for (int i = 0; i < direct.size(); i++) {
        auto [rs, re] = direct[i].first;
        auto [cs, ce] = direct[i].second;
        auto D = compute_block(rs, re, cs, ce, k, Ps, Es, Cs, Ns);
        if (D.array().isNaN().sum()) {
            std::cout << std::endl;
            std::cout << "U " << rs << " " << re << " " << cs << " " << ce << std::endl;
            exit(1);
        }

#pragma omp critical
        {
            done++;

            max_dx = std::max(max_dx, re - rs);
            max_dy = std::max(max_dy, ce - cs);

            direct_blocks.push_back(D);
            this->direct.push_back(std::make_pair(std::make_pair(rs, re), std::make_pair(cs, ce)));
            
            if (verbose)
                std::cout << "\33[2K\rDirect blocks: " << done << " out of " << direct.size() << std::flush;
        }
    }
    if (verbose)
        std::cout << std::endl;

    done = 0;

#pragma omp parallel for
    for (int i = 0; i < approx.size(); i++) {
        auto [rs, re] = approx[i].first;
        auto [cs, ce] = approx[i].second;
        auto A = compute_ACA_block(rs, re, cs, ce, k, Ps, Es, Cs, Ns, aca_tol, APPROX_ACA_MAX_K);

        // if (A.first.array().isNaN().sum()) {
        //     std::cout << std::endl;
        //     std::cout << "U " << rs << " " << re << " " << cs << " " << ce << std::endl;
        //     exit(1);
        // } else if (A.second.array().isNaN().sum()) {
        //     std::cout << std::endl;
        //     std::cout << "V " << rs << " " << re << " " << cs << " " << ce << std::endl;
        //     exit(1);
        // }

#pragma omp critical
        {
            done++;

            max_ax = std::max(max_ax, re - rs);
            max_ay = std::max(max_ay, ce - cs);
            max_r = std::max(max_r, static_cast<int>(A.first.cols()));

            approx_blocks.push_back(A);
            this->approx.push_back(std::make_pair(std::make_pair(rs, re), std::make_pair(cs, ce)));

            if (verbose)
                std::cout << "\33[2K\rApprox blocks: " << done << " out of " << approx.size() << std::flush;
        }
    }
    if (verbose)
        std::cout << std::endl;

    if (verbose) std::printf("Max direct dims: %dx%d\nMax approx dims: %dx%d, rank: %d\n", max_dx, max_dy, max_ax, max_ay, max_r);
}

} // namespace bem3d