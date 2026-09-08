#include "hmatrix.h"

#include <iostream>
#include <vector>

#include "aca.h"
#include "integrate.h"
#include "profiler.h"

#ifdef BEM_USE_CUDA
#include "cuda/adjoint_gpu.h"
#endif

namespace bem3d {

void HMatrix::compute_direct_and_approx_blocks_cpu(const std::vector<std::pair<pii, pii>> &direct,
                                                   const std::vector<std::pair<pii, pii>> &approx,
                                                   double k, const mat3 &Ps, const imat3 &Es,
                                                   const mat3 &Cs, const mat3 &Ns, double aca_tol) {

    int max_dx = 0, max_dy = 0;
    int max_ax = 0, max_ay = 0, max_r = 0;

    int done = 0;

    {
    BEM_SCOPED_TIMER("asm.direct");
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
    } // asm.direct

    done = 0;

    {
    BEM_SCOPED_TIMER("asm.approx");
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
    } // asm.approx

    if (verbose) std::printf("Max direct dims: %dx%d\nMax approx dims: %dx%d, rank: %d\n", max_dx, max_dy, max_ax, max_ay, max_r);
}

#ifdef BEM_USE_CUDA
// Flatten the assembled blocks (all cmat are Eigen ColMajor) and upload to device.
cuda::DeviceHMat *cuda_upload_hmatrix(const HMatrix &H) {
    int nD = H.n_direct();
    std::vector<int> Dmeta(nD * 4);
    std::vector<long> Doff(nD);
    std::vector<std::complex<double>> Ddata;
    // reserve to avoid O(n) reallocations while concatenating block data
    long Dtot = 0;
    for (int i = 0; i < nD; i++)
        Dtot += H.direct_blocks[i].size();
    Ddata.reserve(Dtot);
    for (int i = 0; i < nD; i++) {
        int rs = H.direct[i].first.first, re = H.direct[i].first.second;
        int cs = H.direct[i].second.first, ce = H.direct[i].second.second;
        Dmeta[4 * i + 0] = rs;
        Dmeta[4 * i + 1] = re;
        Dmeta[4 * i + 2] = cs;
        Dmeta[4 * i + 3] = ce;
        Doff[i] = static_cast<long>(Ddata.size());
        const cmat &D = H.direct_blocks[i];
        Ddata.insert(Ddata.end(), D.data(), D.data() + D.size());
    }

    int nA = H.n_approx();
    std::vector<int> Ameta(nA * 5);
    std::vector<long> Uoff(nA), Voff(nA);
    std::vector<std::complex<double>> Udata, Vdata;
    long Utot = 0, Vtot = 0;
    for (int i = 0; i < nA; i++) {
        Utot += H.approx_blocks[i].first.size();
        Vtot += H.approx_blocks[i].second.size();
    }
    Udata.reserve(Utot);
    Vdata.reserve(Vtot);
    int maxrank = 0;
    for (int i = 0; i < nA; i++) {
        int rs = H.approx[i].first.first, re = H.approx[i].first.second;
        int cs = H.approx[i].second.first, ce = H.approx[i].second.second;
        const cmat &U = H.approx_blocks[i].first;  // (re-rs) x rank
        const cmat &V = H.approx_blocks[i].second; // rank x (ce-cs)
        int rank = static_cast<int>(U.cols());
        maxrank = std::max(maxrank, rank);
        Ameta[5 * i + 0] = rs;
        Ameta[5 * i + 1] = re;
        Ameta[5 * i + 2] = cs;
        Ameta[5 * i + 3] = ce;
        Ameta[5 * i + 4] = rank;
        Uoff[i] = static_cast<long>(Udata.size());
        Voff[i] = static_cast<long>(Vdata.size());
        Udata.insert(Udata.end(), U.data(), U.data() + U.size());
        Vdata.insert(Vdata.end(), V.data(), V.data() + V.size());
    }

    return cuda::hmat_upload(static_cast<int>(H.rows()),
                             nD, Dmeta.data(), Doff.data(), Ddata.data(), static_cast<long>(Ddata.size()),
                             nA, Ameta.data(), Uoff.data(), Voff.data(),
                             Udata.data(), static_cast<long>(Udata.size()),
                             Vdata.data(), static_cast<long>(Vdata.size()), maxrank);
}
#endif

} // namespace bem3d