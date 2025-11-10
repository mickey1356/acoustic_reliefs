import numpy as np
import scipy.special as scp
import matplotlib.pyplot as plt
import pickle

from pyoptim.helpers import read_mesh

import acoustics3d as ac3d


c = 343

def dj(l, z):
    return scp.spherical_jn(l, z, True)

def h(l, z):
    return scp.spherical_jn(l, z) + 1j * scp.spherical_yn(l, z)

def dh(l, z):
    return scp.spherical_jn(l, z, True) + 1j * scp.spherical_yn(l, z, True)

def ps(r, theta, k, a, N=100):
    val = 0
    for n in range(N):
        val -= np.power(1j, n) * (2 * n + 1) * (dj(n, k * a) / dh(n, k * a)) * scp.eval_legendre(n, np.cos(theta)) * h(n, k * r)
    return np.conj(val)

def main():
    # freq = 1000
    # sphere = "sphere_s"
    
    for freq in [2000]:
        for sphere in ["sphere_m"]:
            k = 2 * np.pi * freq / c

            r, a = 10, 1
            bem_pts = 360
            num_points = 50
            thetas = np.linspace(0, 2 * np.pi, num=num_points, endpoint=False)
            pressures = np.array([ps(r, theta, k, a) for theta in thetas])

            # compute element size
            V, F = read_mesh(f"test-data/spheres/{sphere}.obj")
            e1 = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=-1)
            e2 = np.linalg.norm(V[F[:, 1]] - V[F[:, 2]], axis=-1)
            e3 = np.linalg.norm(V[F[:, 2]] - V[F[:, 0]], axis=-1)
            esize = max(np.max(e1), np.max(e2), np.max(e3))

            print((np.sum(e1) + np.sum(e2) + np.sum(e3)) / (len(e1) + len(e2) + len(e3)))

            bem_cmplx = ac3d.sphere(f"test-data/spheres/{sphere}.obj", freq, LL=bem_pts, lrad=r, actual=False)
            bem_thetas = np.linspace(0, 2 * np.pi, num=bem_pts, endpoint=False)

            # close the curve
            bem_closed_curve = list(range(bem_pts)) + [0]

            # quantify the error numerically
            actual_dat = np.array([ps(r, theta, k, a) for theta in bem_thetas])
            mae = np.mean(np.abs(actual_dat - bem_cmplx))
            rmae = np.mean(np.abs(actual_dat - bem_cmplx) / np.abs(actual_dat))

            fig, ax = plt.subplots(dpi=300, subplot_kw={"projection": "polar"}, constrained_layout=True)

            fig.suptitle(f"Scattered pressure, f = {freq} Hz, {sphere} ({esize:.2f} m)")
            ax.set_title(f"MAE: {mae:.6f}   Rel. MAE: {rmae:.6f}", {"fontsize": 8})

            ax.plot(thetas, np.real(pressures), "bs", label="analytic real", markersize=1)
            ax.plot(thetas, np.imag(pressures), "rs", label="analytic imag", markersize=1)    
            ax.plot(bem_thetas[bem_closed_curve], np.real(bem_cmplx)[bem_closed_curve], "b-", label="bem real", linewidth=0.5)
            ax.plot(bem_thetas[bem_closed_curve], np.imag(bem_cmplx)[bem_closed_curve], "r-", label="bem imag", linewidth=0.5)

            ax.tick_params(labelsize=5)
            ax.set_rlabel_position(0)
            ax.yaxis.get_major_locator().base.set_params(nbins=5)
            ax.tick_params(pad=-3)

            fig.legend(loc="lower right")
            fig.savefig(f"outputs/sphere_plots/{sphere}_actual_{freq}_Hz.png")

            dd = {
                "a_res": pressures,
                "a_theta": thetas,
                "b_res": bem_cmplx,
                "b_theta": bem_thetas
            }
            with open(f"outputs/sphere_plots/analytical_{freq}.pkl", "wb") as f:
                pickle.dump(dd, f)


if __name__ == "__main__":
    main()
