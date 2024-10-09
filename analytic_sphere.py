import numpy as np
import scipy.special as scp

import matplotlib.pyplot as plt

import build.acoustics3d as ac3d

c = 343

def j(l, z):
    return np.sqrt(np.pi / (2 * z)) * scp.jn(l + 0.5, z)

def h(l, z):
    return np.sqrt(np.pi / (2 * z)) * scp.hankel1(l + 0.5, z)

def dj(l, z):
    return l / z * j(l, z) - j(l + 1, z)

def dh(l, z):
    return l / z * h(l, z) - h(l + 1, z)

def ps(r, theta, k, a, N=100):
    val = 0
    for n in range(N):
        val += np.power(1j, n) * (2 * n + 1) * (dj(n, k * a) / dh(n, k * a)) * scp.eval_legendre(n, np.cos(theta)) * h(n, k * r)
    return val

def main():
    freq = 100
    k = 2 * np.pi * freq / c

    r, a = 10, 1
    bem_pts = 200
    num_points = 200
    thetas = np.linspace(0, 2 * np.pi, num=num_points)
    pressures = np.array([ps(r, theta, k, a) for theta in thetas])

    pressures_real = np.real(pressures)
    pressures_imag = np.imag(pressures)

    points_real = np.column_stack([np.abs(pressures_real) * np.cos(thetas), np.abs(pressures_real) * np.sin(thetas)])
    points_imag = np.column_stack([np.abs(pressures_imag) * np.cos(thetas), np.abs(pressures_imag) * np.sin(thetas)])

    plt_R = max(np.max(np.abs(pressures_real)), np.max(np.abs(pressures_imag)))
    # plt_R = np.mean(np.abs(pressures_imag))
    plt_eps = 0.2 * plt_R

    bem_cmplx = ac3d.sphere("test-data/spheres/sphere_m.obj", freq, LL=bem_pts, lrad=r, actual=False)
    bem_dat = np.column_stack([bem_cmplx.real, bem_cmplx.imag])

    bem_thetas = np.linspace(0, 2 * np.pi, num=bem_dat.shape[0])
    bem_points_real = np.column_stack([np.abs(bem_dat[:, 0]) * np.cos(bem_thetas), np.abs(bem_dat[:, 0]) * np.sin(bem_thetas)])
    bem_points_imag = np.column_stack([np.abs(bem_dat[:, 1]) * np.cos(bem_thetas), np.abs(bem_dat[:, 1]) * np.sin(bem_thetas)])

    # quantify the error numerically
    actual_dat = np.array([ps(r, theta, k, a) for theta in bem_thetas])
    act_real = np.abs(np.real(actual_dat))
    act_imag = np.abs(np.imag(actual_dat))
    bem_real = np.abs(bem_dat[:, 0])
    bem_imag = np.abs(bem_dat[:, 1])
    rel_err_real = (act_real - bem_real) / act_real
    rel_err_imag = (act_imag - bem_imag) / act_imag

    plt.figure(figsize=(8, 8), dpi=200)
    plt.title(f"freq = {freq:.2f} Hz\nka = {k * a:.4f}\n" \
              f"mae: {np.mean(np.abs(act_real - bem_real)):.5f} - {np.mean(np.abs(act_imag - bem_imag)):.5f}\n" \
              f"rel mae: {np.mean(rel_err_real):.5f} - {np.mean(rel_err_imag):.5f}")
    plt.plot(plt_R * np.cos(thetas), plt_R * np.sin(thetas), 'k:', label=f"r={plt_R:.2f}")
    plt.plot(points_real[:, 0], points_real[:, 1], 'b-', label="analytic real")
    plt.plot(points_imag[:, 0], points_imag[:, 1], 'r-', label="analytic imag")
    plt.plot(bem_points_real[:, 0], bem_points_real[:, 1], 'bs', markersize=3, label="bem real")
    plt.plot(bem_points_imag[:, 0], bem_points_imag[:, 1], 'rs', markersize=3, label="bem imag")
    plt.xlim(-plt_R - plt_eps, plt_R + plt_eps)
    plt.ylim(-plt_R - plt_eps, plt_R + plt_eps)
    plt.axvline(x=0, linestyle='--', linewidth=0.5, color='k')
    plt.axhline(y=0, linestyle='--', linewidth=0.5, color='k')
    plt.legend()
    plt.savefig("t_sphere.png")


if __name__ == "__main__":
    main()

    # bempp_sim(50)
