import numpy as np
import tqdm

import build.acoustics3d as ac3d

from pyoptim import mesher
from pyoptim import helpers as H
from pyoptim import diffmesh as dm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# assume tex and uvs are 0,0 = top-left
def eval_uv(tex, uvs):
    h, w = tex.shape
    _uvs = uvs.copy()
    uf, ui = np.modf(_uvs[:, 0] * w - 0.5)
    vf, vi = np.modf(_uvs[:, 1] * h - 0.5)
    uf[uf < 0] = 1 + uf[uf < 0]
    vf[vf < 0] = 1 + vf[vf < 0]
    ui = np.maximum(ui.astype(int), 0)
    vi = np.maximum(vi.astype(int), 0)
    vals = np.column_stack([tex[vi, ui], tex[vi, np.minimum(ui + 1, w - 1)], tex[np.minimum(vi + 1, h - 1), ui], tex[np.minimum(vi + 1, h - 1), np.minimum(ui + 1, w - 1)]])
    return (1 - vf) * (1 - uf) * vals[:, 0] + (1 - vf) * uf * vals[:, 1] + vf * (1 - uf) * vals[:, 2] + vf * uf * vals[:, 3]

# compute the gradient wrt texture
def acoustic_gradient(tex, diffbem, diff_uvs):
    # compute height values (sampled from tex)
    diff_heights = eval_uv(tex, diff_uvs)
    # obtain gradient wrt diff_pts
    val, grad = diffbem.gradient(diff_heights)
    
    # compute gradient wrt tex
    tex_grad = np.zeros_like(tex)
    h, w = tex.shape
    uvs = diff_uvs.copy()
    uf, ui = np.modf(uvs[:, 0] * w - 0.5)
    vf, vi = np.modf(uvs[:, 1] * h - 0.5)
    uf[uf < 0] = 1 + uf[uf < 0]
    vf[vf < 0] = 1 + vf[vf < 0]
    ui = np.maximum(ui.astype(int), 0)
    vi = np.maximum(vi.astype(int), 0)
    for uii, vii, ufi, vfi, g in zip(ui, vi, uf, vf, grad):
        tex_grad[vii, uii] += (1 - vfi) * (1 - ufi) * g
        tex_grad[vii, min(uii + 1, w - 1)] += (1 - vfi) * ufi * g
        tex_grad[min(vii + 1, h - 1), uii] += vfi * (1 - ufi) * g
        tex_grad[min(vii + 1, h - 1), min(uii + 1, w - 1)] += vfi * ufi * g

    # we want to maximize the value, so we negate this
    return 1 - val, -tex_grad

if __name__ == "__main__":

    n = 64
    esize = 0.02

    diffbem = ac3d.DiffBEM(128, 1.5, [1000], 1, 1e-5, 1e-5, 1e-5, np.array([0, 100, 0]), 50, 5, False)
    diffbem.silent = True
    # diffbem.use_actual = True
    # Ps, Es = H.read_mesh("test-data/boxes/meshed_box_284.obj")
    Ps, Es = mesher.box_mesher(esize, 0.6, 0.6, 0.15)
    print(Ps.shape, Es.shape)

    diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]
    diff_Ps = Ps[diff_pts_idx]
    
    Ps, Es = diffbem.precompute(Ps, Es, diff_pts_idx)

    x_max, y_max, z_max = np.max(Ps, axis=0)
    x_min, y_min, z_min = np.min(Ps, axis=0)
    diff_uvs = dm.rect_points_to_uv(diff_Ps, x_min, x_max, z_min, z_max)

    hfield = np.zeros((n, n), dtype=np.float32)
    # ac_v, ac_g = acoustic_gradient(hfield, diffbem, diff_uvs)

    # diff_heights = np.zeros_like(diff_pts_idx, dtype=float)

    # np.save("outputs/gradients/e0.02_res64/adjoint_actual.npy", ac_g)
    # loaded = np.load("outputs/gradients/e0.04_res32/adjoint.npy")
    # print(np.allclose(loaded, ac_g))

    # diff_heights = eval_uv(hfield, diff_uvs)
    # val = 1 - diffbem.value(diff_heights)
    # print(diff_heights.shape)
    # val, grad = diffbem.gradient(diff_heights)

    # diff_heights[17] += h
    # v1 = diffbem.value(diff_heights)
    # diff_heights[17] -= 2*h
    # v2 = diffbem.value(diff_heights)

    # print(grad[17], (v1 - v2) / (2 * h))

    for hh in [6]:
        h = 10 ** (-hh)
        g = np.zeros((n, n))
        for i in tqdm.trange(n):
            for j in tqdm.trange(n, leave=False):
                hfield2 = hfield.copy()
                hfield2[i, j] += h
                diff_heights = eval_uv(hfield2, diff_uvs)
                v1 = 1 - diffbem.value(diff_heights)
                hfield2[i, j] -= (2 * h)
                diff_heights = eval_uv(hfield2, diff_uvs)
                v2 = 1 - diffbem.value(diff_heights)

                g[i, j] = (v1 - v2) / (2 * h)

        # np.save("outputs/gradients/adjoint.npy", ac_g[:n, :n])
        np.save(f"outputs/gradients/e{esize}_res{n}/fd_1e-{hh}.npy", g)

    # vmax = max(np.max(ac_g[:n, :n]), np.max(g))
    # vmin = min(np.min(ac_g[:n, :n]), np.min(g))

    # plt.figure(dpi=300)

    # fig, axs = plt.subplots(nrows=3, ncols=2)
    # axs = axs.flatten()

    # im = axs[0].imshow(ac_g[:n, :n], cmap="RdBu_r", vmax=vmax, vmin=vmin, interpolation="None")
    # axs[0].axis("off")
    # divider = make_axes_locatable(axs[0])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, orientation='vertical')

    # im = axs[1].imshow(g, cmap="RdBu_r", vmax=vmax, vmin=vmin, interpolation="None")
    # axs[1].axis("off")
    # divider = make_axes_locatable(axs[1])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, orientation='vertical')

    # sqr_err = (ac_g[:n, :n] - g) ** 2
    # im = axs[2].imshow(sqr_err, vmin=0, interpolation="None")
    # axs[2].axis("off")
    # divider = make_axes_locatable(axs[2])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, orientation='vertical')

    # rel_err = (ac_g[:n, :n] - g) / g
    # rel_err = np.nan_to_num(rel_err, nan=0)
    # im = axs[3].imshow(rel_err, vmin=0, vmax=1, interpolation="None")
    # axs[3].axis("off")
    # divider = make_axes_locatable(axs[3])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, orientation='vertical')

    # signs = (ac_g[:n, :n] * g) >= 0
    # im = axs[4].imshow(signs, vmin=0, vmax=1, interpolation="None", cmap="gray")
    # axs[4].set_xticks([])
    # axs[4].set_yticks([])
    # axs[4].set_xticks([], minor=True)
    # axs[4].set_yticks([], minor=True)
    # axs[4].axis("off")
    # divider = make_axes_locatable(axs[4])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, orientation='vertical')

    # axs[5].remove()

    # fig.savefig("t_grad.png")

    # print(ac_g[:n, :n])
    # print(g)