import numpy as np

from pyoptim import losses
from pyoptim import mesher
from pyoptim import helpers as H
from pyoptim import diffmesh as dm

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

def preprocess_tex(tex_torch, edge_border):
    hfield = np.pad(tex_torch, edge_border, mode="constant", constant_values=0)
    return hfield


CAM_POS = [(np.pi / 2, 0)] + [(np.pi / 4, i / 2 * np.pi) for i in range(4)]

def main():
    Ps, Es = mesher.box_mesher(0.02, 0.6, 0.6, 0.15)

    # the points we want to optimize are those at the top of the box
    diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]
    diff_Ps = Ps[diff_pts_idx]

    # subdivide top faces if needed
    top_Es_idx = np.where(np.all(np.isin(Es, diff_pts_idx), axis=1))[0]
    Ps, Es = mesher.face_subdivision(Ps, Es, top_Es_idx, 1)
    # the points we want to optimize are those at the top of the box
    diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]
    diff_Ps = Ps[diff_pts_idx]

    x_max, y_max, z_max = np.max(Ps, axis=0)
    x_min, y_min, z_min = np.min(Ps, axis=0)
    diff_uvs = dm.rect_points_to_uv(diff_Ps, x_min, x_max, z_min, z_max)

    # load image
    tex = H.read_image("t_heights2.png", w=256, h=256, format="L")
    # squish tex to lie between ht_low and ht_high
    ht_low = -0.02
    ht_high = 0.02
    tex = (tex - np.min(tex)) / (np.max(tex) - np.min(tex)) * (ht_high - ht_low) + ht_low
    tex = preprocess_tex(tex, 2)

    # diffmesh = dm.DiffMesh(Ps, Es)
    # rendered_imgs = [diffmesh.render(tex, *cam, radius=1) for cam in CAM_POS]
    # H.save_images("rafael_render.png", rendered_imgs + [tex], auto_grid=True)

    heights = eval_uv(tex, diff_uvs)
    Ps[diff_pts_idx, 1] += heights
    H.save_mesh(f"tmp/flowers_low.obj", Ps, Es)

    # H.save_images(f"{out_fname}/checkpoints/renders_{1+it}.png", rendered_imgs + [save_hfield], auto_grid=True)


if __name__ == "__main__":
    main()
