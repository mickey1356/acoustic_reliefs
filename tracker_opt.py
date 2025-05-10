import numpy as np
import torch
import torch.optim
import pickle
import glob, os
import tomlkit
import tqdm

import pyoptim.diffmesh as dm
import pyoptim.mesher as mesher
import pyoptim.helpers as H

DEVICE = "cuda"

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
    hfield = torch.nn.functional.pad(tex_torch, (edge_border, edge_border, edge_border, edge_border), value=0)
    return hfield

def normalize_gradients(grad, edge=0):
    if edge != 0:
        g = grad[edge:-edge, edge:-edge]
    else:
        g = grad[:, :]
    
    norm = torch.linalg.norm(g)

    if norm == 0:
        return g
    else:
        return g / norm

def main():
    folder = "outputs/ac_cat_0.6_multfreq_sample2"
    name = "cat_0.6"

    # folder = "outputs/ac_mountains_0.9_multifreq_sample"
    # name = "mountains_0.9"

    os.makedirs(f"outputs/optim_anims/{name}", exist_ok=True)

    with open(f"{folder}/tracker_dict.pkl", "rb") as f:
        td = pickle.load(f)

    configfile = glob.glob(os.path.join(folder, "*.toml"))[0]

    # get base dim of cuboid
    with open(configfile, "rb") as f:
        config = tomlkit.load(f)

    esize = config["dimensions"]["esize"]
    w = config["dimensions"]["w"]
    b = config["dimensions"]["b"]
    h = config["dimensions"]["h"]
    Ps, Es = mesher.box_mesher(esize, w, b, h)

    # the points we want to optimize are those at the top of the box
    diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]
    diff_Ps = Ps[diff_pts_idx]

    # subdivide top faces if needed
    subdiv_top = config["dimensions"].get("subdiv_top", 0)
    if subdiv_top > 0:
        top_Es_idx = np.where(np.all(np.isin(Es, diff_pts_idx), axis=1))[0]
        Ps, Es = mesher.face_subdivision(Ps, Es, top_Es_idx, subdiv_top)
        # the points we want to optimize are those at the top of the box
        diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]
        diff_Ps = Ps[diff_pts_idx]


    x_max, y_max, z_max = np.max(Ps, axis=0)
    x_min, y_min, z_min = np.min(Ps, axis=0)
    diff_uvs = dm.rect_points_to_uv(diff_Ps, x_min, x_max, z_min, z_max)

    weights = {
        "ac_wt": config["optimization"].get("ac_wt", 5),
        "cl_wt": config["optimization"].get("cl_wt", 3),
        "sm_wt": config["optimization"].get("sm_wt", 10),
        "ba_wt": config["optimization"].get("ba_wt", 0.5),
        "ng_wt": config["optimization"].get("ng_wt", 0.5),
        "rl_wt": config["optimization"].get("rl_wt", 3),
    }

    vw_wts = config["optimization"].get("vw_wts", [3, 1, 1, 1, 1])
    for i, w in enumerate(vw_wts):
        weights[f"vw_{i}_wt"] = w

    vmax = config["optimization"].get("vmax", -1)
    if vmax <= 0:
        vmax = None


    hfield_res = config["image"]["hfield_res"]
    edge_border = config["image"]["edge_border"]
    iters = config["optimization"]["iters"]
    init_val = config["optimization"]["init_val"]

    hfield_torch = torch.full((hfield_res - 2 * edge_border, hfield_res - 2 * edge_border), fill_value=init_val).to(DEVICE)
    
    # set up optimizer
    hfield_torch.requires_grad = True
    opt = torch.optim.Adam([hfield_torch], lr=config["optimization"]["lr"])

    pbar = tqdm.trange(iters, dynamic_ncols=True)
    for it in pbar:
        opt.zero_grad()

        custom_grads = torch.zeros_like(hfield_torch).to(DEVICE)

        for wt_lbl in weights:
            l = wt_lbl[:-3]
            # custom_loss += weights[wt_lbl] * td[f"{l}_v"][it]

            grad = torch.from_numpy(td[f"{l}_g"][it]).to(DEVICE)
            grad = normalize_gradients(grad)
            custom_grads += weights[wt_lbl] * grad

        # add in the acoustic gradient
        hfield_torch.grad = custom_grads

        opt.step()

        # clamp the vmax
        if vmax:
            with torch.no_grad():
                hfield_torch.clamp_(-vmax, vmax)
    
        with torch.no_grad():

            hfield_torch_ng = preprocess_tex(hfield_torch, edge_border)
            hfield = hfield_torch_ng.cpu().detach().numpy()
            heights = eval_uv(hfield, diff_uvs)
            nPs = Ps.copy()
            nPs[diff_pts_idx, 1] += heights
            H.save_mesh(f"outputs/optim_anims/{name}/{it}.obj", nPs, Es)
            if it == 2:
                hfield = hfield_torch.cpu().detach().numpy()
                np.save(f"outputs/optim_anims/{name}/{it}.npy", hfield)



if __name__ == "__main__":
    main()