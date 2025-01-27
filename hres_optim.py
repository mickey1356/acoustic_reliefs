import tomlkit, os, sys, pickle
import glob

import numpy as np
import cv2
import tqdm

import torch
import torch.optim

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

def texture_cliploss(tex_torch, tgt_tensor, loss_fn) -> torch.Tensor:
    img = torch.stack([tex_torch, tex_torch, tex_torch], dim=0).unsqueeze(0)
    loss = loss_fn(img, tgt_tensor)
    return loss

def smoothness(tex_torch):
    data_padded = torch.nn.functional.pad(tex_torch, (1, 1, 1, 1), value=0)
    # minimize squared laplacian
    lap = data_padded[:-2, 1:-1] + data_padded[2:, 1:-1] + data_padded[1:-1, :-2] + data_padded[1:-1, 2:] - 4 * data_padded[1:-1, 1:-1]
    loss = torch.mean(lap * lap)
    return loss

def squared_heights(tex_torch):
    loss = torch.mean(tex_torch * tex_torch)
    return loss

def neg_relu(tex_torch):
    loss = torch.mean(torch.nn.functional.relu(-tex_torch))
    return loss

def barrier_loss(tex_torch, vmax, buffer=1e-3):
    # add a small buffer to avoid infs
    vl = vmax + buffer
    loss = -torch.mean(torch.minimum(torch.log(tex_torch + vl), torch.log(vl - tex_torch)))
    return loss

def regularization_loss(tex_torch, initial):
    return torch.mean((tex_torch - initial) ** 2)

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

CAM_POS = [(np.pi / 2, 0)] + [(np.pi / 4, i / 2 * np.pi) for i in range(4)]

ADD_SUBDIVS = 1
DEVICE = "cuda"
ITERS = 50
RG_WT = 5 # original 2.5

def main(folder):

    configfile = glob.glob(os.path.join(folder, "*.toml"))[0]
    hfieldfile = os.path.join(folder, "hfield.npy")

    # get base dim of cuboid
    with open(configfile, "rb") as f:
        config = tomlkit.load(f)

    esize = config["dimensions"]["esize"]
    w = config["dimensions"]["w"]
    b = config["dimensions"]["b"]
    h = config["dimensions"]["h"]
    Ps, Es = mesher.box_mesher(esize, w, b, h)

    diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]
    diff_Ps = Ps[diff_pts_idx]

    # subdivide top faces if needed
    subdiv_top = config["dimensions"].get("subdiv_top", 0) + ADD_SUBDIVS
    if subdiv_top > 0:
        top_Es_idx = np.where(np.all(np.isin(Es, diff_pts_idx), axis=1))[0]
        Ps, Es = mesher.face_subdivision(Ps, Es, top_Es_idx, subdiv_top)
        # the points we want to optimize are those at the top of the box
        diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]
        diff_Ps = Ps[diff_pts_idx]

    x_max, y_max, z_max = np.max(Ps, axis=0)
    x_min, y_min, z_min = np.min(Ps, axis=0)
    diff_uvs = dm.rect_points_to_uv(diff_Ps, x_min, x_max, z_min, z_max)

    # load the initial heightfield
    edge_border = config["image"]["edge_border"]
    cam_rad = config["image"].get("cam_rad", 1)
    hfield = np.load(hfieldfile)

    # remove the border
    hfield = hfield[edge_border:-edge_border, edge_border:-edge_border]
    hfield_res, _ = hfield.shape

    # use a higher resolution heightfield
    hfield_res = hfield_res * (2 ** ADD_SUBDIVS)

    # upsample the original heightfield
    n_hfield = cv2.resize(hfield, (hfield_res, hfield_res))
    
    # initialize diffmesh (for mitsuba renderer)
    semantic_loss = losses.ImgImgCLIPLoss()
    tgt_fname = tgt_fname = config["optimization"]["tgt_fname"]
    diffmesh = dm.ImageDiffMesh(Ps, Es, tgt_fname, semantic_loss)

    tgt = H.read_image(tgt_fname, hfield_res + 2 * edge_border, hfield_res + 2 * edge_border, format="L")
    tgt_img = np.stack([tgt, tgt, tgt], axis=2)
    tgt_tensor = torch.from_numpy(tgt_img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # set up torch optim
    hfield_torch = torch.from_numpy(n_hfield).to(DEVICE)
    init_nhfield = preprocess_tex(hfield_torch, edge_border)

    # set up optimizer
    hfield_torch.requires_grad = True
    opt = torch.optim.Adam([hfield_torch], lr=1e-3)

    # weights from the config file
    weights = {
        "cl_wt": config["optimization"].get("cl_wt", 3),
        "sm_wt": config["optimization"].get("sm_wt", 10),
        "ba_wt": config["optimization"].get("ba_wt", 0.5),
        "ng_wt": config["optimization"].get("ng_wt", 0.5),
        "rg_wt": RG_WT
    }

    vw_wts = config["optimization"].get("vw_wts", [3, 1, 1, 1, 1])
    for i, w in enumerate(vw_wts):
        weights[f"vw_{i}_wt"] = w

    vmax = config["optimization"].get("vmax", -1)
    if vmax <= 0:
        vmax = None

    tracker_dict = { "last_iter": -1 }
    for wt_lbl in weights:
        l = wt_lbl[:-3]
        tracker_dict[f"{l}_v"] = 0
        tracker_dict[f"{l}_g"] = 0

    # set directories
    out_fname = os.path.join(config["out_folder"], "hres", config["name"])
    os.makedirs(out_fname, exist_ok=True)
    os.makedirs(os.path.join(out_fname, "checkpoints"), exist_ok=True)


    pbar = tqdm.trange(ITERS, dynamic_ncols=True)
    for it in pbar:
        opt.zero_grad()

        # pad the edges to force borders to be 0
        hfield_full = preprocess_tex(hfield_torch, edge_border)

        custom_loss = 0
        custom_grads = torch.zeros_like(hfield_torch).to(DEVICE)

        # rendered view gradients
        hfield = hfield_full.cpu().detach().numpy()
        for i, (el, az) in enumerate(CAM_POS):
            vw_v, vw_g = diffmesh.gradient(hfield, el, az, radius=cam_rad, res=512)
            # vw_g is the full gradient, but we only care about the middle section
            tracker_dict[f"vw_{i}_v"] = vw_v
            tracker_dict[f"vw_{i}_g"] = vw_g[edge_border:-edge_border, edge_border:-edge_border].detach().cpu().numpy()

        # call backward on indidivual losses for individual gradients
        hfield_torch.grad = None
        # if vmax:
        #     hfield_clip = (hfield_full + vmax) / (2 * vmax)
        # else:
        #     hfield_clip = hfield_full
        cl_v = texture_cliploss(hfield_full, tgt_tensor, semantic_loss)
        cl_v.backward(retain_graph=True)

        cl_g = hfield_torch.grad
        tracker_dict["cl_v"] = cl_v.item()
        tracker_dict["cl_g"] = cl_g.detach().cpu().numpy()

        hfield_torch.grad = None
        sm_v = smoothness(hfield_full)
        sm_v.backward(retain_graph=True)        
        sm_g = hfield_torch.grad
        tracker_dict["sm_v"] = sm_v.item()
        tracker_dict["sm_g"] = sm_g.detach().cpu().numpy()

        hfield_torch.grad = None
        ng_v = neg_relu(hfield_full)
        ng_v.backward(retain_graph=True)
        ng_g = hfield_torch.grad
        tracker_dict["ng_v"] = ng_v.item()
        tracker_dict["ng_g"] = ng_g.detach().cpu().numpy()

        if vmax:
            hfield_torch.grad = None
            ba_v = barrier_loss(hfield_full, vmax)
            ba_v.backward(retain_graph=True)
            ba_g = hfield_torch.grad
            tracker_dict["ba_v"] = ba_v.item()
            tracker_dict["ba_g"] = ba_g.detach().cpu().numpy()
        
        hfield_torch.grad = None
        rg_v = regularization_loss(hfield_full, init_nhfield)
        rg_v.backward(retain_graph=True)
        rg_g = hfield_torch.grad
        tracker_dict["rg_v"] = rg_v.item()
        tracker_dict["rg_g"] = rg_g.detach().cpu().numpy()

        # for wt_lbl in weights:
        #     l = wt_lbl[:-3]
        #     print(l, np.linalg.norm(tracker_dict[f"{l}_g"]))

        # weight gradients and sum
        for wt_lbl in weights:
            l = wt_lbl[:-3]
            custom_loss += weights[wt_lbl] * tracker_dict[f"{l}_v"]

            grad = torch.from_numpy(tracker_dict[f"{l}_g"]).to(DEVICE)
            grad = normalize_gradients(grad)
            custom_grads += weights[wt_lbl] * grad

        hfield_torch.grad = custom_grads
        opt.step()

        # clamp the vmax
        if vmax:
            with torch.no_grad():
                hfield_torch.clamp_(-vmax, vmax)

        pbar.set_postfix_str(f"Loss: {custom_loss:.6f}")

        if ((1 + it) % 5) == 0:
            with torch.no_grad():
                save_hfield_torch = preprocess_tex(hfield_torch, edge_border)
                save_hfield = save_hfield_torch.cpu().detach().numpy()
                rendered_imgs = [diffmesh.render(save_hfield, *cam, radius=cam_rad, res=512) for cam in CAM_POS]
                H.save_images(f"{out_fname}/checkpoints/renders_{1+it}.png", rendered_imgs + [save_hfield], auto_grid=True)

    
    with torch.no_grad():
        hfield_torch = preprocess_tex(hfield_torch, edge_border)
        hfield = hfield_torch.cpu().detach().numpy()
        np.save(f"{out_fname}/hfield.npy", hfield)
        rendered_imgs = [diffmesh.render(hfield, *cam, radius=cam_rad, res=512) for cam in CAM_POS]
        H.save_images(f"{out_fname}/renders.png", rendered_imgs + [hfield], auto_grid=True)
        
        heights = eval_uv(hfield, diff_uvs)
        Ps[diff_pts_idx, 1] += heights
        H.save_mesh(f"{out_fname}/{config["name"]}.obj", Ps, Es)


if __name__ == "__main__":
    fdrs = [
        # "outputs/ac_cat_0.02_norm",
        # "outputs/ac_cat_0.6_multfreq_sample2",
        # "outputs/ac_cat_0.9_multifreq_sample",
        # "outputs/ac_corgi_0.6_multfreq_sample",
        # "outputs/ac_corgi_0.9_multfreq_sample",
        # "outputs/ac_matterhorn_0.6_multfreq_sample2",
        # "outputs/ac_matterhorn2_0.9_multifreq_sample",
        # "outputs/ac_merlion_0.6_emap2",
        # "outputs/ac_trees_0.6",
        # "outputs/ac_waves_square2_0.9_multifreq_sample",
        # "outputs/ac_merlion_0.9",
        # "outputs/ac_mountains_0.9_multifreq_sample",
        # "outputs/ac_peppers_0.9_multifreq_sample",
        # "outputs/ac_fuji_0.9_multifreq_sample",
        # "outputs/ac_trees_0.9",
        # "outputs/ac_bunny_64",
        "outputs/io_trees_0.6_2"
    ]

    for folder in fdrs:
        print(folder)
        main(folder)
