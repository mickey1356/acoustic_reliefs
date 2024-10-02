import tomlkit, os, sys

import numpy as np
import tqdm

import torch
import torch.optim

import bemlib3d as bl3d
from pyoptim import losses
from pyoptim import mesher
from pyoptim import helpers as H
from pyoptim import diffmesh as dm

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
def acoustic_gradient(tex, diffbem, diff_uvs, device="cuda"):
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
    return 1 - val, torch.from_numpy(-tex_grad).to(device)

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

# any special preprocessing to the optimization variable
# specifically, padding the edges to try and get the edge vertices to be 0
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
    
def optim_proc(config):
    # dimensions
    hfield_res = config["image"]["hfield_res"]
    edge_border = config["image"]["edge_border"]
    cam_rad = config["image"].get("cam_rad", 1)

    # optimization
    tgt_fname = config["optimization"]["tgt_fname"]
    iters = config["optimization"]["iters"]
    init_val = config["optimization"]["init_val"]
    ac_wt = config["optimization"]["ac_wt"]
    cl_wt = config["optimization"]["cl_wt"]
    sm_wt = config["optimization"]["sm_wt"]
    mh_wt = config["optimization"]["mh_wt"]
    ng_wt = config["optimization"]["ng_wt"]
    vw_wts = config["optimization"]["vw_wts"]

    # renderer weights
    cam_pos = [(np.pi / 2, 0)] + [(np.pi / 4, i / 2 * np.pi) for i in range(4)]

    out_fname = os.path.join(config["out_folder"], config["name"])

    os.makedirs(out_fname, exist_ok=True)

    # save the config
    with open(f"{out_fname}/{config["name"]}.toml", "w") as f:
        tomlkit.dump(config, f)

    save_every = config["optimization"].get("save_every", 25)
    os.makedirs(os.path.join(out_fname, "checkpoints"), exist_ok=True)

    # initialize bem simulation
    # diffbem = bl3d.DiffBEM(cluster_size, radius_factor, freq_bands, n_freqs, approx_ACA_tol, Q_ACA_tol, solver_tol, listener_ds, recompute_matrices)
    diffbem = bl3d.DiffBEM(**config["diffbem"])
    diffbem.silent = True

    # generate mesh (0.6 x 0.6 x 0.15 [w x b x h, default])
    Ps, Es = mesher.box_mesher(**config["dimensions"])

    # the points we want to optimize are those at the top of the box
    diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]
    diff_Ps = Ps[diff_pts_idx]

    # construct cluster tree and set up differentiable points
    # the order of Es is changed, and Ps remains unchanged
    Ps, Es = diffbem.precompute(Ps, Es, diff_pts_idx)

    # initialize diffmesh (for mitsuba renderer)
    diffmesh = dm.ImageDiffMesh(Ps, Es, tgt_fname)

    # we want to parameterize the variables using a heightfield "texture"
    # uv coords are set to be 0,0 at the btm-left (corresponding to -w/2, h, b/2)
    x_max, y_max, z_max = np.max(Ps, axis=0)
    x_min, y_min, z_min = np.min(Ps, axis=0)
    diff_uvs = dm.rect_points_to_uv(diff_Ps, x_min, x_max, z_min, z_max)

    # load target image
    tgt = H.read_image(tgt_fname, hfield_res, hfield_res, format="L")
    tgt_img = np.stack([tgt, tgt, tgt], axis=2)
    # if (np.mean(tgt_img) < np.mean(1 - tgt_img)):
    #     tgt_img = 1 - tgt_img
    tgt_tensor = torch.from_numpy(tgt_img).permute(2, 0, 1).unsqueeze(0).cuda()

    # create clip loss model
    # semantic_loss = losses.ImgImgCLIPLoss()
    semantic_loss = losses.ImgImgLPIPSLoss()

    # set up torch optimization    
    # init with constant value
    hfield_torch = torch.full((hfield_res - 2 * edge_border, hfield_res - 2 * edge_border), fill_value=init_val).cuda()
    # hfield_torch = torch.from_numpy(np.random.rand(hfield_res - 2 * edge_border, hfield_res - 2 * edge_border) * 0.01).cuda()
    
    # set up optimizer
    hfield_torch.requires_grad = True
    opt = torch.optim.Adam([hfield_torch], lr=config["optimization"]["lr"])

    pbar = tqdm.trange(iters, dynamic_ncols=True)
    for it in pbar:
        opt.zero_grad()
        
        # pad the edges to force borders to be 0
        hfield_full = torch.nn.functional.pad(hfield_torch, (edge_border, edge_border, edge_border, edge_border), value=0)

        custom_loss = 0
        custom_grads = torch.zeros_like(hfield_torch).cuda()

        # acoustic gradient
        hfield = hfield_full.cpu().detach().numpy()
        if ac_wt != 0:
            ac_v, ac_g = acoustic_gradient(hfield, diffbem, diff_uvs)
            custom_loss += ac_wt * ac_v
            # ac_g is the full gradient, but we only care about the middle section
            ac_gn = normalize_gradients(ac_g, edge=edge_border)
            custom_grads += ac_wt * ac_gn

        # rendered view gradients
        for ((el, az), wt) in zip(cam_pos, vw_wts):
            vw_v, vw_g = diffmesh.gradient(semantic_loss, hfield, el, az, r=cam_rad)
            custom_loss += wt * vw_v
            # vw_g is the full gradient, but we only care about the middle section
            vw_gn = normalize_gradients(vw_g, edge=edge_border)
            # normalize the gradients
            custom_grads += wt * vw_gn

        # use loss.backward to compute individual gradients
        cl_v = texture_cliploss(hfield_full, tgt_tensor, semantic_loss)
        sm_v = smoothness(hfield_full)
        mh_v = squared_heights(hfield_full)
        ng_v = neg_relu(hfield_full)

        # call backward on indidivual losses for individual gradients
        hfield_torch.grad = None
        cl_v.backward(retain_graph=True)
        cl_gn = normalize_gradients(hfield_torch.grad)
        custom_loss += cl_wt * cl_v.item()
        custom_grads += cl_wt * cl_gn

        hfield_torch.grad = None
        sm_v.backward(retain_graph=True)
        sm_gn = normalize_gradients(hfield_torch.grad)
        custom_loss += sm_wt * sm_v.item()
        custom_grads += sm_wt * sm_gn
        
        hfield_torch.grad = None
        mh_v.backward(retain_graph=True)
        mh_gn = normalize_gradients(hfield_torch.grad)
        custom_loss += mh_wt * mh_v.item()
        custom_grads += mh_wt * mh_gn
        
        hfield_torch.grad = None
        ng_v.backward(retain_graph=True)
        ng_gn = normalize_gradients(hfield_torch.grad)
        custom_loss += ng_wt * ng_v.item()
        custom_grads += ng_wt * ng_gn

        # add in the acoustic gradient
        hfield_torch.grad = normalize_gradients(custom_grads)
        pbar.set_postfix_str(f"Loss: {custom_loss.item():.6f}")

        opt.step()
        # print(f"iter {1+it}: {tl_loss.item():.6f} ({ac_v:.6f}/{ac_gm:.6f} - {rd_v:.6f}/{rd_gm:.6f} - {cl_v:.6f}/{cl_gm:.6f} - {sm_v:.6f}/{sm_gm:.6f} - {mh_v:.6f}/{mh_gm:.6f} - {ng_v:.6f}/{ng_gm:.6f})")

        if ((1 + it) % save_every) == 0:
            with torch.no_grad():
                save_hfield_torch = preprocess_tex(hfield_torch, edge_border)
                save_hfield = save_hfield_torch.cpu().detach().numpy()
                rendered_imgs = [diffmesh.render(save_hfield, *cam, r=cam_rad) for cam in cam_pos]
                H.save_images(f"{out_fname}/checkpoints/renders_{1+it}.png", rendered_imgs + [save_hfield])
                
                save_heights = eval_uv(save_hfield, diff_uvs)
                save_Ps = Ps.copy()
                save_Ps[diff_pts_idx, 1] += save_heights
                H.write_mesh(f"{out_fname}/checkpoints/mesh_{1+it}.obj", save_Ps, Es)


    hfield_torch = preprocess_tex(hfield_torch, edge_border)
    hfield = hfield_torch.cpu().detach().numpy()
    rendered_imgs = [diffmesh.render(hfield, *cam, r=cam_rad) for cam in cam_pos]
    H.save_images(f"{out_fname}/renders.png", rendered_imgs + [hfield])
    
    heights = eval_uv(hfield, diff_uvs)

    Ps[diff_pts_idx, 1] += heights
    H.write_mesh(f"{out_fname}/mesh.obj", Ps, Es)


if __name__ == "__main__":
    configfile = "configs/base.toml" if len(sys.argv) == 1 else sys.argv[1]

    print(f"config: {configfile}")
    with open(configfile, "rb") as f:
        config = tomlkit.load(f)

    optim_proc(config)
