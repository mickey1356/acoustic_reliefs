import tomlkit, os, sys, pickle

import numpy as np
import tqdm

import torch
import torch.optim

import build.acoustics3d as ac3d

from pyoptim import losses
from pyoptim import mesher
from pyoptim import helpers as H
from pyoptim import diffmesh as dm


CAM_POS = [(np.pi / 2, 0)] + [(np.pi / 4, i / 2 * np.pi) for i in range(4)]
DEVICE = "cuda"
SEED = 42

np.random.seed(SEED)

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
def acoustic_gradient(tex, diffbem, diff_uvs, device=DEVICE):
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

def barrier_loss(tex_torch, vmax, buffer=1e-3):
    # add a small buffer to avoid infs
    vl = vmax + buffer
    loss = -torch.mean(torch.minimum(torch.log(tex_torch + vl), torch.log(vl - tex_torch)))
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

    guidance_type = config["optimization"].get("guidance_type", "acoustics_only")
    normalize_grads = config["optimization"].get("normalize_grads", True)

    out_fname = os.path.join(config["out_folder"], config["name"])

    os.makedirs(out_fname, exist_ok=True)

    print(f"Save location: {config["name"]}")
    # save the config
    with open(f"{out_fname}/{config["name"]}.toml", "w") as f:
        tomlkit.dump(config, f)

    save_every = config["optimization"].get("save_every", 50)
    os.makedirs(os.path.join(out_fname, "checkpoints"), exist_ok=True)

    # initialize bem simulation
    # diffbem = bl3d.DiffBEM(cluster_size, radius_factor, freq_bands, n_freqs, approx_ACA_tol, Q_ACA_tol, solver_tol, listener_ds, recompute_matrices)
    diffbem_cfg = {
        "cluster_size": config["diffbem"].get("cluster_size", 128),
        "radius_factor": config["diffbem"].get("radius_factor", 1.5),
        "freq_bands": config["diffbem"].get("freq_bands", [1000]),
        "n_freqs": config["diffbem"].get("n_freqs", 1),
        "approx_ACA_tol": config["diffbem"].get("approx_ACA_tol", 1e-5),
        "Q_ACA_tol": config["diffbem"].get("Q_ACA_tol", 1e-5),
        "solver_tol": config["diffbem"].get("solver_tol", 1e-5),
        "src_pt": np.array(config["diffbem"].get("src_pt", [0, 100, 0])),
        "listener_radius": config["diffbem"].get("listener_radius", 50),
        "listener_ds": config["diffbem"].get("listener_ds", 5),
        "recompute_matrices": config["diffbem"].get("recompute_matrices", False),
    }
    diffbem = ac3d.DiffBEM(**diffbem_cfg)
    diffbem.silent = True

    sample_freq = config["diffbem"].get("sample_freq", False)
    if sample_freq:
        freq_weights = 1 / np.array(diffbem_cfg["freq_bands"])
        freq_weights /= np.sum(freq_weights)
        sampled_freqs = np.random.choice(diffbem_cfg["freq_bands"], size=(iters, ), p=freq_weights)

    # generate mesh (0.6 x 0.6 x 0.15 [w x b x h, default])
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

    if hfield_res <= 0:
        hfield_res = 2 * int(np.ceil(np.sqrt(len(diff_pts_idx))))
    print(f"Heightfield resolution: {hfield_res}")

    # construct cluster tree and set up differentiable points
    # the order of Es is changed, and Ps remains unchanged
    Ps, Es = diffbem.precompute(Ps, Es, diff_pts_idx)

    # we want to parameterize the variables using a heightfield "texture"
    # uv coords are set to be 0,0 at the btm-left (corresponding to -w/2, h, b/2)
    x_max, y_max, z_max = np.max(Ps, axis=0)
    x_min, y_min, z_min = np.min(Ps, axis=0)
    diff_uvs = dm.rect_points_to_uv(diff_Ps, x_min, x_max, z_min, z_max)

    # load target image
    if guidance_type in ["image", "image_only"]:
        # create img-img loss
        semantic_loss = losses.ImgImgCLIPLoss()
        # semantic_loss = losses.ImgImgLPIPSLoss()

        # initialize diffmesh (for mitsuba renderer)
        diffmesh = dm.ImageDiffMesh(Ps, Es, tgt_fname, semantic_loss)

        tgt = H.read_image(tgt_fname, hfield_res, hfield_res, format="L")
        tgt_img = np.stack([tgt, tgt, tgt], axis=2)
        tgt_tensor = torch.from_numpy(tgt_img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    elif guidance_type in ["text", "text_only"]:
        pass
    
    elif guidance_type in ["acoustics_only"]:
        diffmesh = dm.DiffMesh(Ps, Es)

    # set up torch optimization    
    # init with constant value
    hfield_torch = torch.full((hfield_res - 2 * edge_border, hfield_res - 2 * edge_border), fill_value=init_val).to(DEVICE)
    
    # set up optimizer
    hfield_torch.requires_grad = True
    opt = torch.optim.Adam([hfield_torch], lr=config["optimization"]["lr"])

    # hfield = preprocess_tex(hfield_torch, edge_border).cpu().detach().numpy()
    # imgs = [diffmesh.render(hfield, e, a, radius=cam_rad) for e, a in CAM_POS]
    # H.save_images("t_init.png", imgs, auto_grid=True)
    # imgs = [diffmesh.check_ref(e, a, radius=cam_rad) for e, a in CAM_POS]
    # H.save_images("t_ref.png", imgs, auto_grid=True)
    # exit()

    tracker_dict = { "last_iter": -1 }
    for wt_lbl in weights:
        l = wt_lbl[:-3]
        tracker_dict[f"{l}_v"] = np.zeros(iters)
        tracker_dict[f"{l}_g"] = np.zeros((iters, *hfield_torch.shape))

    savelosses = []

    pbar = tqdm.trange(iters, dynamic_ncols=True)
    for it in pbar:
        opt.zero_grad()
        
        # pad the edges to force borders to be 0
        hfield_full = preprocess_tex(hfield_torch, edge_border)

        custom_loss = 0
        custom_grads = torch.zeros_like(hfield_torch).to(DEVICE)

        # acoustic gradient
        hfield = hfield_full.cpu().detach().numpy()

        f = ",".join([str(f) for f in diffbem_cfg["freq_bands"]])
        if sample_freq:
            diffbem.set_band(sampled_freqs[it])
            f = sampled_freqs[it]

        # these guidance types require the acoustics gradient
        if guidance_type in ["acoustics_only", "image", "text"]:
            if weights["ac_wt"] != 0:
                ac_v, ac_g = acoustic_gradient(hfield, diffbem, diff_uvs)
                # ac_g is the full gradient, but we only care about the middle section
                tracker_dict["ac_v"][it] = ac_v
                tracker_dict["ac_g"][it] = ac_g[edge_border:-edge_border, edge_border:-edge_border].detach().cpu().numpy()


        # these guidance types use the diff mesh 
        if guidance_type in ["image_only", "text_only", "image", "text"]:
            # rendered view gradients
            for i, (el, az) in enumerate(CAM_POS):
                vw_v, vw_g = diffmesh.gradient(hfield, el, az, radius=cam_rad)
                # vw_g is the full gradient, but we only care about the middle section
                tracker_dict[f"vw_{i}_v"][it] = vw_v
                tracker_dict[f"vw_{i}_g"][it] = vw_g[edge_border:-edge_border, edge_border:-edge_border].detach().cpu().numpy()

            # call backward on indidivual losses for individual gradients
            hfield_torch.grad = None
            # if vmax:
            #     hfield_clip = (hfield_full + vmax) / (2 * vmax)
            # else:
            #     hfield_clip = hfield_full
            cl_v = texture_cliploss(hfield_full, tgt_tensor, semantic_loss)
            cl_v.backward(retain_graph=True)

            cl_g = hfield_torch.grad
            tracker_dict["cl_v"][it] = cl_v.item()
            tracker_dict["cl_g"][it] = cl_g.detach().cpu().numpy()


        hfield_torch.grad = None
        sm_v = smoothness(hfield_full)
        sm_v.backward(retain_graph=True)        
        sm_g = hfield_torch.grad
        tracker_dict["sm_v"][it] = sm_v.item()
        tracker_dict["sm_g"][it] = sm_g.detach().cpu().numpy()
        
        # hfield_torch.grad = None
        # mh_v = squared_heights(hfield_full)
        # mh_v.backward(retain_graph=True)
        # mh_g = hfield_torch.grad
        # mh_gn = normalize_gradients(hfield_torch.grad)
        # custom_loss += mh_wt * mh_v.item()
        # custom_grads += mh_wt * mh_gn
        
        hfield_torch.grad = None
        ng_v = neg_relu(hfield_full)
        ng_v.backward(retain_graph=True)
        ng_g = hfield_torch.grad
        tracker_dict["ng_v"][it] = ng_v.item()
        tracker_dict["ng_g"][it] = ng_g.detach().cpu().numpy()

        if vmax:
            hfield_torch.grad = None
            ba_v = barrier_loss(hfield_full, vmax)
            ba_v.backward(retain_graph=True)
            ba_g = hfield_torch.grad
            tracker_dict["ba_v"][it] = ba_v.item()
            tracker_dict["ba_g"][it] = ba_g.detach().cpu().numpy()

        
        # weight gradients and sum
        for wt_lbl in weights:
            l = wt_lbl[:-3]
            custom_loss += weights[wt_lbl] * tracker_dict[f"{l}_v"][it]

            grad = torch.from_numpy(tracker_dict[f"{l}_g"][it]).to(DEVICE)
            if normalize_grads:
                grad = normalize_gradients(grad)
            custom_grads += weights[wt_lbl] * grad

        # add in the acoustic gradient
        hfield_torch.grad = custom_grads

        opt.step()

        # clamp the vmax
        if vmax:
            with torch.no_grad():
                hfield_torch.clamp_(-vmax, vmax)
        
        # print(f"iter {1+it}: {tl_loss.item():.6f} ({ac_v:.6f}/{ac_gm:.6f} - {rd_v:.6f}/{rd_gm:.6f} - {cl_v:.6f}/{cl_gm:.6f} - {sm_v:.6f}/{sm_gm:.6f} - {mh_v:.6f}/{mh_gm:.6f} - {ng_v:.6f}/{ng_gm:.6f})")
        pbar.set_postfix_str(f"Loss: {custom_loss:.6f} - Freq: {f}")
        savelosses.append(custom_loss.item())

        if ((1 + it) % save_every) == 0:
            with torch.no_grad():
                save_hfield_torch = preprocess_tex(hfield_torch, edge_border)
                save_hfield = save_hfield_torch.cpu().detach().numpy()
                rendered_imgs = [diffmesh.render(save_hfield, *cam, radius=cam_rad) for cam in CAM_POS]
                H.save_images(f"{out_fname}/checkpoints/renders_{1+it}.png", rendered_imgs + [save_hfield], auto_grid=True)
                
                # save_heights = eval_uv(save_hfield, diff_uvs)
                # save_Ps = Ps.copy()
                # save_Ps[diff_pts_idx, 1] += save_heights
                # H.save_mesh(f"{out_fname}/checkpoints/mesh_{1+it}.obj", save_Ps, Es)

        # save the tracker dict every iteration (overwrites itself)
        tracker_dict["last_iter"] = it
        # with open(f"{out_fname}/tracker_dict.pkl", "wb") as f:
        #     pickle.dump(tracker_dict, f)

    with torch.no_grad():
        hfield_torch = preprocess_tex(hfield_torch, edge_border)
        hfield = hfield_torch.cpu().detach().numpy()
        np.save(f"{out_fname}/hfield.npy", hfield)
        rendered_imgs = [diffmesh.render(hfield, *cam, radius=cam_rad) for cam in CAM_POS]
        H.save_images(f"{out_fname}/renders.png", rendered_imgs + [hfield], auto_grid=True)
        
        heights = eval_uv(hfield, diff_uvs)
        Ps[diff_pts_idx, 1] += heights
        H.save_mesh(f"{out_fname}/mesh.obj", Ps, Es)
    
    # plot losses and save
    import matplotlib.pyplot as plt
    plt.plot(savelosses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.savefig(f"{out_fname}/losses.png")
    plt.close()


if __name__ == "__main__":
    configfile = "configs/base.toml" if len(sys.argv) == 1 else sys.argv[1]

    print(f"config: {configfile}")
    with open(configfile, "rb") as f:
        config = tomlkit.load(f)

    optim_proc(config)
