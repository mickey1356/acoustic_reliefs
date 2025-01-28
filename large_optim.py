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

def build_small_mesh(w, b, h=0.15, esize=0.02, ndiv=1, offx=0, offz=0):
    small_Ps, small_Es = mesher.box_mesher(esize, w, b, h)
    small_Ps[:, 0] += offx
    small_Ps[:, 2] += offz
    small_diff_pts_idx = np.where(np.isclose(small_Ps[:, 1], np.max(small_Ps[:, 1])))[0]
    small_top_Es_idx = np.where(np.all(np.isin(small_Es, small_diff_pts_idx), axis=1))[0]
    small_Ps, small_Es = mesher.face_subdivision(small_Ps, small_Es, small_top_Es_idx, ndiv)
    small_diff_pts_idx = np.where(np.isclose(small_Ps[:, 1], np.max(small_Ps[:, 1])))[0]
    small_diff_Ps = small_Ps[small_diff_pts_idx]
    return small_Ps, small_Es, small_diff_Ps, small_diff_pts_idx


CAM_POS = [(np.pi / 2, 0)] + [(np.pi / 4, i / 2 * np.pi) for i in range(4)]
DEVICE = "cuda"
SEED = 42

FREQS = [800, 1000, 1250, 1600, 2000, 2500, 3150]
np.random.seed(SEED)

# for the large optim, we tile multiple 0.6 x 0.6 diffusers each with 128x128 hfields

NUM_W = 3
NUM_B = 2
RES_X = 128
RES_Z = 128
DIM_W = 0.6
DIM_B = 0.6
DIM_H = 0.15

ESIZE = 0.02
NDIV = 0

ITERS = 500
VMAX = 0.07
BORDER = 2

CAM_RAD = 2.5
TGT_FNAME = "test-data/images/waves_32.jpg"
RD_RES = (3 * 128, 2 * 128)

NAME = "waves_32"

WEIGHTS = {
    "ac_wt": 7,
    "cl_wt": 5,
    "sm_wt": 15,
    "ba_wt": 1,
    "ng_wt": 2,
}
VW_WEIGHTS = [7, 2, 2, 2, 2]


def main():
    os.makedirs(f"outputs/large/{NAME}/checkpoints", exist_ok=True)

    tres_x = NUM_W * RES_X
    tres_z = NUM_B * RES_Z
    tw = NUM_W * DIM_W
    tb = NUM_B * DIM_B

    # define hfield
    hfield_torch = torch.zeros((tres_z - 2 * BORDER, tres_x - 2 * BORDER), dtype=torch.float32).to(DEVICE)
    hfield_torch.requires_grad = True
    opt = torch.optim.Adam([hfield_torch], lr=1e-3)


    # sample freqs
    freq_weights = 1 / np.array(FREQS)
    freq_weights /= np.sum(freq_weights)
    sampled_freqs = np.random.choice(FREQS, size=(ITERS, ), p=freq_weights)

    # setup config
    diffbem_cfg = {
        "cluster_size": 128,
        "radius_factor": 1.5,
        "freq_bands": [1000],
        "n_freqs": 1,
        "approx_ACA_tol": 1e-5,
        "Q_ACA_tol": 1e-5,
        "solver_tol": 1e-5,
        "src_pt": np.array([0, 100, 0]),
        "listener_radius": 50,
        "listener_ds": 5,
        "recompute_matrices": False,
    }

    # weights
    for i, w in enumerate(VW_WEIGHTS):
        WEIGHTS[f"vw_{i}_wt"] = w

    tracker_dict = { "last_iter": -1 }
    for wt_lbl in WEIGHTS:
        l = wt_lbl[:-3]
        tracker_dict[f"{l}_v"] = np.zeros(ITERS)
        tracker_dict[f"{l}_g"] = np.zeros((ITERS, *hfield_torch.shape))

    # 1. define the small meshes (left to right, top to btm, ie -x to x, z to -z)
    sx = -tw / 2 + DIM_W / 2
    sz = -tb / 2 + DIM_B / 2
    small_meshes = []
    for iz in range(NUM_B):
        # keep track of which sides has borders
        for ix in range(NUM_W):
            borders = ""
            if ix == 0:
                borders += "l"
            if ix == NUM_W - 1:
                borders += "r"
            if iz == 0:
                borders += "u"
            if iz == NUM_B - 1:
                borders += "d"
            
            offx = sx + ix * DIM_W
            offz = sz + iz * DIM_B
            sP, sE, sdP, sdPi = build_small_mesh(DIM_W, DIM_B, DIM_H, ESIZE, NDIV, offx, offz)
            diffbem = ac3d.DiffBEM(**diffbem_cfg)
            diffbem.silent = True
            sP, sE = diffbem.precompute(sP, sE, sdPi)
            
            sx_max, _, sz_max = np.max(sP, axis=0)
            sx_min, _, sz_min = np.min(sP, axis=0)
            sdUV = dm.rect_points_to_uv(sdP, sx_min, sx_max, sz_min, sz_max)

            small_meshes.append((sP, sE, sdP, sdPi, sdUV, diffbem, borders, ix, iz))

    # 2. define large mesh
    large_Ps, large_Es = mesher.box_mesher(ESIZE, tw, tb, DIM_H)
    large_diff_pts_idx = np.where(np.isclose(large_Ps[:, 1], np.max(large_Ps[:, 1])))[0]
    large_top_Es_idx = np.where(np.all(np.isin(large_Es, large_diff_pts_idx), axis=1))[0]
    large_Ps, large_Es = mesher.face_subdivision(large_Ps, large_Es, large_top_Es_idx, NDIV)
    # the points we want to optimize are those at the top of the box
    large_diff_pts_idx = np.where(np.isclose(large_Ps[:, 1], np.max(large_Ps[:, 1])))[0]
    large_diff_Ps = large_Ps[large_diff_pts_idx]

    x_max, _, z_max = np.max(large_Ps, axis=0)
    x_min, _, z_min = np.min(large_Ps, axis=0)
    large_diff_uvs = dm.rect_points_to_uv(large_diff_Ps, x_min, x_max, z_min, z_max)

    # initialize diffmesh (for mitsuba renderer)
    semantic_loss = losses.ImgImgCLIPLoss()
    diffmesh = dm.ImageDiffMesh(large_Ps, large_Es, TGT_FNAME, semantic_loss)

    tgt = H.read_image(TGT_FNAME, NUM_W * RES_X, NUM_B * RES_Z, format="L")
    tgt_img = np.stack([tgt, tgt, tgt], axis=2)
    tgt_tensor = torch.from_numpy(tgt_img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # hfield = preprocess_tex(hfield_torch, BORDER).cpu().detach().numpy()
    # init_imgs = [diffmesh.render(hfield, e, a, radius=CAM_RAD, resx=RD_RES[0], resy=RD_RES[1]) for e, a in CAM_POS]
    # imgs = [diffmesh.check_ref(e, a, radius=CAM_RAD, resx=RD_RES[0], resy=RD_RES[1]) for e, a in CAM_POS]
    # H.save_images("t_ref.png", init_imgs + imgs, auto_grid=True)
    # exit()

    pbar = tqdm.trange(ITERS, dynamic_ncols=True)
    for it in pbar:
        opt.zero_grad()

        # for tracking purposes
        f = sampled_freqs[it]

        # pad the edges to force borders to be 0
        hfield_full = preprocess_tex(hfield_torch, BORDER)

        # init loss and grad
        custom_loss = 0
        custom_grads = torch.zeros_like(hfield_torch).to(DEVICE)

        # get the numpy version of the full hfield
        hfield = hfield_full.cpu().detach().numpy()

        # compute acoustic gradient (local)
        # create an empty "template" to hold all the gradients
        all_grad = np.zeros_like(hfield)
        if WEIGHTS["ac_wt"] != 0:
            for mesh_idx, (sp, se, sdp, sdpi, sduv, sdb, borders, six, siz) in enumerate(small_meshes):
                # set the frequency band
                sdb.set_band(sampled_freqs[it])

                # figure out which part of the large heightmap needs to be extracted
                shf = hfield[siz * RES_Z : (siz + 1) * RES_Z, six * RES_X : (six + 1) * RES_X]
                ac_v, ac_g = acoustic_gradient(shf, sdb, sduv)

                # the ac_v technically doesn't make any sense here
                tracker_dict["ac_v"][it] += ac_v / (NUM_W * NUM_B)
                all_grad[siz * RES_Z : (siz + 1) * RES_Z, six * RES_X : (six + 1) * RES_X] = ac_g.detach().cpu().numpy()
            # now we can cut away the acoustic gradient
            tracker_dict["ac_g"][it] = all_grad[BORDER:-BORDER, BORDER:-BORDER]

        # everything else is global

        # rendered view gradients
        for i, (el, az) in enumerate(CAM_POS):
            vw_v, vw_g = diffmesh.gradient(hfield, el, az, radius=CAM_RAD, resx=RD_RES[0], resy=RD_RES[1])
            tracker_dict[f"vw_{i}_v"][it] = vw_v
            # vw_g is the full gradient, but we only care about the middle section
            tracker_dict[f"vw_{i}_g"][it] = vw_g[BORDER:-BORDER, BORDER:-BORDER].detach().cpu().numpy()

        # call backward on indidivual losses for individual gradients
        hfield_torch.grad = None
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

        hfield_torch.grad = None
        ng_v = neg_relu(hfield_full)
        ng_v.backward(retain_graph=True)
        ng_g = hfield_torch.grad
        tracker_dict["ng_v"][it] = ng_v.item()
        tracker_dict["ng_g"][it] = ng_g.detach().cpu().numpy()

        hfield_torch.grad = None
        ba_v = barrier_loss(hfield_full, VMAX)
        ba_v.backward(retain_graph=True)
        ba_g = hfield_torch.grad
        tracker_dict["ba_v"][it] = ba_v.item()
        tracker_dict["ba_g"][it] = ba_g.detach().cpu().numpy()

        # weight gradients and sum
        for wt_lbl in WEIGHTS:
            l = wt_lbl[:-3]
            custom_loss += WEIGHTS[wt_lbl] * tracker_dict[f"{l}_v"][it]

            grad = torch.from_numpy(tracker_dict[f"{l}_g"][it]).to(DEVICE)
            grad = normalize_gradients(grad)
            custom_grads += WEIGHTS[wt_lbl] * grad

        hfield_torch.grad = custom_grads
        opt.step()

        # clamp the vmax
        with torch.no_grad():
            hfield_torch.clamp_(-VMAX, VMAX)

        pbar.set_postfix_str(f"Loss: {custom_loss:.6f} | Frequency: {f}")

        if ((1 + it) % 50) == 0:
            with torch.no_grad():
                hfield = preprocess_tex(hfield_torch, BORDER).cpu().detach().numpy()
                imgs = [diffmesh.render(hfield, *cam, radius=CAM_RAD, resx=RD_RES[0], resy=RD_RES[1]) for cam in CAM_POS]
                heights = eval_uv(hfield, large_diff_uvs)
                large_Ps[large_diff_pts_idx, 1] += heights
                np.save(f"outputs/large/{NAME}/checkpoints/hf_{it+1}.npy", hfield)
                H.save_images(f"outputs/large/{NAME}/checkpoints/rd_{it+1}.png", imgs + [hfield], auto_grid=True)
                H.save_mesh(f"outputs/large/{NAME}/checkpoints/ms_{it+1}.obj", large_Ps, large_Es)

        # save the tracker dict every iteration (overwrites itself)
        tracker_dict["last_iter"] = it
        with open(f"outputs/large/{NAME}/tracker_dict.pkl", "wb") as f:
            pickle.dump(tracker_dict, f)


    hfield = preprocess_tex(hfield_torch, BORDER).cpu().detach().numpy()
    imgs = [diffmesh.render(hfield, *cam, radius=CAM_RAD, resx=RD_RES[0], resy=RD_RES[1]) for cam in CAM_POS]
    heights = eval_uv(hfield, large_diff_uvs)
    large_Ps[large_diff_pts_idx, 1] += heights
    np.save(f"outputs/large/{NAME}/hfield.npy", hfield)
    H.save_images(f"outputs/large/{NAME}/renders.png", imgs + [hfield], auto_grid=True)
    H.save_mesh(f"outputs/large/{NAME}/mesh.obj", large_Ps, large_Es)

if __name__ == "__main__":
    main()