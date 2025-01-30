
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
DEVICE = "cuda"
SEED = 42

# for the large optim, we tile multiple 0.6 x 0.6 diffusers each with 128x128 hfields
NUM_W = 3
NUM_B = 2
RES_X = 128
RES_Z = 128
DIM_W = 0.6
DIM_B = 0.6
DIM_H = 0.15

ESIZE = 0.02
NDIV = 1
ADD_DIV = 1

ITERS = 50
VMAX = 0.07
BORDER = 2

CAM_RAD = 2.5
TGT_FNAME = "test-data/images/landscape2.jpg"
RD_RES = (3 * 256, 2 * 256)

NAME = "landscape2"
INIT_HFIELD = "outputs/large/landscape2/hfield.npy"

WEIGHTS = {
    "cl_wt": 5,
    "sm_wt": 15,
    "ba_wt": 1,
    "ng_wt": 2,
    "rg_wt": 5,
}
VW_WEIGHTS = [7, 2, 2, 2, 2]


def main():
    # get the large mesh
    Ps, Es = mesher.box_mesher(ESIZE, NUM_W * DIM_W, NUM_B * DIM_B, DIM_H)
    diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]
    diff_Ps = Ps[diff_pts_idx]

    # subdivide top faces if needed
    subdiv_top = NDIV + ADD_DIV
    if subdiv_top > 0:
        top_Es_idx = np.where(np.all(np.isin(Es, diff_pts_idx), axis=1))[0]
        Ps, Es = mesher.face_subdivision(Ps, Es, top_Es_idx, subdiv_top)
        # the points we want to optimize are those at the top of the box
        diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]
        diff_Ps = Ps[diff_pts_idx]

    x_max, y_max, z_max = np.max(Ps, axis=0)
    x_min, y_min, z_min = np.min(Ps, axis=0)
    diff_uvs = dm.rect_points_to_uv(diff_Ps, x_min, x_max, z_min, z_max)

    # init hfield
    hfield = np.load(INIT_HFIELD)

    # remove the border
    hfield = hfield[BORDER:-BORDER, BORDER:-BORDER]
    hf_rh, hf_rw = hfield.shape

    # use a higher resolution heightfield
    mult = (2 ** ADD_DIV)

    # upsample the original heightfield
    n_hfield = cv2.resize(hfield, (mult * hf_rw, mult * hf_rh))

    # initialize diffmesh (for mitsuba renderer)
    semantic_loss = losses.ImgImgCLIPLoss()
    diffmesh = dm.ImageDiffMesh(Ps, Es, TGT_FNAME, semantic_loss)

    # for texture clip loss
    tgt = H.read_image(TGT_FNAME, mult * hf_rw + 2 * BORDER, mult * hf_rh + 2 * BORDER, format="L")
    tgt_img = np.stack([tgt, tgt, tgt], axis=2)
    tgt_tensor = torch.from_numpy(tgt_img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # set up torch optim
    hfield_torch = torch.from_numpy(n_hfield).to(DEVICE)
    init_nhfield = preprocess_tex(hfield_torch, BORDER)

    # set up optimizer
    hfield_torch.requires_grad = True
    opt = torch.optim.Adam([hfield_torch], lr=1e-3)

    # weights
    for i, w in enumerate(VW_WEIGHTS):
        WEIGHTS[f"vw_{i}_wt"] = w

    tracker_dict = { "last_iter": -1 }
    for wt_lbl in WEIGHTS:
        l = wt_lbl[:-3]
        tracker_dict[f"{l}_v"] = 0
        tracker_dict[f"{l}_g"] = np.zeros(shape=hfield_torch.shape)


    pbar = tqdm.trange(ITERS, dynamic_ncols=True)
    for it in pbar:
        opt.zero_grad()

        # pad the edges to force borders to be 0
        hfield_full = preprocess_tex(hfield_torch, BORDER)

        custom_loss = 0
        custom_grads = torch.zeros_like(hfield_torch).to(DEVICE)

        # rendered view gradients
        hfield = hfield_full.cpu().detach().numpy()
        for i, (el, az) in enumerate(CAM_POS):
            vw_v, vw_g = diffmesh.gradient(hfield, el, az, radius=CAM_RAD, resx=RD_RES[0], resy=RD_RES[1])
            # vw_g is the full gradient, but we only care about the middle section
            tracker_dict[f"vw_{i}_v"] = vw_v
            tracker_dict[f"vw_{i}_g"] = vw_g[BORDER:-BORDER, BORDER:-BORDER].detach().cpu().numpy()

        hfield_torch.grad = None
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

        hfield_torch.grad = None
        ba_v = barrier_loss(hfield_full, VMAX)
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

        for wt_lbl in WEIGHTS:
            l = wt_lbl[:-3]
            custom_loss += WEIGHTS[wt_lbl] * tracker_dict[f"{l}_v"]

            grad = torch.from_numpy(tracker_dict[f"{l}_g"]).to(DEVICE)
            grad = normalize_gradients(grad)
            custom_grads += WEIGHTS[wt_lbl] * grad

        hfield_torch.grad = custom_grads
        opt.step()

        with torch.no_grad():
            hfield_torch.clamp_(-VMAX, VMAX)

    with torch.no_grad():
        hfield_torch = preprocess_tex(hfield_torch, BORDER)
        hfield = hfield_torch.cpu().detach().numpy()
        rendered_imgs = [diffmesh.render(hfield, *cam, radius=CAM_RAD, resx=RD_RES[0], resy=RD_RES[1]) for cam in CAM_POS]
        heights = eval_uv(hfield, diff_uvs)
        Ps[diff_pts_idx, 1] += heights
        np.save(f"outputs/large/{NAME}/hfield_hres.npy", hfield)
        H.save_images(f"outputs/large/{NAME}/hres_renders.png", rendered_imgs + [hfield], auto_grid=True)
        H.save_mesh(f"outputs/large/{NAME}/hres_mesh.obj", Ps, Es)


if __name__ == "__main__":
    main()