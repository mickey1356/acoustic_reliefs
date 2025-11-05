import numpy as np
import trimesh
import matplotlib
import matplotlib.pyplot as plt

def read_mesh(fname):
    mesh = trimesh.load_mesh(fname)
    Ps = mesh.vertices
    Es = mesh.faces
    return Ps, Es

if __name__ == "__main__":

    freq = 1000 # 100, 400, 1000

    V, F = read_mesh("arsound/final.obj")
    print(V.shape, F.shape)
    vals = np.load(f"arsound/desk_vals_{freq}Hz.npy")
    print(vals.shape)

    # get the pressure magnitude
    mags = np.abs(vals)
    
    # plot a rough histogram of the values
    plt.hist(mags, bins=100)
    plt.savefig(f"arsound/{freq}Hz_hist.png")

    # clip the magnitudes to the 95th percentile
    clip_val = np.percentile(mags, 95)
    mags = np.clip(mags, 0, clip_val)

    # normalize so they range from 0 to 1
    mags = (mags - mags.min()) / (mags.max() - mags.min())

    # get a colormap
    cmap = matplotlib.colormaps["RdBu_r"]
    colors = cmap(mags)

    V_new = V[F.reshape(-1), :]
    F_new = np.arange(V_new.shape[0]).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=V_new, faces=F_new, smooth=False, process=False)
    mesh.visual.face_colors = (colors[:, :3] * 255).astype(np.uint8)
    print(mesh.visual.face_colors)
    mesh.export(f"arsound/{freq}Hz_colored.ply")