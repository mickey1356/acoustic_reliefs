import numpy as np
from PIL import Image
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def write_mesh(fname, verts, faces, face_type="triangle"):
    mesh = meshio.Mesh(
        points = verts,
        cells = [(face_type, faces)]
    )
    mesh.write(fname)


def read_mesh(fname, face_type="triangle"):
    mesh = meshio.read(fname)
    Ps = mesh.points
    Es = mesh.cells_dict[face_type]
    return Ps, Es

def read_image(fname, w=None, h=None, format="L", resample=Image.Resampling.BILINEAR):
    with Image.open(fname) as pimg:
        # img = np.asarray(pimg.convert("RGB")) / 255
        if w and h:
            img = np.asarray(pimg.resize((w, h), resample=Image.Resampling.BILINEAR).convert(format)) / 255
        else:
            img = np.asarray(pimg.convert(format)) / 255
    return img

def save_images(fname, imgs, dpi=200, fig_axis=0, show_ax="off"):
    fig, axs = plt.subplots(len(imgs), figsize=(3 + fig_axis, (3 + len(imgs)) if len(imgs) > 1 else 3), dpi=dpi, constrained_layout=True)
    plt.rc('font', size=6)
    if len(imgs) == 1:
        axs = [axs]
    for a, i in zip(axs, imgs):
        if i.ndim == 2 or i.shape[2] == 1:
            im = a.imshow(i, cmap="gray")
            divider = make_axes_locatable(a)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax)
        else:
            a.imshow(i)
        a.axis(show_ax)
    fig.savefig(fname)

