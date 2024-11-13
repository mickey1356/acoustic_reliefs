import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == "__main__":
    folder = "e0.04_res32"
    # folder = "e0.02_res64"
    ad_est = np.load(f"outputs/gradients/{folder}/adjoint.npy")
    ad_act = np.load(f"outputs/gradients/{folder}/adjoint_actual.npy")
    fd = np.load(f"outputs/gradients/{folder}/fd_1e-6.npy")
    # fds = [np.load(f"outputs/gradients/e0.04_res32/fd_1e-{ii}.npy") for ii in range(2, 17)]

    vmax = np.max(ad_act)
    vmin = np.min(ad_act)

    # plt.figure(figsize=(20, 20), dpi=600)
    fig, axs = plt.subplots(nrows=4, ncols=3, dpi=300, layout='constrained')
    axs = axs.flatten()

    im = axs[0].imshow(ad_act, cmap="RdBu_r", vmax=vmax, vmin=vmin, interpolation="None")
    axs[0].axis("off")
    axs[0].set_title("adjoint (actual)", {'fontsize': 8})
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    bar = fig.colorbar(im, cax=cax, orientation='vertical')
    bar.ax.tick_params(labelsize=5)

    im = axs[1].imshow(ad_est, cmap="RdBu_r", vmax=vmax, vmin=vmin, interpolation="None")
    axs[1].axis("off")
    axs[1].set_title("adjoint (est.)", {'fontsize': 8})
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    bar = fig.colorbar(im, cax=cax, orientation='vertical')
    bar.ax.tick_params(labelsize=5)

    im = axs[2].imshow(fd, cmap="RdBu_r", vmax=vmax, vmin=vmin, interpolation="None")
    axs[2].axis("off")
    axs[2].set_title("finite diffs", {'fontsize': 8})
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    bar = fig.colorbar(im, cax=cax, orientation='vertical')
    bar.ax.tick_params(labelsize=5)

    axs[3].remove()

    sqr_err = (ad_est - ad_act) ** 2
    im = axs[4].imshow(sqr_err, vmin=0, interpolation="None")
    axs[4].axis("off")
    axs[4].set_title("sqr err (ad_act - ad_est)", {'fontsize': 8})
    divider = make_axes_locatable(axs[4])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    bar = fig.colorbar(im, cax=cax, orientation='vertical', format="%1.1e")
    bar.ax.tick_params(labelsize=5)

    sqr_err = (fd - ad_act) ** 2
    im = axs[5].imshow(sqr_err, vmin=0, interpolation="None")
    axs[5].axis("off")
    axs[5].set_title("sqr err (ad_act - fd)", {'fontsize': 8})
    divider = make_axes_locatable(axs[5])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    bar = fig.colorbar(im, cax=cax, orientation='vertical', format="%1.1e")
    bar.ax.tick_params(labelsize=5)

    axs[6].remove()

    rel_err = np.abs(ad_est - ad_act) / ad_act
    rel_err = np.nan_to_num(rel_err, nan=0)
    im = axs[7].imshow(rel_err, vmin=0, vmax=1, interpolation="None")
    axs[7].axis("off")
    axs[7].set_title("rel err (ad_act - ad_est)", {'fontsize': 8})
    divider = make_axes_locatable(axs[7])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    bar = fig.colorbar(im, cax=cax, orientation='vertical')
    bar.ax.tick_params(labelsize=5)

    rel_err = np.abs(fd - ad_act) / ad_act
    rel_err = np.nan_to_num(rel_err, nan=0)
    im = axs[8].imshow(rel_err, vmin=0, vmax=1, interpolation="None")
    axs[8].axis("off")
    axs[8].set_title("rel err (ad_act - fd)", {'fontsize': 8})
    divider = make_axes_locatable(axs[8])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    bar = fig.colorbar(im, cax=cax, orientation='vertical')
    bar.ax.tick_params(labelsize=5)

    axs[9].remove()

    signs = (ad_est * ad_act) >= 0
    im = axs[10].imshow(signs, vmin=0, vmax=1, interpolation="None", cmap="gray")
    axs[10].set_xticks([])
    axs[10].set_yticks([])
    axs[10].set_xticks([], minor=True)
    axs[10].set_yticks([], minor=True)
    axs[10].set_title("signs differ (ad_act - ad_est)", {'fontsize': 8})
    divider = make_axes_locatable(axs[10])

    signs = (fd * ad_act) >= 0
    im = axs[11].imshow(signs, vmin=0, vmax=1, interpolation="None", cmap="gray")
    axs[11].set_xticks([])
    axs[11].set_yticks([])
    axs[11].set_xticks([], minor=True)
    axs[11].set_yticks([], minor=True)
    axs[11].set_title("signs differ (ad_act - fd)", {'fontsize': 8})
    divider = make_axes_locatable(axs[11])


    fig.savefig("t_grad.png")
