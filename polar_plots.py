import numpy as np
import matplotlib.pyplot as plt
import tqdm

for i, deg in enumerate(tqdm.trange(5, 180, 5)):
    p = np.load(f"outputs/polar_plots/500/{deg}.npy")
    p_cat = p[:, 0]
    p_rect = p[:, 1]
    p_prd = p[:, 2]

    fig, ax = plt.subplots(dpi=200, subplot_kw={"projection": "polar"}, constrained_layout=True)
    ax.set_title("Average relative scattered pressure (500 Hz)")
    thetas_l = np.linspace(0, np.pi, num=181)

    ax.plot(thetas_l, p_cat / np.max(p_rect), label="Cat (0.6 m)")
    ax.plot(thetas_l, p_rect / np.max(p_rect), label="Cuboid")
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.axvline(deg / 180 * np.pi, color="red", linestyle="dashed", label="Source")
    ax.set_ylim([0,1.2])
    ax.legend()
    
    plt.savefig(f"outputs/polar_plots/gifs/{i}.jpg")
    plt.close()
