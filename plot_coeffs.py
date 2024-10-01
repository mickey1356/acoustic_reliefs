import matplotlib.pyplot as plt
import numpy as np

freqs = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000]

# to_plot = ["box_0.6", "esize_0.02_band_500", "esize_0.02_band_1000", "esize_0.02_band_1600", "esize_0.02_band_2000", "esize_0.02_band_3150"]
# to_plot = ["box_0.6"] + [f"esize_0.02_int{i}" for i in range(15)]
# to_plot = ["rect_1.2", "cat_esize_0.02_dim_1.2_band_1000_bad", "cat_esize_0.02_dim_1.2_band_2000_bad", "cat_esize_0.02_dim_1.2_band_3150_bad"]
to_plot = ["rect_0.6", "rect_0.6_0"]

lbls = {tp: tp for tp in to_plot}

dat = []
with open('outputs/coeffs.csv') as f:
    dat = f.readlines()
dat = [line.split(",") for line in dat]
coef_dat = {line[0]: np.array(list(map(float, line[1:]))) for line in dat}

plt.ioff()
plt.figure(dpi=300)
for lbl in to_plot:
    avg = np.mean(coef_dat[lbl])
    plt.plot(freqs, coef_dat[lbl], label=f"{lbls[lbl]} - {avg:.3f}")
plt.legend(prop={"size": 6})
plt.grid(True, which="both", ls=":")
plt.savefig(f"outputs/plots/out.png")
# plt.show()
