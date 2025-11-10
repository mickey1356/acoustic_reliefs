import pyoptim.helpers as H
import pyoptim.mesher as mesher
import numpy as np

import matplotlib.pyplot as plt

import acoustics3d as ac3d

def pressure(mesh_fname, frequency, listeners=None, silent=False, src_pt=[0, 100, 0]):
    diffbem = ac3d.DiffBEM(128, 1.5, [-1], 3, 1e-5, 1e-5, 1e-5, np.array(src_pt), 1, 5, False)
    diffbem.silent = silent
    Ps, Es = H.read_mesh(mesh_fname)
    Ps, Es = diffbem.set_mesh(Ps, Es)
    if listeners is None:
        return diffbem.get_listeners(), diffbem.pvals(frequency)
    else:
        return listeners, diffbem.pvals(frequency, listeners)

def pressure_fband(mesh_fname, band, n=3, octave=3, listeners=None, silent=False, src_pt=[0, 100, 0]):
    s = 0
    freqs = freqs_from_band(band, n, octave)
    for freq in freqs:
        l, p_c = pressure(mesh_fname, freq, listeners, silent, src_pt)
        s += np.abs(p_c)
    return l, s / len(freqs)

def freqs_from_band(band, n=3, octave=3):
    bw = 2 ** (1 / (2 * octave))
    low = np.log(band / bw)
    high = np.log(band * bw)
    return np.exp(np.linspace(low, high, num=n))

def main():

    for band in [500, 1000, 2000]:
        # band = 1000
        n = 3
        octave = 3

        for theta_s in range(5, 176, 5):
            r_s = 10
            # theta_s = 90
            theta_s_rad = theta_s / 180 * np.pi
            src_pt = [np.cos(theta_s_rad) * r_s, np.sin(theta_s_rad) * r_s, 0]

            num_listeners = 181
            thetas_l = np.linspace(0, np.pi, num=num_listeners)
            r_l = 5
            lst_pts = np.column_stack([np.cos(thetas_l) * r_l, np.sin(thetas_l) * r_l, np.zeros_like(thetas_l)])


            fig, ax = plt.subplots(dpi=300, subplot_kw={"projection": "polar"}, constrained_layout=True)
            ax.set_title(f"Avg. scattered pressure")

            _, pr_cat = pressure_fband("meshes/cat_0.6.obj", band=band, n=n, octave=octave, listeners=lst_pts, src_pt=src_pt)
            ax.plot(thetas_l, pr_cat, label="cat")

            _, pr_rect = pressure_fband("test-data/boxes/meshed_box_6464.obj", band=band, n=n, octave=octave, listeners=lst_pts, src_pt=src_pt)
            ax.plot(thetas_l, pr_rect, label="rect")

            _, pr_prd = pressure_fband("test-data/boxes/meshed_prd_l.obj", band=band, n=n, octave=octave, listeners=lst_pts, src_pt=src_pt)
            ax.plot(thetas_l, pr_prd, label="prd")

            ax.axvline(theta_s_rad, color="red", linestyle="dashed", label="src")
            ax.set_yticklabels([])
            
            fig.legend(loc="lower right")
            np.save(f"outputs/polar_plots/{band}/{theta_s}.npy", np.column_stack([pr_cat, pr_rect, pr_prd]))
            fig.savefig(f"outputs/polar_plots/{band}/{theta_s}.png")
            plt.close()


if __name__ == "__main__":
    main()
