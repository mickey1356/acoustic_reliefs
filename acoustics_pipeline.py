from build import acoustics3d as ac3d
import pyoptim.helpers as H
import pyoptim.mesher as mesher
import numpy as np

ALL_FREQ_BANDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000]


def pipeline(mesh_fname, save_name=None, silent=False, src_pt=[0, 100, 0], lr=50):
    diffbem = ac3d.DiffBEM(64, 1.5, [-1], 3, 1e-5, 1e-5, 1e-5, np.array(src_pt), lr, 5, False)
    diffbem.silent = silent
    Ps, Es = H.read_mesh(mesh_fname)
    Ps, Es = diffbem.set_mesh(Ps, Es)
    values = []
    for freq_band in ALL_FREQ_BANDS:
        values.append(diffbem.band_value(freq_band))
    if save_name:
        save(save_name, values)
    return values

def rect_pipeline(dim, esize, save_name=None, silent=False, src_pt=[0, 100, 0], lr=50):
    Ps, Es = mesher.box_mesher(esize, w=dim, b=dim, h=0.15)
    diffbem = ac3d.DiffBEM(64, 1.5, [-1], 3, 1e-5, 1e-5, 1e-5, np.array(src_pt), lr, 5, False)
    diffbem.silent = silent
    Ps, Es = diffbem.set_mesh(Ps, Es)
    values = []
    for freq_band in ALL_FREQ_BANDS:
        values.append(diffbem.band_value(freq_band))
    if save_name:
        save(save_name, values)
    return values

def save(save_name, coeffs):
    with open("outputs/coeffs.csv", "a") as f:
        f.write(f"{save_name}," + ",".join(str(c) for c in coeffs) + "\n")

def pl(folders):
    import os
    for folder in sorted(folders):
        mesh_name = f"{folder}/mesh.obj"
        print(mesh_name)
        print()
        coeffs = pipeline(mesh_name)
        with open("outputs/coeffs.csv", "a") as f:
            f.write(os.path.basename(folder) + "," + ",".join(str(c) for c in coeffs) + "\n")


if __name__ == "__main__":
    import glob

    # pipeline("outputs/ac_0.04/mesh.obj", "ac_0.04")
    # pipeline("outputs/ac_0.04/mesh.obj", "ac_0.04_close", src_pt=[0, 10, 0], lr=5)
    # pipeline("outputs/ac_0.02/mesh.obj", "ac_0.02")
    # pipeline("outputs/ac_0.02/mesh.obj", "ac_0.02_close", src_pt=[0, 10, 0], lr=5)
    # pipeline("outputs/ac_0.01/mesh.obj", "ac_0.01")
    # pipeline("outputs/ac_0.01/mesh.obj", "ac_0.01_close", src_pt=[0, 10, 0], lr=5)

    pipeline("outputs/ac_0.02_near/mesh.obj", "ac_0.02_near_optim")
    pipeline("outputs/ac_0.02_near/mesh.obj", "ac_0.02_near_optim_close", src_pt=[0, 10, 0], lr=5)
    pipeline("outputs/ac_0.01_near/mesh.obj", "ac_0.01_near_optim")
    pipeline("outputs/ac_0.01_near/mesh.obj", "ac_0.01_near_optim_close", src_pt=[0, 10, 0], lr=5)
    # rect_pipeline(0.6, 0.02, "nrect_0.02_close", src_pt=[0, 10, 0], lr=5)

    # pipeline("outputs/ac_bernd_0.04_norm/mesh.obj", "bernd_0.04")
    # pipeline("outputs/ac_cat_0.04_norm/mesh.obj", "cat_0.04")
    # pipeline("outputs/ac_mountains_0.04_norm/mesh.obj", "mountains_0.04")
    # pipeline("outputs/ac_matterhorn_0.04_norm/mesh.obj", "matterhorn_0.04")
    # pipeline("outputs/ac_peppers_0.04_norm/mesh.obj", "peppers_0.04")
    # pipeline("outputs/ac_tux_0.04_norm/mesh.obj", "tux_0.04")

    # pipeline("outputs/ac_bernd_0.02_norm/mesh.obj", "bernd_0.02")
    # pipeline("outputs/ac_cat_0.02_norm/mesh.obj", "cat_0.02")
    # pipeline("outputs/ac_mountains_0.02_norm/mesh.obj", "mountains_0.02")
    # pipeline("outputs/ac_matterhorn_0.02_norm/mesh.obj", "matterhorn_0.02")
    # pipeline("outputs/ac_peppers_0.02_norm/mesh.obj", "peppers_0.02")
    # pipeline("outputs/ac_tux_0.02_norm/mesh.obj", "tux_0.02")

    # folders = glob.glob("outputs/acoustics/esize_0.02*")
    # pl(folders)

    # additional = ["outputs/acoustics/cat_esize_0.01_dim_0.8_band_1000_ac_5_cl_3_sm_10_mh_0.5"]
    # pl(additional)

    # for folder in sorted(folders):
    #     mesh_name = f"{folder}/mesh.obj"
    #     print(mesh_name)
    #     print()
    #     coeffs = pipeline(mesh_name)
    #     with open("outputs/coeffs.csv", "a") as f:
    #         f.write(folder + "," + ",".join(str(c) for c in coeffs) + "\n")


    # coeffs = pipeline("outputs/acoustics_n/test/mesh.obj")
    # with open("outputs/coeffs.csv", "a") as f:
        # f.write("test," + ",".join(str(c) for c in coeffs) + "\n")
