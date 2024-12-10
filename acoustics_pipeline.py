from build import acoustics3d as ac3d
import pyoptim.helpers as H
import pyoptim.mesher as mesher
import numpy as np

ALL_FREQ_BANDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000]
ONE_SIXTH_FREQ_BANDS = [106, 119, 133, 150, 168, 188, 211, 237, 266, 299, 335, 376, 422, 473, 530, 600, 670, 750, 840, 940, 1060, 1190, 1330, 1500, 1680, 1880, 2110, 2370, 2660, 2990, 3350, 3760, 4220]

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

def rect_pipeline(dim, esize, save_name=None, silent=False, ht=0.15, src_pt=[0, 100, 0], lr=50):
    Ps, Es = mesher.box_mesher(esize, w=dim, b=dim, h=ht)
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

    # pipeline("outputs/ac_cat_0.02_norm_multifreq/mesh.obj", "cat_0.02_multifreq")
    # pipeline("outputs/ac_cat_0.02_norm_multifreq/mesh.obj", "cat_0.02_multifreq_close", src_pt=[0, 10, 0], lr=5)
    # pipeline("outputs/ac_matterhorn_0.02_norm_multifreq/mesh.obj", "matterhorn_0.02_multifreq")
    # pipeline("outputs/ac_matterhorn_0.02_norm_multifreq/mesh.obj", "matterhorn_0.02_multifreq_close", src_pt=[0, 10, 0], lr=5)

    # pipeline("outputs/ac_cat_0.02_norm_multifreq2/mesh.obj", "cat_0.02_multifreq2")
    # pipeline("outputs/ac_cat_0.02_norm_multifreq2/mesh.obj", "cat_0.02_multifreq2_close", src_pt=[0, 10, 0], lr=5)
    # pipeline("outputs/ac_matterhorn_0.02_norm_multifreq2/mesh.obj", "matterhorn_0.02_multifreq2")
    # pipeline("outputs/ac_matterhorn_0.02_norm_multifreq2/mesh.obj", "matterhorn_0.02_multifreq2_close", src_pt=[0, 10, 0], lr=5)

    # pipeline("outputs/ac_cat_0.02_norm_multifreq3/mesh.obj", "cat_0.02_multifreq3")
    # pipeline("outputs/ac_cat_0.02_norm_multifreq3/mesh.obj", "cat_0.02_multifreq3_close", src_pt=[0, 10, 0], lr=5)
    # pipeline("outputs/ac_matterhorn_0.02_norm_multifreq3/mesh.obj", "matterhorn_0.02_multifreq3")
    # pipeline("outputs/ac_matterhorn_0.02_norm_multifreq3/mesh.obj", "matterhorn_0.02_multifreq3_close", src_pt=[0, 10, 0], lr=5)

    # pipeline("outputs/ac_0.02_0.6_multfreq_sample3/mesh.obj", "ac_0.6_multifreq_sample3")
    # pipeline("outputs/ac_cat_0.6_multfreq_sample3/mesh.obj", "ac_cat_0.6_multifreq_sample3")
    # pipeline("outputs/ac_bunny_0.6_multifreq_sample/mesh.obj", "ac_bunny_0.6_multifreq_sample")
    # pipeline("outputs/ac_matterhorn_0.6_multfreq_sample2/mesh.obj", "ac_matterhorn_0.6_multifreq_sample2")
    # pipeline("outputs/ac_0.02_1500/mesh.obj", "ac_0.02_1500")
    # pipeline("outputs/ac_0.02_1550/mesh.obj", "ac_0.02_1550")
    # pipeline("outputs/ac_0.02_1650/mesh.obj", "ac_0.02_1650")
    # pipeline("outputs/ac_0.02_1700/mesh.obj", "ac_0.02_1700")
    # pipeline("outputs/ac_cat_0.9_multifreq_sample/mesh.obj", "ac_cat_0.9_multifreq_sample")

    pipeline("outputs/ac_mountains_0.6_multifreq_sample/mesh.obj", "ac_mountains_0.6_multifreq_sample")
    pipeline("outputs/ac_bunny_0.9_multifreq_sample/mesh.obj", "ac_bunny_0.9_multifreq_sample")
    
    # pipeline("outputs/ac_cat_0.02_norm_0.9/mesh.obj", "cat_0.02_0.9")
    # pipeline("outputs/ac_cat_0.02_norm_0.9/mesh.obj", "cat_0.02_0.9_close", src_pt=[0, 10, 0], lr=5)
    # pipeline("test-data/boxes/meshed_prd_s.obj", "prd_s_close", src_pt=[0, 10, 0], lr=5)
    # pipeline("test-data/boxes/meshed_prd_l.obj", "prd_l_close", src_pt=[0, 10, 0], lr=5)

    # pipeline("outputs/ac_bernd_0.04_norm/mesh.obj", "bernd_0.04")
    # pipeline("outputs/ac_cat_0.04_norm/mesh.obj", "cat_0.04")
    # pipeline("outputs/ac_mountains_0.04_norm/mesh.obj", "mountains_0.04")
    # pipeline("outputs/ac_matterhorn_0.04_norm/mesh.obj", "matterhorn_0.04")
    # pipeline("outputs/ac_peppers_0.04_norm/mesh.obj", "peppers_0.04")
    # pipeline("outputs/ac_tux_0.04_norm/mesh.obj", "tux_0.04")
