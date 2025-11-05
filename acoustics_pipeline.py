from build import acoustics3d as ac3d
import pyoptim.helpers as H
import pyoptim.mesher as mesher
import numpy as np

ALL_FREQ_BANDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000]
ONE_SIXTH_FREQ_BANDS = [106, 119, 133, 150, 168, 188, 211, 237, 266, 299, 335, 376, 422, 473, 530, 600, 670, 750, 840, 940, 1060, 1190, 1330, 1500, 1680, 1880, 2110, 2370, 2660, 2990, 3350, 3760, 4220]

def mesher_pipeline(V, F, e_size=0.02, save_name=None, silent=False, src_pt=[0, 100, 0], lr=50):
    diffbem = ac3d.DiffBEM(128, 1.5, [-1], 3, 1e-5, 1e-5, 1e-5, np.array(src_pt), lr, 5, False)
    diffbem.silent = silent
    Ps, Es = mesher.mesh_surface(V, F, e_size)
    Ps, Es = diffbem.set_mesh(Ps, Es)
    values = []
    for freq_band in ALL_FREQ_BANDS:
        values.append(diffbem.band_value(freq_band))
        print(f"freq {freq_band} Hz done")
    if save_name:
        save(save_name, values)
    return values

def pipeline(mesh_fname, save_name=None, silent=False, src_pt=[0, 100, 0], lr=50):
    diffbem = ac3d.DiffBEM(128, 1.5, [-1], 3, 1e-5, 1e-5, 1e-5, np.array(src_pt), lr, 5, False)
    diffbem.silent = silent
    Ps, Es = H.read_mesh(mesh_fname)
    Ps, Es = diffbem.set_mesh(Ps, Es)
    values = []
    for freq_band in ALL_FREQ_BANDS:
        values.append(diffbem.band_value(freq_band))
        print(f"freq {freq_band} Hz done")
    if save_name:
        save(save_name, values)
    return values

def rect_pipeline(dimw, dimb, esize, save_name=None, silent=False, ht=0.15, src_pt=[0, 100, 0], lr=50):
    Ps, Es = mesher.box_mesher(esize, w=dimw, b=dimb, h=ht)
    diffbem = ac3d.DiffBEM(128, 1.5, [-1], 3, 1e-5, 1e-5, 1e-5, np.array(src_pt), lr, 5, False)
    diffbem.silent = silent
    Ps, Es = diffbem.set_mesh(Ps, Es)
    values = []
    for freq_band in ALL_FREQ_BANDS:
        values.append(diffbem.band_value(freq_band))
        print(f"freq {freq_band} Hz done")
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

    # pipeline("<PATH_TO_MESH>", "<OUTPUT_NAME>")
    