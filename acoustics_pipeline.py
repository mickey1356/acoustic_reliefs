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

    # V, F = H.read_obj("tmp/qrd_notri.obj")
    # mesher_pipeline(V, F, save_name="qrd_x0_nn_17", src_pt=[0, 3, 0], lr=1.71)

    # rect_pipeline(0.6, 0.6, 0.02, save_name="rect_z0_n", src_pt=[0, 3, 0], lr=1.7)
    # pipeline("tmp/meshed_prd_s.obj", save_name="prd_z0_n", src_pt=[0, 3, 0], lr=1.7)

    # V, F = H.read_obj("tmp/cylinders.obj")
    # mesher_pipeline(V, F, save_name="cylinders_x0_nn_17", src_pt=[0, 3, 0], lr=1.71)

    # V, F = H.read_obj("tmp/hemispheres.obj")
    # mesher_pipeline(V, F, save_name="hemispheres_x0_nn_17", src_pt=[0, 3, 0], lr=1.71)

    # V, F = H.read_obj("tmp/noise.obj")
    # mesher_pipeline(V, F, save_name="noise_x0_nn_17", src_pt=[0, 3, 0], lr=1.71)

    # V, F = H.read_obj("tmp/measurement.obj")
    # mesher_pipeline(V, F, e_size=0.04, save_name="measurement_z0_nn_17", src_pt=[0, 3, 0], lr=1.71)


    # V, F = H.read_obj("tmp/relief.obj")
    # mesher_pipeline(V, F, e_size=0.04, save_name="relief_z0_nn_17", src_pt=[0, 3, 0], lr=1.71)

    # V, F = H.read_obj("tmp/measurement3.obj")
    # mesher_pipeline(V, F, e_size=0.04, save_name="measurement3_z0_nn_17", src_pt=[0, 3, 0], lr=1.71)

    # pipeline("outputs/hres/ptlight_waveleft/ptlight_waveleft.obj", save_name="waveleft_z0_n", src_pt=[0, 3, 0], lr=1.7)

    # pipeline("tmp/raf1_1.obj", "raf1_1", silent=True)
    # pipeline("tmp/raf1_2.obj", "raf1_2", silent=True)
    # pipeline("tmp/raf1_3.obj", "raf1_3", silent=True)
    # pipeline("tmp/raf2_1.obj", "raf2_1", silent=True)
    # pipeline("tmp/raf2_2.obj", "raf2_2", silent=True)
    # pipeline("tmp/raf2_3.obj", "raf2_3", silent=True)

    # angles = [(0, 0), (30, 0), (30, 60), (30, 120), (30, 180), (30, 240), (30, 300), (60, 0), (60, 60), (60, 120), (60, 180), (60, 240), (60, 300)]
    # angles = [(30, 0), (30, 60), (30, 120), (30, 180), (30, 240), (30, 300), (60, 60), (60, 180), (60, 300)]
    angles = [(0, 0), (15, 0), (30, 0), (45, 0), (60, 0), (75, 0)]
    for el, az in angles:
        src_pt = [100 * np.sin(np.radians(el)) * np.cos(np.radians(az)), 100 * np.cos(np.radians(el)), -100 * np.sin(np.radians(el)) * np.sin(np.radians(az))]
        rect_pipeline(0.6, 0.6, 0.02, save_name=f"rect_normal_{el}_{az}", silent=True, ht=0.15, src_pt=src_pt, lr=50)
        # pipeline("outputs/hres/ptlight_waves_abstract/ptlight_waves_abstract.obj", save_name=f"normal_{el}_{az}", silent=True, src_pt=src_pt)

    # pipeline("tmp/flowers.obj", save_name="flower_relief", silent=True)

    # positions = []
    # for i in range(20):
    #     pt = np.random.normal(0, 1, 3)
    #     pt = pt / np.linalg.norm(pt) * 100
    #     # limit to upper hemisphere (flip y)
    #     pt[1] = abs(pt[1])
    #     positions.append(pt)
    #     pipeline("outputs/hres/ptlight_waves_abstract/ptlight_waves_abstract.obj", save_name=f"r{i}", silent=True, src_pt=pt)
    # np.savetxt("rand_pos.txt", positions)

    # pipeline("outputs/hres/ptlight_waves_abstract/ptlight_waves_abstract.obj", save_name="normal_lr", silent=True, src_pt=[0, 100, 0], lr=50)
    # pipeline("outputs/hres/ptlight_waves_abstract/ptlight_waves_abstract.obj", save_name="normal_lr2", silent=True, src_pt=[0, 100, 0], lr=100)