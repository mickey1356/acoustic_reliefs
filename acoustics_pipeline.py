from build import bemlib3d as bl3d
import pyoptim.helpers as H
import pyoptim.mesher as mesher

ALL_FREQ_BANDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000]


def pipeline(mesh_fname, silent=False):
    diffbem = bl3d.DiffBEM(64, 1.5, [-1], 3, 1e-3, 1e-3, 1e-3, 5, False)
    diffbem.silent = silent
    Ps, Es = H.read_mesh(mesh_fname)
    Ps, Es = diffbem.set_mesh(Ps, Es)
    values = []
    for freq_band in ALL_FREQ_BANDS:
        values.append(diffbem.band_value(freq_band))
    return values


def rect_pipeline(dim, esize, silent=False):
    Ps, Es = mesher.box_mesher(esize, w=dim, b=dim, h=0.15)
    diffbem = bl3d.DiffBEM(64, 1.5, [-1], 3, 1e-3, 1e-3, 1e-3, 5, False)
    diffbem.silent = silent
    Ps, Es = diffbem.set_mesh(Ps, Es)
    values = []
    for freq_band in ALL_FREQ_BANDS:
        values.append(diffbem.band_value(freq_band))
    return values

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

    for d in [0.6]:
        coeffs = rect_pipeline(d, 0.02)
        with open("outputs/coeffs.csv", "a") as f:
            f.write(f"rect_{d}," + ",".join(str(c) for c in coeffs) + "\n")

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
