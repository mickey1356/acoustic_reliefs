import numpy as np

import build.acoustics3d as ac3d

from pyoptim import helpers as H

if __name__ == "__main__":
    # V, F = H.read_mesh("arsound/desk.obj")
    # V = V[:, :3] # discard colors
    # print(V.shape, F.shape)

    # # center the mesh
    # V -= V.mean(axis=0)
    # # reorient the axes (currently x points towards the camera, and y points SW, z points SE)
    # # we want x to point E, y to point towards the camera, and z to point S
    # # create the new basis vectors
    # new_x = np.array([0, -1, 1]) / np.sqrt(2)  # new x
    # new_y = np.array([1, 0, 0])  # new y
    # new_z = np.array([0, 1, 1]) / np.sqrt(2) # new z
    # R = np.vstack([new_x, new_y, new_z]).T  # rotation
    # V = V @ R  # rotate

    # H.save_mesh("arsound/desk_fix.obj", V, F)


    V, F = H.read_mesh("arsound/desk_fix.obj")
    print(V.shape, F.shape)

    diffbem = ac3d.DiffBEM(128, 1.5, [-1], 3, 1e-5, 1e-5, 1e-5, np.array([0, 15, 0]), 50, 5, False)
    # diffbem.silent = True

    for freq in [100, 400, 1000, 2500]:
        Ps, Es = diffbem.set_mesh(V, F)
        vals = diffbem.surface_vals(freq)
        np.save(f"arsound/desk_vals_{freq}Hz.npy", vals)