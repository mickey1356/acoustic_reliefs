import numpy as np
import os

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
from mitsuba import ScalarTransform4f as T
import drjit as dr

import torch

# mitsuba's eval_1 assumes 0,0 as top-left
def rect_points_to_uv(points, x_min=-0.3, x_max=0.3, z_min=-0.3, z_max=0.3, uvmin=0.0001, uvmax=0.9999):
    return np.column_stack([(points[:, 0] - x_min) / (x_max - x_min) * (uvmax - uvmin) + uvmin, (points[:, 2] - z_min) / (z_max - z_min) * (uvmax - uvmin) + uvmin])


BASE_SCENE = {
    "type": "scene",
    "integrator": {
        "type": "prb_projective",
        "max_depth": 5,
        "sppi": 0,
        "hide_emitters": True,
    },
    "emitter": {
        # "type": "envmap",
        # "filename": "test-data/envmaps/evening_sun.hdr",
        # "scale": 1,
        # "filename": "test-data/envmaps/mitsuba_interior.exr",
        # "scale": 1,
        # "filename": "test-data/envmaps/interior.exr",
        # "scale": 0.2,
        # "type": "directional",
        # "direction": [0, -1, 0],
        # "irradiance": {
        #     "type": "rgb",
        #     "value": 2,
        # }
        "type": "point",
        "position": [0, 0.5, -0.866],
        "intensity": {
            "type": "rgb",
            "value": 2.5,
        },
    },
    "emitter2": {
        "type": "constant",
        "radiance": {
            "type": "rgb",
            "value": 0.2,
        }
    }
}

class DiffMesh:
    def __init__(self, box_Ps, box_Es, config, device="cuda"):
        # first we want to get the top-facing face of the box
        # we assume +x is right, +y is up, and -z is forward
        # get dimensions of box
        x_max, y_max, z_max = np.max(box_Ps, axis=0)
        x_min, y_min, z_min = np.min(box_Ps, axis=0)
        self.w, h, self.b = x_max - x_min, y_max - y_min, z_max - z_min
        # get the rows of box_Es where all(box_Ps[ei, 1] == h)
        tmp_Es = box_Es[np.all(np.isclose(box_Ps[box_Es, 1], h), axis=1)]
        # technically, box_Ps, top_Es gives the plane mesh, but it'd probably be good to remove the unnecessary vertices
        top_Ps = []
        top_Es = []
        verts_seen = {}
        for face in tmp_Es:
            e = []
            for pi in face:
                if pi not in verts_seen:
                    verts_seen[pi] = len(top_Ps)
                    top_Ps.append(box_Ps[pi])
                e.append(verts_seen[pi])
            top_Es.append(e)
        top_Ps = np.array(top_Ps)
        top_Es = np.array(top_Es, dtype=int)
        # set the y value to 0
        top_Ps[:, 1] = 0
        # compute UVs
        top_UVs = rect_points_to_uv(top_Ps, x_min, x_max, z_min, z_max)

        # create mitsuba mesh
        mesh = mi.Mesh("hfield", vertex_count=len(top_Ps), face_count=len(top_Es), has_vertex_normals=True, has_vertex_texcoords=True)
        mesh_params = mi.traverse(mesh)
        mesh_params["vertex_positions"] = mi.Float(top_Ps.ravel())
        mesh_params["faces"] = mi.Int(top_Es.ravel())
        mesh_params["vertex_texcoords"] = mi.Float(top_UVs.ravel())
        mesh_params.update()

        # save the mesh
        mesh.write_ply(os.path.join(config["out_folder"], config["name"], "base_mesh.ply"))



