import numpy as np

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
from mitsuba import ScalarTransform4f as T
import drjit as dr

from pyoptim import mesher
from pyoptim import helpers as H
from pyoptim import diffmesh as dm

def eval_uv(tex, uvs):
    h, w = tex.shape
    _uvs = uvs.copy()
    uf, ui = np.modf(_uvs[:, 0] * w - 0.5)
    vf, vi = np.modf(_uvs[:, 1] * h - 0.5)
    uf[uf < 0] = 1 + uf[uf < 0]
    vf[vf < 0] = 1 + vf[vf < 0]
    ui = np.maximum(ui.astype(int), 0)
    vi = np.maximum(vi.astype(int), 0)
    vals = np.column_stack([tex[vi, ui], tex[vi, np.minimum(ui + 1, w - 1)], tex[np.minimum(vi + 1, h - 1), ui], tex[np.minimum(vi + 1, h - 1), np.minimum(ui + 1, w - 1)]])
    return (1 - vf) * (1 - uf) * vals[:, 0] + (1 - vf) * uf * vals[:, 1] + vf * (1 - uf) * vals[:, 2] + vf * uf * vals[:, 3]

BASE_SCENE = {
    "type": "scene",
    "integrator": {
        "type": "path",
        # "sppi": 0,
        # "hide_emitters": True,
    },
    "emitter": {
        "type": "envmap",
        "filename": "test-data/envmaps/evening_sun.hdr",
        "scale": 1,
    },
    "mesh": {
        "type": "obj",
        "filename": "outputs/hres/ac_trees_0.6/ac_trees_0.6.obj",
        # "face_normals": True,
        "bsdf": {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [1, 1, 1]
            }
        }
    }
}

def get_sensor(elev, azi, radius=1, res=256):
    origin = [radius * np.cos(elev) * np.cos(azi), radius * np.sin(elev), -radius * np.cos(elev) * np.sin(azi)]
    sensor = {
        "type": "perspective",
        "fov": 45,
        "to_world": T.look_at(origin=origin, target=[0, 0, 0], up=[0, 0, -1]),
        
        "sampler": {
            "type": "independent",
            "sample_count": 512,
        },
        "film": {
            "type": "hdrfilm",
            "width": res,
            "height": res,
            "sample_border": True,
            "pixel_format": "rgb",
        },
    }
    return mi.load_dict(sensor)


CAM_POS = [(np.pi / 2, 0)] + [(np.pi / 4, i / 2 * np.pi) for i in range(4)]
# CAM_POS = [(np.pi / 2, 0)]
# 1.4 for 0.9, 0.98 for 0.6
RADIUS = 1
RES = 1024

if __name__ == "__main__":
    scene_dict = BASE_SCENE.copy()
    scene = mi.load_dict(scene_dict)

    imgs = [mi.render(scene, sensor=get_sensor(elev, azi, RADIUS, RES), spp=512) for (elev, azi) in CAM_POS]
    # H.save_images("test.png", imgs, auto_grid=True)
    mi.util.write_bitmap("test.png", imgs[0])
    # ii = H.read_image("test.png")
    # i = (np.array(mi.util.convert_to_bitmap(imgs[0])) / 255.0)
    # import matplotlib.pyplot as plt
    # print(i.shape)
    # plt.imsave("test2.png", i)

    # ii = np.load("outputs/fab_cat_0.6/hfield.npy")
    # plt.imsave("test3.png", ii, cmap="gray")
    # H.save_images("test2.png", [i, imgs[0]], auto_grid=True)
    # H.save_images(f"test.png", imgs, auto_grid=True)
