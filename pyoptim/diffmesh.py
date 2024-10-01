import numpy as np
import skimage as ski

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
from mitsuba import ScalarTransform4f as T
import drjit as dr

import torch

from . import helpers as H
from . import cliploss as cl

BASE_SCENE = {
    "type": "scene",
    "integrator": {
        "type": "direct_projective",
        "sppi": 0,
        "hide_emitters": True,
    },
    "emitter": {
        "type": "envmap",
        "filename": "test-data/envmaps/evening_sun.hdr",
        "scale": 1.5,
    },
}

# helper class to perform differentiable rendering of a heightfield
# constructor takes the initial (flat) mesh, and saves the top-facing face
# also takes the target image as input
class DiffMesh:
    def __init__(self, box_Ps, box_Es, tgt_fname, vlim=None, device="cuda"):
        self.vlim = vlim
        # first we want to get the top-facing face of the box
        # we assume +x is right, +y is up, and -z is forward
        # get dimensions of box
        x_max, y_max, z_max = np.max(box_Ps, axis=0)
        x_min, y_min, z_min = np.min(box_Ps, axis=0)
        w, h, b = x_max - x_min, y_max - y_min, z_max - z_min
        # get the rows of box_Es where all(box_Ps[ei, 1] == h)
        tmp_Es = box_Es[np.all(box_Ps[box_Es, 1] == h, axis=1)]
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

        # initialize and save the semantic loss function
        self.semantic_loss = cl.CLIPConvLoss(torch.device(device), clip_conv_layer=3)
        self.device = device

        # set up the rendering things
        # prepare the reference scene
        # mitsuba flips UV coords of OBJ files
        bmp = mi.Bitmap(tgt_fname).convert(mi.Bitmap.PixelFormat.Y)
        ref_scene_dict = BASE_SCENE.copy()
        ref_scene_dict["ref_mesh"] = {
            "type": "obj",
            "filename": "pyoptim/plane_uv.obj",
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "bitmap",
                    "bitmap": bmp,
                }
            }
        }
        self.ref_scene = mi.load_dict(ref_scene_dict)

        # prepare the variable scene
        # create the mesh (in mitsuba)
        mesh = mi.Mesh("hfield", vertex_count=len(top_Ps), face_count=len(top_Es), has_vertex_normals=True, has_vertex_texcoords=True)
        mesh_params = mi.traverse(mesh)
        mesh_params["vertex_positions"] = mi.Float(top_Ps.ravel())
        mesh_params["faces"] = mi.Int(top_Es.ravel())
        mesh_params["vertex_texcoords"] = mi.Float(top_UVs.ravel())
        mesh_params["bsdf.reflectance.value"] = 1
        mesh_params.update()
        scene_dict = BASE_SCENE.copy()
        scene_dict["hfield"] = mesh
        self.scene = mi.load_dict(scene_dict)

        self.scene_params = mi.traverse(self.scene)
        self.positions_initial = dr.unravel(mi.Vector3f, self.scene_params["hfield.vertex_positions"])
        self.normals_initial = dr.unravel(mi.Vector3f, self.scene_params["hfield.vertex_normals"])

        self.hfield_si = dr.zeros(mi.SurfaceInteraction3f, dr.width(self.positions_initial))
        self.hfield_si.uv = dr.unravel(type(self.hfield_si.uv), self.scene_params["hfield.vertex_texcoords"])

        # create the bitmap texture from which we sample the heightfield
        self.hfield_tex = mi.load_dict({
            "type": "bitmap",
            "id": "hfield_tex",
            "bitmap": mi.Bitmap(dr.zeros(mi.TensorXf, (2, 2))),
            "raw": True
        })
        self.params = mi.traverse(self.hfield_tex)
        self.params.keep(["data"])
        dr.enable_grad(self.params["data"])
        

    def get_sensor(self, elev, azi, r=1, res=256):
        origin = [r * np.cos(elev) * np.cos(azi), r * np.sin(elev), -r * np.cos(elev) * np.sin(azi)]
        sensor = {
            "type": "perspective",
            "fov": 40,
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

    # gradient takes a heightfield texture (as a numpy array) and a camera, samples it onto the mesh, renders it
    # returns the loss wrt target image projected on the same flat mesh, as well as the gradient
    def gradient(self, hfield, elev, azi, r=1, res=256):
        img = self.render(hfield, elev, azi, r, res)

        # render the reference scene
        ref_img = mi.render(self.ref_scene, sensor=self.get_sensor(elev, azi, r=1, res=res))
        ref_img_torch = torch.from_numpy(np.array(mi.util.convert_to_bitmap(ref_img)) / 255.0).permute(2, 0, 1).unsqueeze(0)

        # wrap clip loss so it's compatible with drjit
        @dr.wrap_ad(source="drjit", target="torch")
        def view_clip_loss(img):
            img_torch = img.permute(2, 0, 1).unsqueeze(0)
            loss_dict = self.semantic_loss(img_torch, ref_img_torch)
            loss = 0
            for k in loss_dict:
                loss += loss_dict[k].float()
            return loss
        
        # actually compute the loss
        loss = view_clip_loss(img)
        # compute and extract the gradient
        dr.backward(loss)
        grad = dr.grad(self.params["data"])
        # return gradient as torch array
        return np.array(loss)[0], torch.from_numpy(np.array(grad).squeeze()).to(self.device)
        

    def render(self, hfield, elev, azi, r=1, res=256):
        sensor = self.get_sensor(elev, azi, r, res)
        # apply the new hfield mesh, we assume hfield is 2D (H x W)
        self.params["data"] = mi.TensorXf(hfield.squeeze()[:, :, np.newaxis])
        if self.vlim:
            self.params["data"] = dr.clamp(self.params["data"], -self.vlim, self.vlim)
        self.params.update()
        dr.enable_grad(self.params["data"])

        height_vals = self.hfield_tex.eval_1(self.hfield_si)
        new_positions = height_vals * self.normals_initial + self.positions_initial
        self.scene_params["hfield.vertex_positions"] = dr.ravel(new_positions)
        self.scene_params.update()
        
        # differentiably render image
        img = mi.render(self.scene, self.params, sensor=sensor)
        return img



    # don't use this, this is only here for testing purposes
    def save_mesh(self, out_name, hfield):
        # apply the new hfield mesh, we assume hfield is 2D (H x W)
        self.params["data"] = mi.TensorXf(hfield.squeeze()[:, :, np.newaxis])
        if self.vlim:
            self.params["data"] = dr.clamp(self.params["data"], -self.vlim, self.vlim)

        height_vals = self.hfield_tex.eval_1(self.hfield_si)
        new_positions = height_vals * self.normals_initial + self.positions_initial
        self.scene_params["hfield.vertex_positions"] = dr.ravel(new_positions)
        self.scene_params.update()

        mesh = [m for m in self.scene.shapes() if m.id() == "hfield"][0]
        mesh.write_ply(out_name)


# mitsuba's eval_1 assumes 0,0 as top-left
def rect_points_to_uv(points, x_min=-0.3, x_max=0.3, z_min=-0.3, z_max=0.3, uvmin=0.0001, uvmax=0.9999):
    return np.column_stack([(points[:, 0] - x_min) / (x_max - x_min) * (uvmax - uvmin) + uvmin, (points[:, 2] - z_min) / (z_max - z_min) * (uvmax - uvmin) + uvmin])

