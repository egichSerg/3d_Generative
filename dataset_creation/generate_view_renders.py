# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os

import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes, IO
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftGouraudShader,
    SoftSilhouetteShader,
    look_at_view_transform,
    TexturesVertex
)


# create the default data directory
current_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(current_dir, ".", "data")


def generate_view_renders(
    num_views: int = 40, data_dir: str = DATA_DIR, object_name: str = 'cow.obj', img_size: int = 128, azimuth_range: float = 180, elev_level: float = 2.7
):
    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Load obj file
    obj_filename = os.path.join(data_dir, object_name)
    if not os.path.isfile(obj_filename):
        raise Exception(f"No such file :(\nFilename:{obj_filename}")
    
    mesh = IO().load_mesh(obj_filename, device=device)

    if mesh.textures is None:
        verts_colors = torch.full((1, mesh.verts_packed().shape[0], 3), 1.0, device=device)
        textures = TexturesVertex(verts_features=verts_colors)
        mesh.textures = textures

    # scale and center mesh by normalizing
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-(center.expand(N, 3)))
    mesh.scale_verts_((1.0 / float(scale)))

    
    # Get a batch of viewing angles.
    elev = torch.linspace(-elev_level, elev_level, 3).repeat(num_views)
    azim = torch.linspace(-azimuth_range, azimuth_range, num_views*3) + 180.0

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=img_size, blur_radius=0.0, faces_per_pixel=1, bin_size=0
    )

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftGouraudShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )

    meshes = mesh.extend(num_views*3)

    target_images = renderer(meshes, cameras=cameras, lights=lights)

    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(
        image_size=img_size, blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma, faces_per_pixel=50
    )

    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader(),
    )

    silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)
    silhouette_binary = (silhouette_images[..., 3] > 1e-4).float()

    return cameras, target_images[..., :3], silhouette_binary
