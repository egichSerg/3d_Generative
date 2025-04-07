import os, json, sys, glob, time, math
import torch

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path

from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
FoVPerspectiveCameras, VolumeRenderer,
NDCGridRaysampler, EmissionAbsorptionRaymarcher
)

from generate_view_renders import generate_view_renders


def huber(x, y, scaling=0.1):
    diff_sq = (x-y) ** 2
    loss = ((1 + diff_sq / (scaling ** 2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss


class VolumeModel(torch.nn.Module):
    def __init__(self, renderer, volume_size = [64] * 3, voxel_size = 0.1):
        super().__init__()
        
        self.log_densities = torch.nn.Parameter(
            -4.0 * torch.ones(1, *volume_size)
        )
        self.log_colors = torch.nn.Parameter(
            torch.zeros(3, *volume_size)
        )
        self._voxel_size = voxel_size
        self._renderer = renderer

        self.colors = None
        self.densities = None

    def forward(self, cameras):
        batch_size = cameras.R.shape[0]
        densities = torch.sigmoid(self.log_densities)
        colors = torch.sigmoid(self.log_colors)
        volumes = Volumes(
            densities = densities.unsqueeze(0).expand(
                batch_size, *self.log_densities.shape
            ),
            features = colors.unsqueeze(0).expand(
                batch_size, *self.log_colors.shape
            ),
            voxel_size = self._voxel_size
        )

        self.colors = colors.detach()
        self.densities = densities.detach()

        return self._renderer(cameras=cameras, volumes=volumes)[0]

    def get_volumes(self):
        return {'colors': self.colors.cpu(), 'densities': self.densities.cpu()}


class DatasetCreator:
    def __init__(self, render_size: int = 256,  volume_size: int = 256, volume_extent_world: float = 3.0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.root_dir = os.path.dirname(os.path.realpath(__file__))

        self.render_size = render_size
        self.volume_extent_world = volume_extent_world
        self.volume_size = [volume_size] * 3
        self.voxel_size = volume_extent_world / volume_size
        
        raysampler = NDCGridRaysampler(
            image_width = render_size,
            image_height = render_size,
            n_pts_per_ray = 150,
            min_depth = 0.1,
            max_depth = volume_extent_world,
        )

        raymarcher = EmissionAbsorptionRaymarcher()
        
        renderer = VolumeRenderer(
            raysampler=raysampler,
            raymarcher=raymarcher,
        )
        
        self.volume_model = VolumeModel(
            renderer,
            volume_size = [volume_size] * 3,
            voxel_size = volume_extent_world / volume_size
        ).to(self.device)

    
    def reset_volume_model(self):
        del self.volume_model      
        torch.cuda.empty_cache()

        raysampler = NDCGridRaysampler(
            image_width = self.render_size,
            image_height = self.render_size,
            n_pts_per_ray = 150,
            min_depth = 0.1,
            max_depth = self.volume_extent_world,
        )

        raymarcher = EmissionAbsorptionRaymarcher()
        
        renderer = VolumeRenderer(
            raysampler=raysampler,
            raymarcher=raymarcher,
        )
        
        self.volume_model = VolumeModel(
            renderer,
            volume_size = self.volume_size,
            voxel_size = self.voxel_size
        ).to(self.device)
        # for layer in self.volume_model.children():
        #    if hasattr(layer, 'reset_parameters'):
        #        layer.reset_parameters()

    
    def get_obj_from_dir(self, data_dir):
        return glob.glob(f'{self.root_dir}/{data_dir}/*.obj')[0]

    
    def get_off_from_dir(self, data_dir):
        return glob.glob(f'{self.root_dir}/{data_dir}/*.off')[0]


    def save_triplane(self, save_dir, obj_path, obj_name, csv_file):
        volumes = self.volume_model.get_volumes()
        root_dir = Path(self.root_dir)
        obj_path = obj_path.split('/')[1]
        class_name = obj_name.split('_')[0]
        (root_dir / save_dir / class_name / obj_path / obj_name).mkdir(parents=True, exist_ok=True)
        
        colors_path = root_dir / save_dir / class_name / obj_path / obj_name / f'{obj_name}_rgb.pt'
        torch.save(volumes['colors'], colors_path)

        densities_path = root_dir / save_dir / class_name / obj_path / obj_name / f'{obj_name}_d.pt'
        torch.save(volumes['densities'], densities_path)
        
        with open(csv_file, 'a') as file:
            file.write(f'{obj_name},{class_name},{obj_path},{colors_path},{densities_path}\n')

    
    def save_grid(self, renders, save_dir, obj_path, obj_name):
        root_dir = Path(self.root_dir)
        obj_path = obj_path.split('/')[1]
        class_name = obj_name.split('_')[0]
        fig, ax = plt.subplots(2, 2, figsize=(6, 6))
        ax = ax.ravel()
        ax[0].imshow(renders[0])
        ax[1].imshow(renders[1])
        ax[2].imshow(renders[2])
        ax[3].imshow(renders[3])
        
        fig.savefig(root_dir / save_dir / class_name / obj_path / obj_name / f'{obj_name}.png')
    
    
    def create_triplane(self, data_dir: str, dataset_dir: str, object_name: str, csv_save_file: str, batch_size: int = 10, learning_rate: float = 0.1):

        #set scene
        # obj = self.get_obj_from_dir(data_dir=data_dir)
        obj = object_name
        target_cameras, target_images, target_silhouttes = generate_view_renders(num_views = 40, data_dir=data_dir, object_name=obj, img_size=self.render_size)
        torch.cuda.empty_cache()
        
        target_cameras = target_cameras.to(self.device)
        target_images = target_images.to(self.device)
        target_silhouttes = target_silhouttes.to(self.device)

        #set model for training
        optimizer = torch.optim.Adam(self.volume_model.parameters())
        lr = learning_rate
        batch_size = batch_size
        

        loss_item = 1000.0
        n_iter = 3000            # expected number of iterations
        iteration = 0
        # train
        self.volume_model.train()
        with tqdm(total=n_iter) as pbar:
            # for iteration in tqdm(range(n_iter)):
            while loss_item > 0.02:
                self.volume_model.train()
                if iteration == round(n_iter * 0.75):
                    optimizer = torch.optim.Adam(self.volume_model.parameters(), lr = lr * 0.1)
            
                # forward pass
                batch_idx = torch.randperm(len(target_cameras))[:batch_size]
            
                batch_cameras = FoVPerspectiveCameras(
                    R = target_cameras.R[batch_idx],
                    T = target_cameras.T[batch_idx],
                    zfar = target_cameras.zfar[batch_idx],
                    aspect_ratio = target_cameras.aspect_ratio[batch_idx],
                    fov = target_cameras.fov[batch_idx],
                    device = self.device
                )
            
                rendered_images, rendered_silhouttes = self.volume_model(batch_cameras).split([3, 1], dim = -1)
    
                # calculating the loss
                sil_err = huber(
                    rendered_silhouttes[..., 0],
                    target_silhouttes[batch_idx]
                ).abs().mean()
            
                color_err = huber(
                    rendered_images,
                    target_images[batch_idx]
                ).abs().mean()
            
                loss = color_err + sil_err
            
                # rest is following
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                loss_item = loss.item()
                pbar.update(1)
                iteration += 1


        print(f'Final Loss: {loss.item()}')
        # get results
        self.volume_model.eval()
        with torch.no_grad():
            rendered_images_eval, rendered_silhouttes_eval = self.volume_model(target_cameras[[0, 29, 59, 89]]).split([3, 1], dim = -1)

        renders = rendered_images_eval.detach().cpu().numpy()
        # save triplane
        obj_name = obj.split('/')[-1].split('.')[0]
        self.save_triplane(save_dir=dataset_dir, obj_path=obj, obj_name=obj_name, csv_file=csv_save_file)
        self.save_grid(renders=renders, save_dir=dataset_dir, obj_path=obj, obj_name=obj_name)
        
        # reset model for next mesh
        self.reset_volume_model()
        torch.cuda.empty_cache()
        
        return renders
          