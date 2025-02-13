# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import cv2
import torchvision
import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps


@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.step = 0
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        # Smoothing on the depth vectors 1e-6 (the best performance)
        # 1e-7 for uterus experience
        self.loss_depth = self.split(expected_depth) * 1e-6
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    # Splitting two by two (central selected ray and one of the up, down, left , right randomly and subtracting each of the splitted couples
    def split(self, tensor):
        if len(tensor) > 1:
            # Reshape the tensor to (2048, 2) by squeezing the singleton dimension
            reshaped_tensor = tensor.squeeze().reshape(-1, 2)
            # Calculate the differences between every two consecutive elements
            differences = torch.abs(reshaped_tensor[:, 1] - reshaped_tensor[:, 0])
            loss_depth = torch.sum(differences)
        else:
            loss_depth = torch.sum(torch.tensor(tensor))
        return loss_depth

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)+ self.loss_depth
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], dir,
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        # colormap_options= colormaps.ColormapOptions2
        # to compute scale factor between GT depth and rendered depth, we do not take into account the last rendered depth.
        #because they used colormap and convert single channel depth map into three color map. We do not want do it!
        #But we will use accumulation after normalization:
        # depth_norm = self.normalize(outputs["depth"])
        # output = torch.clip(output, 0, 1)
        # depth = depth_norm * outputs["accumulation"] + (1 - outputs["accumulation"])
        # sam = torch.Tensor.cpu(colored_image).numpy()

        mask_dir = dir/"mask_val_ERODE"
        specular_mask = sorted(os.listdir(mask_dir))[batch["image_idx"]]
        specular_read = torchvision.io.read_image(str(mask_dir/specular_mask)).permute([1,2,0]).bool()
        if "mask" in batch: #This mask comes from .json file
            mask = torch.cat((batch["mask"], batch["mask"], batch["mask"]), dim=2)
            # acc = torch.where(mask == 0.0, 1.0, acc)
            # acc = acc * mask
            # depth = depth * mask
            predicted_rgb = torch.where(mask == 0.0, 1.0, predicted_rgb)
            gt_rgb = torch.where(mask == 0.0, 1.0, gt_rgb)
        # Get corresponding Gt depth from "depth_2" folder
        # if os.path.exists(dir/"depths_2") and "mask" in batch:
            # depth_path = dir/"depths_2"
            # depth_GT = torchvision.io.read_image(str(depth_path / specular_mask)).permute([1, 2, 0]).cuda(0)[:, :, 0:1]
            # depth_GT = self.normalize(depth_GT)
            # depth_GT_mask = (depth_GT * mask)[:, :, 0:1]
            #get depths without highlight indices
            # depth_GT_mask_wo_spe = depth_GT_mask[specular_read[:, :, 0:1]].to(torch.float32)
            # depth_wo_spe = depth[specular_read[:, :, 0:1]]

            #Compute alpha and disparity between GT depth and rendered depth
            # depth_resh = torch.reshape(depth, [-1])
            # depth_GT_resh = torch.reshape(depth_GT_mask, [-1]).to(torch.float32)
            # alpha = torch.tensordot(torch.transpose(depth_resh, -1, 0), depth_GT_resh, dims=1) / torch.tensordot(torch.transpose(depth_resh,-1,0),depth_resh,dims=1)
            # alpha_wo_spe = torch.tensordot(torch.transpose(depth_wo_spe, -1,0), depth_GT_mask_wo_spe, dims=1) / torch.tensordot(torch.transpose(depth_wo_spe,-1,0),depth_wo_spe,dims=1)
            # disparity = torch.mean(torch.abs(depth_GT_mask - depth * alpha), dim=2)
            #Compute depth rmse
            # rmse_depth_wo_bg = self.rmse(depth_GT_resh, depth_resh * alpha) # Error for wo bg
            # rmse_depth_wo_spe = self.rmse(depth_GT_mask_wo_spe, depth_wo_spe * alpha_wo_spe)  #Error for wo spe


            #display disparity and convert it into torch to save them
            # fig = self.plot_disprity(disparity, rmse_depth_wo_bg, rmse_depth_wo_spe)
            # buf = io.BytesIO()
            # fig.savefig(buf, format='png')
            # buf.seek(0)
            # Read the buffer into a PIL image
            # image = Image.open(buf)
            # Convert the PIL image to a PyTorch tensor
            # tensor_disparity = torch.tensor(np.array(image)) / 255.0  # Normalize to [0, 1]
            # combined_disparity = torch.cat([tensor_disparity], dim=1)
            # combined_depth_GT = torch.cat([depth_GT_mask.to(torch.float32)], dim=1)
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Calculate psnr without specular pixels
        predicted_rgb_wo_specular = predicted_rgb[specular_read]
        gt_rgb_wo_specular = gt_rgb[specular_read]
        psnr_wo_spe = self.psnr(gt_rgb_wo_specular, predicted_rgb_wo_specular)
        # Calculate psnr, ssim and lpips with bg 0 (masked) if available
        if "mask" in batch:
            gt_rgb_w_bg_0 = torch.moveaxis(gt_rgb * mask, -1, 0)[None, ...]
            predicted_rgb_w_bg_0 = torch.moveaxis(predicted_rgb * mask, -1, 0)[None, ...]
            psnr_wo_bg = self.psnr(gt_rgb[mask], predicted_rgb[mask])
            ssim_w_bg_0 = self.ssim(gt_rgb_w_bg_0, predicted_rgb_w_bg_0)
            lpips_w_bg_0 = self.lpips(gt_rgb_w_bg_0, predicted_rgb_w_bg_0)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)
        # metrics_dict = {"psnr_wo_spe": float(psnr_wo_spe.item())}
        metrics_dict["psnr_wo_spe"] = float(psnr_wo_spe)
        if "mask" in batch:
            metrics_dict["psnr_wo_bg"] = float(psnr_wo_bg)
            metrics_dict["ssim_w_bg_0"] = float(ssim_w_bg_0)
            metrics_dict["lpips_w_bg_0"] = float(lpips_w_bg_0)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        # if os.path.exists(dir / "depths_2") and "mask" in batch:
            # Compute rmse depth between masked GT depth and masked rendered depth map
            # metrics_dict["rmse_depth_wo_bg"] = float(rmse_depth_wo_bg)
            # metrics_dict["rmse_depth_wo_spe"] = float(rmse_depth_wo_spe)
            # images_dict["disparity"] = combined_disparity # Add disparity map into image_dict
            # images_dict["depth_GT"] = combined_depth_GT


        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            if "mask" in batch:
                prop_depth_i = prop_depth_i
            else:
                prop_depth_i = prop_depth_i
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    def plot_disprity(self, disparity, rmse_value_bg, rmse_value_spe):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        rmse_bg = float(rmse_value_bg)
        rmse_spe = float(rmse_value_spe)
        frame_error = "Depth RMSE_wo_bg: "+ str(round(rmse_bg, ndigits= 4))+"(Mm)"+"Depth RMSE_wo_spe: "+ str(round(rmse_spe, ndigits=4))+"(Mn)"
        disparity = self.normalize(disparity) # Normalization
        # Plot dynamic colormap for disparity map
        fig = plt.figure()
        ax = fig.add_subplot()
        # # I like to position my colorbars this way, but you don't have to
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        cv0 = torch.Tensor.cpu(disparity).numpy()
        im = ax.imshow(cv0)  # Here make an AxesImage rather than contour
        # cb = fig.colorbar(im, cax=cax)
        # tx = ax.set_title(framename)
        # plt.savefig(output_path)
        fig.colorbar(im, cax=cax)
        ax.set_title(frame_error)
        return(fig)

    def normalize(self, x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 10e-10)
    def rmse(self, actual, pred): return torch.sqrt(torch.square(torch.subtract(actual, pred)).mean())