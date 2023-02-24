# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Implementation of K-Planes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Type, Literal

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import KPlanesDensityField
from nerfstudio.fields.kplanes_field import KPlanesField
from nerfstudio.model_components.losses import (
    MSELoss, distortion_loss, interlevel_loss,
    plane_tv_loss, l1_time_planes_loss, time_smoothness_loss
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler, UniformSampler,
    UniformLinDispPiecewiseSampler
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc


@dataclass
class KPlanesModelConfig(ModelConfig):
    """K-Planes model config"""

    _target: Type = field(default_factory=lambda: KPlanesModel)
    """target class to instantiate"""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    space_resolution: Sequence[int] = (256, 256, 256)
    """"""
    multiscale_multipliers: Sequence[int] = (1, 2)
    """"""
    time_resolution: int = 150
    """"""
    feature_dim: int = 32
    """"""
    is_dynamic: bool = True
    """"""
    concat_features_across_scales: bool = False
    """"""
    use_linear_decoder: bool = False
    """"""
    linear_decoder_layers: int = 1
    """"""
    use_appearance_embedding: bool = False
    """"""
    appearance_embedding_dim: int = 32
    """"""
    spatial_distortion: str = "contraction"
    global_translation: Optional[torch.Tensor] = None  # TODO: Unused
    global_scale: Optional[torch.Tensor] = None  # TODO: Unused

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
            {"feature_dim": 8, "resolution": (128, 128, 128)},
            {"feature_dim": 8, "resolution": (256, 256, 256)},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""

    # Regularization weights
    distortion_loss_mult: float = 0.0
    interlevel_loss_mult: float = 1.0
    plane_tv_mult: float = 0.0
    plane_tv_propnets_mult: float = 0.0
    l1_time_planes_mult: float = 0.0
    l1_time_planes_propnets_mult: float = 0.0
    time_smoothness_mult: float = 0.0
    time_smoothness_propnets_mult: float = 0.0


# noinspection PyAttributeOutsideInit
class KPlanesModel(Model):
    """K-Planes Model
    """

    config: KPlanesModelConfig

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        spatial_distortion = None
        if self.config.spatial_distortion == "contraction":
            spatial_distortion = SceneContraction(order=float("inf"))

        self.field = KPlanesField(
            aabb=self.scene_box.aabb,
            space_resolution=self.config.space_resolution,
            multiscale_multipliers=self.config.multiscale_multipliers,
            time_resolution=self.config.time_resolution,
            feature_dim=self.config.feature_dim,
            is_dynamic=self.config.is_dynamic,
            concat_features_across_scales=self.config.concat_features_across_scales,
            use_linear_decoder=self.config.use_linear_decoder,
            linear_decoder_layers=self.config.linear_decoder_layers,
            use_appearance_embedding=self.config.use_appearance_embedding,
            num_images=self.num_train_data,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            spatial_distortion=spatial_distortion,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = KPlanesDensityField(
                aabb=self.scene_box.aabb,
                resolution=prop_net_args["resolution"],
                num_output_coords=prop_net_args["feature_dim"],
                spatial_distortion=spatial_distortion,
                use_linear_decoder=self.config.use_linear_decoder,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = KPlanesDensityField(
                    aabb=self.scene_box.aabb,
                    resolution=prop_net_args["resolution"],
                    num_output_coords=prop_net_args["feature_dim"],
                    spatial_distortion=spatial_distortion,
                    use_linear_decoder=self.config.use_linear_decoder,
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
        initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=self.config.use_single_jitter)
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

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # colliders
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
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

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {
            "proposal_networks": list(self.proposal_networks.parameters()),
            "fields": list(self.field.parameters())
        }
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # uniform sampling
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )
        field_outputs = self.field(ray_samples)
        rgb, density = field_outputs[FieldHeadNames.RGB], field_outputs[FieldHeadNames.DENSITY]
        weights = ray_samples.get_weights(density)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=rgb, weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(
                outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])
        loss_dict = {"rgb_loss": rgb_loss}

        if self.training:  # TODO: if mult == 0 avoid computing
            # Distortion loss
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            # Interlevel loss: used for training the proposal sampler (aka histogram loss)
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            # TV on planes
            loss_dict["plane_tv"] = self.config.plane_tv_mult * plane_tv_loss(
                self.field.grids
            )
            loss_dict["plane_tv_propnets"] = self.config.plane_tv_propnets_mult * plane_tv_loss(
                [p.grids for p in self.proposal_networks]
            )
            # L1 loss on the time planes (to force them around 1)
            loss_dict["l1_time_planes"] = self.config.l1_time_planes_mult * l1_time_planes_loss(
                self.field.grids
            )
            loss_dict["l1_time_planes_propnets"] = self.config.l1_time_planes_propnets_mult * l1_time_planes_loss(
                [p.grids for p in self.proposal_networks]
            )
            # Time-smoothness regularizer
            loss_dict["time_smoothness"] = self.config.time_smoothness_mult * time_smoothness_loss(
                self.field.grids
            )
            loss_dict["time_smoothness_propnets"] = self.config.time_smoothness_propnets_mult * time_smoothness_loss(
                [p.grids for p in self.proposal_networks]
            )

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth}
        return metrics_dict, images_dict
