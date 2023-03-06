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
K-Planes field implementations using tiny-cuda-nn, torch, ....
"""
import itertools
from typing import Optional, Sequence

import tinycudann as tcnn
import torch
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from nerfstudio.fields.instant_ngp_field import get_normalized_directions
from nerfstudio.utils.interpolation import grid_sample_wrapper


def interpolate_kplanes(
    pts: TensorType,
    ms_grids: Sequence[Sequence[torch.nn.Module]],
    concat_features: bool,
) -> TensorType:
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), 2))
    multi_scale_interp = [] if concat_features else 0.0
    grid: torch.nn.ParameterList
    for scale_id, grid in enumerate(ms_grids):  # type: ignore
        interp_space = 1.0
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(-1, feature_dim)
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)  # type: ignore
        else:
            multi_scale_interp = multi_scale_interp + interp_space  # type: ignore

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)  # type: ignore
    return multi_scale_interp  # type: ignore


def init_grid_param(
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5) -> torch.nn.ParameterList:
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    coo_combs = list(itertools.combinations(range(in_dim), 2))
    grid_coefs = torch.nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = torch.nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            torch.nn.init.ones_(new_grid_coef)
        else:  # Initialize spatial planes as uniform[a, b]
            torch.nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


class KPlanesField(Field):
    """
    Args:
        aabb: parameters of scene aabb bounds
        use_appearance_embedding: whether to use appearance embedding
        num_images: number of images, requried if use_appearance_embedding is True
        appearance_embedding_dim: dimension of appearance embedding
    """

    def __init__(
        self,
        aabb,
        space_resolution: Sequence[int] = (256, 256, 256),
        multiscale_multipliers: Sequence[int] = (1, 2),
        time_resolution: int = 150,
        feature_dim: int = 32,
        is_dynamic: bool = True,
        concat_features_across_scales: bool = True,
        use_linear_decoder: bool = False,
        linear_decoder_layers: int = 1,
        use_appearance_embedding: bool = False,
        num_images: Optional[int] = None,
        appearance_embedding_dim: int = 32,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.concat_features_across_scales = concat_features_across_scales
        self.is_dynamic = is_dynamic
        self.use_linear_decoder = use_linear_decoder
        self.multiscale_multipliers = multiscale_multipliers
        self.feature_dim = (
            feature_dim * len(self.multiscale_multipliers)
            if self.concat_features_across_scales
            else feature_dim
        )

        # 1. Initialize planes
        self.grids = torch.nn.ModuleList()
        for res in self.multiscale_multipliers:
            resolution = [r * res for r in space_resolution]
            if self.is_dynamic:
                resolution.append(time_resolution)
            self.grids.append(init_grid_param(
                in_dim=4 if self.is_dynamic else 3,
                out_dim=feature_dim,
                reso=resolution,
            ))

        # 2. Init appearance code-related parameters
        self.use_appearance_embedding = use_appearance_embedding
        self.appearance_embedding_dim = 0
        if use_appearance_embedding:
            assert num_images is not None
            self.appearance_embedding_dim = appearance_embedding_dim
            # this will initialize as normal_(0.0, 1.0)
            self.appearance_embedding = Embedding(num_images, appearance_embedding_dim)

        # 3. Init decoder params
        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        if self.use_linear_decoder:
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for
            # combining the color features into RGB. This architecture is based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.appearance_embedding_dim,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.geo_feat_dim = 15
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.in_dim_color = (
                    self.direction_encoder.n_output_dims
                    + self.geo_feat_dim
                    + self.appearance_embedding_dim
            )
            self.color_net = tcnn.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0  # from [-2, 2] to [-1, 1]
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
        n_rays, n_samples = positions.shape[:2]

        timestamps = ray_samples.times
        if timestamps is not None:
            # Normalize timestamps from [0, 1] to [-1, 1]
            timestamps = (timestamps * 2) - 1
            positions = torch.cat((positions, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        positions = positions.reshape(-1, positions.shape[-1])

        features = interpolate_kplanes(
            positions,
            ms_grids=self.grids,  # notype
            concat_features=self.concat_features_across_scales,
        )

        if len(features) < 1:
            features = torch.zeros((0, 1), device=features.device)
        if self.use_linear_decoder:
            density_before_activation = self.sigma_net(features)  # [batch, 1]
        else:
            features = self.sigma_net(features)
            features, density_before_activation = torch.split(
                features, [self.geo_feat_dim, 1], dim=-1
            )
        density = trunc_exp(density_before_activation.to(positions)).view(n_rays, n_samples, 1)
        return density, features

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        assert density_embedding is not None
        n_rays, n_samples = ray_samples.frustums.shape
        directions = ray_samples.frustums.directions
        directions = directions.reshape(-1, 3)

        if self.use_linear_decoder:
            color_features = [density_embedding]
        else:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)
            color_features = [encoded_directions, density_embedding.view(-1, self.geo_feat_dim)]

        if self.use_appearance_embedding:
            assert ray_samples.camera_indices is not None
            camera_indices = ray_samples.camera_indices.squeeze()
            if self.training:
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                # Average of appearance embeddings for test data
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.appearance_embedding.mean(dim=0)

            # expand embedded_appearance from n_rays, dim to n_rays*n_samples, dim
            ea_dim = embedded_appearance.shape[-1]
            embedded_appearance = (
                embedded_appearance.view(-1, 1, ea_dim)
                                   .expand(n_rays, n_samples, -1)
                                   .reshape(-1, ea_dim)
            )
            if self.use_linear_decoder:
                directions = torch.cat((directions, embedded_appearance), dim=-1)
            else:
                color_features.append(embedded_appearance)

        color_features = torch.cat(color_features, dim=-1)

        if self.use_linear_decoder:
            basis_values = self.color_basis(directions)  # [batch, color_feature_len * 3]
            basis_values = basis_values.view(color_features.shape[0], 3, -1)  # [batch, 3, color_feature_len]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = rgb.to(directions)
            rgb = torch.sigmoid(rgb).view(n_rays, n_samples, 3)
        else:
            rgb = self.color_net(color_features).to(directions).view(n_rays, n_samples, 3)

        return rgb

    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        mask: Optional[TensorType] = None,
        bg_color: Optional[TensorType] = None,
    ):
        density, density_features = self.get_density(ray_samples)
        rgb = self.get_outputs(ray_samples, density_features)  # type: ignore

        return {FieldHeadNames.DENSITY: density, FieldHeadNames.RGB: rgb}

