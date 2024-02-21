import torch.nn as nn
import torch

class MeanVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict
