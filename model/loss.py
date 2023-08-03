import torch
from torch import nn
import studio
import torch.nn.functional as F

def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()

class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            studio.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
        ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = studio.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


class NeRFLoss(nn.Module):
    def __init__(self,config, lambda_opacity=5e-2, lambda_distortion=1e-3):
        super().__init__()
        self.config = config
        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion

    def forward(self, results, target, **kwargs):
        d = {}
        # rays_valid = results['rays_valid'].view(-1)
        # rays_valid = target['fg_mask'].view(-1).to(torch.bool)
        
        # d['rgb'] = F.mse_loss(results['rgb'],target['rays'].to(self.device),reduction='mean') * self.config.lambda_rgb
        # d['rgb'] = (results['rgb']-target['rays'])**2 * self.config.lambda_rgb
        # if rays_valid.sum() > 0:
        #     d['rgb'] = F.smooth_l1_loss(results['rgb'][rays_valid],target['rays'][rays_valid]) * self.config.lambda_rgb
        # else:
        #     d['rgb'] = F.smooth_l1_loss(results['rgb'],target['rays']) * self.config.lambda_rgb
        
        d['rgb'] = F.smooth_l1_loss(results['rgb'],target['rays']) * self.config.lambda_rgb
        # d['rgb'] = F.smooth_l1_loss(results['rgb'][rays_valid],target['rays'][rays_valid]) * self.config.lambda_rgb#训练的epoch小的话很不好
        opacity = results['opacity']
        # o = results['opacity'][results['rays_valid']]+1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opacity*(-opacity*torch.log(opacity))*self.config.lambda_opacity
        # d['opacity'] = binary_cross_entropy(opacity, target['fg_mask'].float())*self.config.lambda_opacity
        if self.config.use_normal:
            d['eikonal']=((torch.linalg.norm(results['grad'], ord=2, dim=-1) - 1.)**2).mean()*self.config.lambda_eikonal
        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                DistortionLoss.apply(results['ws'], results['deltas'],
                                     results['ts'], results['rays_a'])

        return d
