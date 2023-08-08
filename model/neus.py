
import math
import torch 
import torch.nn.functional as F
from nerfacc import rendering, ray_marching, OccupancyGrid, ContractionType,\
    ray_aabb_intersect,render_weight_from_alpha,accumulate_along_rays,render_weight_from_density
from scripts.load_tool import draw_poses
import torch.nn.functional as F
import torch
from .tcnn_nerf import SDF,RenderingNet,VarianceNetwork
from .base import baseModule
import model
NEAR_DISTANCE = 0.01

class NeuS(baseModule):
    def __init__(self,config):#config.model
        super().__init__(config)
        self.config = config
        
        # self.render_step_size = 1.732 * 2 * self.config.aabb.radius_z / self.config.num_samples_per_ray
        
        self.geometry_network = model.make(self.config.geometry_network.name,self.config.geometry_network)
        self.variance = model.make('vairance',self.config.point_sample.inv_cdf.init_variance)
        self.color_network = model.make(self.config.color_network.name,self.config.color_network)
        self.geometry_network.contract_type = ContractionType.AABB
        
        # self.register_buffer('background_color', torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32), persistent=False)
        
        self.cos_anneal_ratio = 1.0
        
        # self.loss = NeRFLoss(lambda_distortion=0)
        if self.config.learned_background:
            self.geometry_network_bg = model.make(self.config.geometry_network_bg.name, self.config.geometry_network_bg)
            self.color_network_bg = model.make(self.config.color_network_bg.name, self.config.color_network_bg)
            self.geometry_network_bg.contract_type = ContractionType.UN_BOUNDED_SPHERE
            self.near_plane_bg, self.far_plane_bg = 0.1, 1e3
            self.cone_angle_bg = 10**(math.log10(self.far_plane_bg) / self.config.point_sample.num_samples_per_ray_bg) - 1.
            self.render_step_size_bg = 0.01            
            
            
            
            
            
    def update_step(self,epoch,global_step):#更新cos_anneal_ratio
        cos_anneal_end = self.config.point_sample.inv_cdf.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)
        def occ_eval_fn(x):
            sdf = self.geometry_network(x, with_grad=False, with_fea=False)['sigma']
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
        def occ_eval_fn_bg(x):
            # out = self.geometry_network_bg(x)
            density = self.geometry_network_bg(x, with_fea=False, with_grad=False)["sigma"]
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size_bg) based on taylor series
            return density[...,None] * self.render_step_size_bg
        
        if self.config.point_sample.use_nerfacc:
            #没问题
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn,occ_thre=0.001)
            if self.config.learned_background:#没问题
                self.occupancy_grid_bg.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn_bg,occ_thre=0.01)
        # else:

        #     def occ_eval_fn(x):
        #         sdf = self.geometry_network(x, with_grad=False, with_fea=False)['sigma']
        #         inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        #         inv_s = inv_s.expand(sdf.shape[0], 1)
        #         estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
        #         estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
        #         prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        #         next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        #         p = prev_cdf - next_cdf
        #         c = prev_cdf
        #         alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
        #         return alpha.view(-1)
        #     self.update_extra_state(occ_eval_fn=lambda pts: occ_eval_fn(x=pts))
        
            
        
    # def get_alpha(self, sdf, normal, dirs, dists):
    def get_alpha(self, geo, dists):
        sdf,normal,dirs=geo['sigma'],geo['normals'],geo['dirs']
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-6) / (c + 1e-6)).view(-1).clip(0.0, 1.0)
        return alpha.reshape(-1)
    def forward(self,rays_o,rays_d,split):
        if self.config.point_sample.use_nerfacc:
            sdf_grad_samples = []


            def alpha_fn(t_starts, t_ends, ray_indices):
                ray_indices = ray_indices.long()
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                midpoints = (t_starts + t_ends) / 2.
                positions = t_origins + t_dirs * midpoints
                out = self.geometry_network(positions, with_fea=True, with_grad=True)
                out['dirs'] = t_dirs
                dists = t_ends - t_starts
                alpha = self.get_alpha(out, dists)#获取rho
                return alpha#[N_samples, 1]
            def rgb_alpha_fn(t_starts, t_ends, ray_indices):
                ray_indices = ray_indices.long()
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                midpoints = (t_starts + t_ends) / 2.
                positions = t_origins + t_dirs * midpoints
                out = self.geometry_network(positions, with_fea=True, with_grad=True)
                normals, sdf_grad, feature = out["normals"],out["grad"],out["fea"]
                out['dirs'] = t_dirs
                sdf_grad_samples.append(sdf_grad)
                dists = t_ends - t_starts
                alpha = self.get_alpha(out, dists)
                if self.config.color_network.use_normal:
                    rgb = self.color_network(t_dirs,feature,normals)
                else:
                    rgb = self.color_network(t_dirs,feature)
                return rgb, alpha
            # draw_poses(rays_o_=rays_o,rays_d_=rays_d,aabb_=self.scene_aabb)

            with torch.no_grad():#ray_marching过程中不累加nabla

                ray_indices, t_starts, t_ends = \
                    ray_marching(
                        rays_o,rays_d,
                        scene_aabb=self.scene_aabb,
                        grid = self.occupancy_grid,
                        alpha_fn = None,
                        near_plane=None, far_plane=None,
                        render_step_size=self.render_step_size,
                        stratified=True,
                        cone_angle = 0.0,
                        alpha_thre = 0.0
                    )
            n_rays = rays_o.shape[0]
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            dists = t_ends - t_starts

            # if self.config.geometry.grad_type == 'finite_difference':
                # sdf, sdf_grad, feature, sdf_laplace = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True)
            # else:
            out= self.geometry_network(positions, with_grad=True, with_fea=True)
            out['dirs'] = t_dirs
            sdf, sdf_grad, feature =out['sigma'],out['grad'],out['fea']
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            alpha = self.get_alpha(out, dists)[...,None]
            rgb = self.color_network(t_dirs,feature, normal)

            weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
            opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
            depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
            comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
            # comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)       
            comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
            comp_normal = F.normalize(comp_normal, p=2, dim=-1)




            # rgb, opacity, depth = rendering(
            #     t_starts,
            #     t_ends,
            #     ray_indices,
            #     n_rays=rays_o.shape[0],
            #     rgb_alpha_fn=rgb_alpha_fn,
            #     # render_bkgd=self.background_color,
            # )
            result = {
                'sdf': sdf,
                'rgb': comp_rgb,
                'comp_normal': comp_normal,
                # 'opacity': opacity,
                'opacity': torch.clamp(opacity,1.e-3,1.-1.e-3),
                'depth': depth,
                'rays_valid': opacity > 0,
                'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays_o.device),
                # 'grad': torch.concat(sdf_grad_samples,dim=0),
                "grad":sdf_grad,
                'inv_s':self.variance.inv_s
            } 
            if self.config.learned_background:
                result_bg = self.forward_bg(rays_o,rays_d)

            result_full = {
                "sdf": sdf,
                "grad":sdf_grad,
                "rgb": result["rgb"] + result_bg["rgb"] * (1.0 - result["opacity"]),
                "depth": result["depth"],
                "num_samples": result["num_samples"] + result_bg["num_samples"],
                "rays_valid": result["rays_valid"] | result_bg["rays_valid"],
                'inv_s':self.variance.inv_s
            }
            return result_full
        else:            
            cam_near = torch.ones([rays_o.shape[0],1],device=rays_o.device) * 0.
            # cam_far = torch.ones([rays_o.shape[0],1],device=rays_o.device) * torch.norm(self.scale)
            cam_far = torch.ones([rays_o.shape[0],1],device=rays_o.device) * 15
            cam_near_far = torch.cat([cam_near,cam_far],dim=-1)
            result = self.render(rays_o,rays_d,cam_near_far=cam_near_far,split=split,perturb=True)
            
            inv_s=self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            result['inv_s']=inv_s[0]
            return result
    def forward_bg(self,rays_o,rays_d):
        n_rays=rays_o.shape[0]
        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            out = self.geometry_network_bg(positions,with_grad=False,with_fea=False)
            density=out['sigma']
            return density[...,None] 
        _, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
        # if the ray intersects with the bounding box, start from the farther intersection point
        # otherwise start from self.far_plane_bg
        # note that in nerfacc t_max is set to 1e10 if there is no intersection
        # draw_poses(rays_o_=rays_o,rays_d_=rays_d,t_min=_,t_max=t_max,aabb_=self.scene_aabb[None,...])
        near_plane = torch.where(t_max > 1e9, torch.tensor(self.near_plane_bg,device=t_max.device,dtype=torch.float32), t_max)
        # 重叠
        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=None,
                grid=self.occupancy_grid_bg,
                sigma_fn=sigma_fn,
                near_plane=near_plane, far_plane=self.far_plane_bg,
                render_step_size=self.render_step_size_bg,
                stratified=True,
                cone_angle=self.cone_angle_bg,
                alpha_thre=0.0
            )       
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints  
        intervals = t_ends - t_starts

        out = self.geometry_network_bg(positions,with_grad=False,with_fea=True) 
        density,feature=out['sigma'],out['fea']
        rgb = self.color_network_bg(t_dirs,feature)

        weights = render_weight_from_density(t_starts, t_ends, density[...,None], ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)       

        out = {
            'rgb': comp_rgb,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays_o.device)
        }

        if self.training:
            out.update({
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': intervals.view(-1),
                'ray_indices': ray_indices.view(-1)
            })

        return out
        
        
        
        

    
    
    