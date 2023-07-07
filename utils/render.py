            
import torch
import studio
from model.custom_functions import rendering_with_alpha,rendering_W_from_alpha,\
    march_rays_train, near_far_from_aabb, composite_rays_train, \
        morton3D, morton3D_invert, packbits,march_rays,composite_rays

def get_alphas(sigma, dists):#计算
    alphas = torch.ones_like(sigma) - torch.exp(- sigma * dists)
    return alphas.view(-1,1)

def get_rays_indices(values):
    n_rays, n_samples = values.shape # [N_rays, N_samples]
    rays = torch.zeros([n_rays,2],dtype=torch.int32,device=values.device)
    offsets = torch.arange(0,n_rays,1) * n_samples
    samples = torch.ones([n_rays]) * n_samples
    # rays[:,0] = torch.cumsum(sample_each_ray,0)
    rays[:,0] = offsets
    rays[:,1] = samples
    return rays

    
def sample_pdf(bins, weights, n_sample_importance, det=False):#64个三维采样点，16个分位点
    # 根据pdf采样，pdf是由weight计算的
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)#[batch_size,N_sanmples]64
    cdf = torch.cumsum(pdf, -1)# [batch_size,N_sanmples]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_sample_importance, 1. - 0.5 / n_sample_importance, steps=n_sample_importance,device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_sample_importance])#每一条射线上都有一个cdf，这里的u是cdf的采样点[512,16]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_sample_importance])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)#从左开始找，返回一个shape和u相等，元素为cdf的每一行在第一个大于等于u的列索引+1
    #可以近似为分位点，返回u.shape:[512,16]
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)#截0
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)#截上限N_samples
    inds_g = torch.stack([below, above], -1)  # (N_rays, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]# [N_rays, N_samples, N_samples]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)#相当于插了一个映射矩阵inds_g 
    # 512 16 2，也即分位点取到cdf中
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)#cdf对应分位点的z_vals

    denom = (cdf_g[..., 1] - cdf_g[..., 0])#pdf
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])#dt，根据u和cdf的值对z_vals线性插值

    return samples

def up_sample(
    rays_o,rays_d,z_vals,
    n_importance,sample_dist,
    # b_bg,fb_ratio,
    geometry_network=None):
    assert geometry_network is not None
    pts = (rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]).view(-1,3) # [N_rays, N_samples, 3]
    # studio.contract_rect(pts,b_bg,fb_ratio,int(pts.shape[0]))
    geo_output = geometry_network(pts,with_grad=False,with_fea=False)
    
    sigmas = geo_output['sigma']
    
    dists = z_vals[...,1:] - z_vals[...,:-1] # [N_rays,N_samples - 1]    
    dists = torch.cat([dists, sample_dist.expand(dists[..., :1].shape)], -1).view(-1)
    alphas = get_alphas(sigmas,dists) # [N_rays * N_samples , 1]
    rays = get_rays_indices(alphas)
    weights = rendering_W_from_alpha(rays,alphas)
    dists = dists.view(z_vals.shape)
    weights = weights.view(z_vals.shape)[:,:-1]
    z_samples = sample_pdf(dists,weights,n_importance,det=True)
    return z_samples

def cat_z_vals(rays_o, rays_d, z_vals, new_z_vals,last=False):
    # batch_size, n_samples = z_vals.shape#n_samples为粗采样点数
    # _, n_importance = new_z_vals.shape#n_importance为细采样点数
    # pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
    z_vals, index = torch.sort(z_vals, dim=-1)
    if not last:
        pass

    return z_vals

def render_from_cdf(\
    rays_o,rays_d,
    nears,fars,
    config,b_bg,
    up_sample_steps=1,
    geometry_network=None,
    color_network=None
):
    assert geometry_network is not None
    assert color_network is not None
    """
    rays_o,rays_d: [N_rays,3]
    z_vals: [N_rays,N_samples]
    nears, fars: [N_rays]
    n_importance: int
    """
    device = rays_o.device
    n_importance=config.n_importance
    fb_ratio = torch.ones([1,1,1],dtype=torch.float32).to(device)*config.fb_ratio
    n_rays,n_samples = rays_o.shape[0],config.num_samples_per_ray
    z_vals = torch.linspace(0.0, 1.0, n_samples,device=device).view(1,-1)
    z_vals = (nears + (fars - nears)).view(-1,1) * z_vals # [N_samples]
    
    #fine sample
    with torch.no_grad(): 
        sample_dist = (fars - nears).view(-1,1) / n_samples

        for _ in range(0,up_sample_steps):
            
            z_vals_sample = up_sample(rays_o,rays_d,z_vals,n_importance,sample_dist,geometry_network)
            z_vals = cat_z_vals(rays_o,rays_d,z_vals,z_vals_sample)

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None] # [N_rays, N_samples, 3]
    pts = pts.view(-1,3)
    # studio.contract_rect(pts,b_bg,fb_ratio,int(pts.shape[0]))
    geo_output = geometry_network(pts,with_grad=False,with_fea=True)
    sigmas, feas = geo_output['sigma'],geo_output['fea']
    
    dists = z_vals[...,1:] - z_vals[...,:-1] # [N_rays,N_samples - 1]    
    dists = torch.cat([dists, sample_dist.expand(dists[..., :1].shape)], -1).view(-1)
    
    
    alphas = get_alphas(sigmas,dists).view(-1,1) # [N_rays * N_samples , 1]
    
    dirs = rays_d[:,None,:].expand(n_rays,z_vals.shape[1],3).reshape(-1,3)
    rays = get_rays_indices(z_vals)
    z_vals = z_vals.view(-1,1)
    dists = dists.view(-1,1)
    
    ts = torch.cat([z_vals,dists],dim=-1)
    
    # weights = rendering_W_from_alpha(rays,alphas)
    rgbs = color_network(dirs,feas)
    image,opacities,depth = rendering_with_alpha(alphas,rgbs,ts,rays,rays_o.shape[0])
    results = {
        'num_points':pts.shape[0],
        # 'weights':weights,
        # 'rays_valid':weights_sum>0,
        'opacity':torch.clamp(opacities,1e-12,1000),
        'num_points':pts.shape[0],
        }
    results['depth']=depth
    results['rgb']=image
    return results
def render_from_raymarch(\
    rays_o,rays_d,
    center,scale,density_bitfield,
    C,H,nears,fars,config,perturb,
    split,
    geometry_network=None,
    color_network=None,
    
):
    assert geometry_network is not None
    assert color_network is not None
    device = rays_o.device
    N=rays_o.shape[0]
    rays_o -= center.view(-1,3)#需要平移到以center为原点坐标系
    scene_aabb =scene_aabb - center.repeat([2])
    # assert in_aabb(rays_o[0,:],scene_aabb)
    # fb_ratio = torch.ones([1,1,1],dtype=torch.float32).to(device)*config.fb_ratio
    N=rays_o.shape[0]
    aabb = scene_aabb.clone()
    aabb[0:2] -= scale[:2]
    aabb[3:5] += scale[:2]#扩大一点使得far不会终止到aabb上
    nears,fars = near_far_from_aabb( # 位移不改变nears，fars
        rays_o,rays_d,aabb,0.02#确定far时需要把射线打到地面上，而不是在边界
    )
        
    
    fb_ratio = torch.ones([1,1,1],dtype=torch.float32).to(device)*config.fb_ratio
    results={}
    if split=='train' or split == 'val':
    # if split=='train':
        # with torch.no_grad():
        xyzs, dirs, ts, rays = \
            march_rays_train(rays_o, rays_d, scale, fb_ratio,
                                    True, density_bitfield, 
                                    C, H, 
                                    nears, fars, perturb, 
                                    config.dt_gamma, config.num_samples_per_ray,)
            
        # draw_poses(rays_o_=rays_o,rays_d_=rays_d,pts3d=xyzs.to('cpu'),aabb_=self.scene_aabb[None,...],t_min=nears,t_max=fars)
        # draw_poses(pts3d=xyzs.to('cpu'),aabb_=self.scene_aabb[None,...],t_min=nears,t_max=fars)
        
        xyzs += center.view(-1,3)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        with torch.cuda.amp.autocast(enabled=config.fp16):
            # geo_output = geometry_network(xyzs,with_fea=True,with_grad=False)
            # sigmas,feas = geo_output['sigma'],geo_output['fea']
            # rgbs = color_network(dirs,feas)
            geo_output = geometry_network(xyzs,with_fea=True,with_grad=True)
            
            if config.color_network.use_normal:
                geo_output = geometry_network(xyzs,with_fea=True,with_grad=True)
                sigmas,feas,normals,grad = geo_output['sigma'],geo_output['fea'],geo_output['normals'],geo_output["grad"]
                # rgbs = color_network(dirs,feas)
                rgbs = color_network(dirs,feas,normals)
            else:
                geo_output = geometry_network(xyzs,with_fea=True,with_grad=False)
                sigmas,feas = geo_output['sigma'],geo_output['fea']
                rgbs = color_network(dirs,feas)
        if config.rendering_from_alpha:
            if config.color_network.use_normal:
                alphas = get_alphas(sigmas,normals,dirs,ts[:,1])
            else:
                alphas = get_alphas(sigmas,ts[:,1])
            image,opacities,depth = rendering_with_alpha(alphas,rgbs,ts,rays,rays_o.shape[0])
            results = {
                'num_points':xyzs.shape[0],
                # 'weights':weights,
                # 'rays_valid':weights_sum>0,
                'opacity':torch.clamp(opacities,1e-12,1000),
                'num_points':xyzs.shape[0],
                
            }
            if config.color_network.use_normal:
                results['normals']=normals
                results['grad']=grad
        else:
            weights, weights_sum, depth, image = \
                composite_rays_train(sigmas, rgbs, ts, rays, config.T_thresh)
        
            results = {
                'num_points':xyzs.shape[0],
                'weights':weights,
                # 'rays_valid':weights_sum>0,
                'opacity':torch.clamp(weights_sum,1e-12,1000),
                'num_points':xyzs.shape[0]
            }
        # opacity = torch.clamp(weights_sum,1e-12,1000)
        
        
    elif split=='hhh':
    # elif split=='val':
        pass
        dtype = torch.float32
        weights_sum = torch.zeros(N, dtype=dtype, device=device)
        depth = torch.zeros(N, dtype=dtype, device=device)
        image = torch.zeros(N, 3, dtype=dtype, device=device)
        
        n_alive = N
        rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
        rays_t = nears.clone() # [N]
        step = 0
    
        while step < 100:
            # count alive rays 
            n_alive = rays_alive.shape[0]
            
            # exit loop
            if n_alive <= 0:
                break
            # decide compact_steps
            n_step = max(min(N // n_alive, 8), 1)
            xyzs, dirs, ts = march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, scale, True, density_bitfield, C, H, nears, fars, perturb if step == 0 else False, config.dt_gamma, config.num_samples_per_ray)
            
            dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
            with torch.cuda.amp.autocast(enabled=config.fp16):
                # outputs = self(xyzs, dirs, shading=shading)
                # sigmas = outputs['sigma']
                # rgbs = outputs['color']
                geo_output = geometry_network(xyzs,with_fea=True,with_grad=False)
                sigmas,feas = geo_output['sigma'],geo_output['fea']
                rgbs = color_network(dirs,feas)
                
            composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, config.T_thresh)
            rays_alive = rays_alive[rays_alive >= 0]
            # print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')
            step += n_step
    
    results['depth']=depth
    results['rgb']=image
    return results


