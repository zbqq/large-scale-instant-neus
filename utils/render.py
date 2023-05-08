            

# def render(model,ngp, rays_o, rays_d):# [N_samples 3]
def render(model,ngp,color_net, rays_o, rays_d):# [N_samples 3]
    """_summary_
    Args:
        rays_o (_type_): N_rays 3
        rays_d (_type_): N_rays 3
        perturb_overwrite (int, optional): _description_. Defaults to -1.
        background_rgb (_type_, optional): _description_. Defaults to None.
    """
    rays_o = rays_o.contiguous(); rays_d = rays_d.contiguous()
    # batch_size = len(rays_o)
    _,hits_t,__= RayAABBIntersector.apply(rays_o, rays_d, ngp.center, ngp.half_size, model.max_hits)#N_rays 1 2
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE#统一t1
    
    (rays_a, xyzs, dirs,#N_rays 3 [r_idx,start_idx,samples ];N_samples 1;N_samples 1
    deltas, ts, total_samples) = RayMarcher.apply(#N_samples 1
        rays_o, rays_d, hits_t[:,0],model.density_bitfield,
            ngp.cascades, ngp.scale,
            ngp.exp_step_factor,model.grid_size, MAX_SAMPLES)
    
    # sigmas,rgbs = ngp(xyzs,dirs,retu_fea=True)
    sdf,fea = ngp(xyzs,retu_fea=True)
    sigmas = torch.sigmoid(sdf)
    rgbs = color_net(dirs,fea)#N_samples 1;N_samples 3
    # sigmas和rgbs均为一维
    
    
    vr_samples,opacity,depth,rgb,ws = VolumeRenderer.apply(sigmas,rgbs.contiguous(),deltas,ts,rays_a,ngp.T_threshold)
    rgb_bg = torch.ones(3, device=rays_o.device)
    rgb = rgb + rgb_bg*rearrange(1-opacity, 'n -> n 1')
    
    return {
        'rgb': rgb,
        # 's_val': s_val,
        # 'cdf_fine': ret_fine['cdf'],
        'opacity':opacity,#尽量非0即1
        'ws': ws,
        'ts':ts,
        'deltas':deltas
        # 'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
        # 'gradients': gradients,
        # 'weights': weights,
        # 'gradient_error': ret_fine['gradient_error'],
        # 'inside_sphere': ret_fine['inside_sphere']
    }