import torch
import numpy as np
from kornia import create_meshgrid
from einops import rearrange
import cv2
import torch.nn.functional as F

@torch.cuda.amp.autocast(dtype=torch.float32)
def get_ray_directions(H, W, K, device='cpu', random=False, return_uv=False, flatten=True):
    
    grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if random:
        directions = \
            torch.stack([(u-cx+torch.rand_like(u))/fx,
                         (v-cy+torch.rand_like(v))/fy,
                         torch.ones_like(u)], -1)
    else: # pass by the center
        directions = \
            torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)

    if return_uv:
        return directions, grid
    return directions


# @torch.cuda.amp.autocast(dtype=torch.float32)
# def get_rays(directions, c2w):
    
#     if c2w.ndim==2:
#         # Rotate ray directions from camera coordinate to the world coordinate
#         rays_d = directions @ c2w[:, :3].T
#     else:
#         rays_d = rearrange(directions, 'n c -> n 1 c') @ \
#                  rearrange(c2w[..., :3], 'n a b -> n b a')
#         rays_d = rearrange(rays_d, 'n 1 c -> n c')
#     # The origin of all rays is the camera origin in world coordinate
#     rays_o = c2w[..., 3].expand_as(rays_d)

#     return rays_o, rays_d

@torch.cuda.amp.autocast(dtype=torch.float32)
def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        rays_o = c2w[:,:,3].expand(rays_d.shape)
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
            rays_o = c2w[None,None,:,3].expand(rays_d.shape)
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
            rays_o = c2w[:,None,None,:,3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d




@torch.cuda.amp.autocast(dtype=torch.float32)
def axisangle_to_R(v):
    """
    Convert an axis-angle vector to rotation matrix
    from https://github.com/ActiveVisionLab/nerfmm/blob/main/utils/lie_group_helper.py#L47

    Inputs:
        v: (3) or (B, 3)
    
    Outputs:
        R: (3, 3) or (B, 3, 3)
    """
    v_ndim = v.ndim
    if v_ndim==1:
        v = rearrange(v, 'c -> 1 c')
    zero = torch.zeros_like(v[:, :1]) # (B, 1)
    skew_v0 = torch.cat([zero, -v[:, 2:3], v[:, 1:2]], 1) # (B, 3)
    skew_v1 = torch.cat([v[:, 2:3], zero, -v[:, 0:1]], 1)
    skew_v2 = torch.cat([-v[:, 1:2], v[:, 0:1], zero], 1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1) # (B, 3, 3)

    norm_v = rearrange(torch.norm(v, dim=1)+1e-7, 'b -> b 1 1')
    eye = torch.eye(3, device=v.device)
    R = eye + (torch.sin(norm_v)/norm_v)*skew_v + \
        ((1-torch.cos(norm_v))/norm_v**2)*(skew_v@skew_v)
    if v_ndim==1:
        R = rearrange(R, '1 c d -> c d')
    return R


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses, pts3d=None):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of 3d point cloud (if None, center of cameras).
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    if pts3d is not None:
        center = pts3d.mean(0)
    else:
        center = poses[..., 3].mean(0)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses, pts3d=None):
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3) reconstructed point cloud

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pts3d_centered: (N, 3) centered point cloud
    """

    pose_avg = average_poses(poses, pts3d) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_inv = np.linalg.inv(pose_avg_homo)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_inv @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    if pts3d is not None:
        pts3d_centered = pts3d @ pose_avg_inv[:, :3].T + pose_avg_inv[:, 3:].T
        return poses_centered, pts3d_centered

    return poses_centered

def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    return center

def normalize_poses(poses, pts, up_est_method, center_est_method):
    if center_est_method == 'camera':
        # estimation scene center as the average of all camera positions
        center = poses[...,3].mean(0)
    elif center_est_method == 'lookat':
        # estimation scene center as the average of the intersection of selected pairs of camera rays
        cams_ori = poses[...,3]
        cams_dir = poses[:,:3,:3] @ torch.as_tensor([0.,0.,-1.])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1,0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1,0)
        t = torch.linalg.lstsq(A, b).solution
        center = (torch.stack([cams_dir, cams_dir.roll(1,0)], dim=-1) * t[:,None,:] + torch.stack([cams_ori, cams_ori.roll(1,0)], dim=-1)).mean((0,2))
    elif center_est_method == 'point':
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the final scene center
        center = poses[...,3].mean(0)
    else:
        raise NotImplementedError(f'Unknown center estimation method: {center_est_method}')

    if up_est_method == 'ground':
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        import pyransac3d as pyrsc
        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(pts.numpy(), thresh=0.01) # TODO: determine thresh based on scene scale
        plane_eq = torch.as_tensor(plane_eq) # A, B, C, D in Ax + By + Cz + D = 0
        z = F.normalize(plane_eq[:3], dim=-1) # plane normal as up direction
        signed_distance = (torch.cat([pts, torch.ones_like(pts[...,0:1])], dim=-1) * plane_eq).sum(-1)
        if signed_distance.mean() < 0:
            z = -z # flip the direction if points lie under the plane
    elif up_est_method == 'camera':
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[...,3] - center).mean(0), dim=0)
    else:
        raise NotImplementedError(f'Unknown up estimation method: {up_est_method}')

    # new axis
    y_ = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    if center_est_method == 'point':
        # rotation
        Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, torch.as_tensor([[0.,0.,0.]]).T], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]

        # translation and scaling
        poses_min, poses_max = poses_norm[...,3].min(0)[0], poses_norm[...,3].max(0)[0]
        pts_fg = pts[(poses_min[0] < pts[:,0]) & (pts[:,0] < poses_max[0]) & (poses_min[1] < pts[:,1]) & (pts[:,1] < poses_max[1])]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        poses_homo = torch.cat([poses_norm, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses_norm.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([torch.eye(3), t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        poses_norm[...,3] /= scale
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        pts = pts / scale
    else:
        # rotation and translation
        Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3] # (N_images, 4, 4)

        # scaling
        scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        poses_norm[...,3] /= scale

        # apply the transformation to the point cloud
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        pts = pts / scale

    return poses_norm, pts
def create_spheric_poses(radius, mean_h, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
        mean_h: mean camera height
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,2*mean_h],
            [0,0,1,-t]
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0],
            [0,np.cos(phi),-np.sin(phi)],
            [0,np.sin(phi), np.cos(phi)]
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th)],
            [0,1,0],
            [np.sin(th),0, np.cos(th)]
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0],[0,0,1],[0,1,0]]) @ c2w
        return c2w

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/12, radius)]
    return np.stack(spheric_poses, 0)

def get_aabb_imgmask(aabb,w2c,K):
        
        #     z  ^
        #        |________
        #       /|       /|
        #      /_|_____ / |      y
        #     |  |_____|__|____>
        #     | /      | /
        #   x |/_______|/
        #     / 
        #    /  
        #   <  
                          
              
    P_w2i = K @ w2c # [3,4]
    up = (P_w2i @ torch.tensor([
               [aabb[0],aabb[1],aabb[5],1],
               [aabb[3],aabb[1],aabb[5],1],
               [aabb[3],aabb[4],aabb[5],1],
               [aabb[0],aabb[4],aabb[5],1]]).T).T
    
    down = (P_w2i @ torch.tensor([
               [aabb[0],aabb[1],aabb[2],1],
               [aabb[3],aabb[1],aabb[2],1],
               [aabb[3],aabb[4],aabb[2],1],
               [aabb[0],aabb[4],aabb[2],1]]).T).T
    
    left = (P_w2i @ torch.tensor([
               [aabb[0],aabb[1],aabb[2],1],
               [aabb[3],aabb[1],aabb[2],1],
               [aabb[3],aabb[1],aabb[5],1],
               [aabb[0],aabb[1],aabb[5],1]]).T).T
    
    right = (P_w2i @ torch.tensor([
               [aabb[0],aabb[4],aabb[2],1],
               [aabb[3],aabb[4],aabb[2],1],
               [aabb[3],aabb[4],aabb[5],1],
               [aabb[0],aabb[4],aabb[5],1]]).T).T
    
    forward = (P_w2i @ torch.tensor([
               [aabb[3],aabb[1],aabb[2],1],
               [aabb[3],aabb[4],aabb[2],1],
               [aabb[3],aabb[4],aabb[5],1],
               [aabb[3],aabb[1],aabb[5],1]]).T).T
    
    back = (P_w2i @ torch.tensor([
               [aabb[0],aabb[1],aabb[2],1],
               [aabb[0],aabb[4],aabb[2],1],
               [aabb[0],aabb[4],aabb[5],1],
               [aabb[0],aabb[1],aabb[5],1]]).T).T
    
    faces_image = torch.stack([up,down,left,right,forward,back]) # [6,4,3]
    for i in range(0,faces_image.shape[0]):
        faces_image[i:i+1,:,:3] /= faces_image[i:i+1,:,2:] 
    faces_image = faces_image[:,:,:2].to(torch.int32).numpy()
    return faces_image


BGR={0:np.array([0,0,0],dtype=np.uint8),
     1:np.array([0,0,128],dtype=np.uint8),
     2:np.array([0,128,0],dtype=np.uint8),
     4:np.array([128,0,0],dtype=np.uint8),
     8:np.array([32,32,32],dtype=np.uint8)}

def draw_aabb_mask(image,w2c,K,aabbs):
    if isinstance(image,torch.Tensor):
        image = (image/image.max()*255).to(torch.uint8).cpu().numpy()
    image_with_aabbmask = image.copy()
    
    assert aabbs.shape[0] <= 16
    color = BGR[0]
    for i in range(0,aabbs.shape[0]):
        mask = np.zeros_like(image,dtype=np.uint8)
        mask_pts = get_aabb_imgmask(aabbs[i],w2c,K)
        mask_pts[0]
        color = list(BGR[1 & (i+1)] + BGR[2 & (i+1)] + BGR[4 & (i+1)] + BGR[8 & (i+1)])
        color = [int(comp) for comp in color]
        
        mask = cv2.fillPoly(mask, [mask_pts[0]], color) # up face
        image_with_aabbmask = cv2.addWeighted(image_with_aabbmask,1.0,mask,0.4,0)
        # for j in range(0,mask_pts.shape[0]):
        #     mask = cv2.fillPoly(mask, [mask_pts[j]], color) # up face
        #     image_with_aabbmask = cv2.addWeighted(image_with_aabbmask,1.0,mask,0.4,0)
        pass
    return image_with_aabbmask.astype(np.float32)/255.0