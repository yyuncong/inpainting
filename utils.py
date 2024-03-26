import numpy as np
import torch
from typing import Tuple
import re
from splatting import splatting_function


def get_camera_params(extrinsics, intrinsics, resolution):
    R = extrinsics[:, :3, :3]
    T = extrinsics[:, :3, -1]
    h, w, f = intrinsics[:, 1, 2] * 2, intrinsics[:, 0, 2] * 2, intrinsics[:, 0, 0]
    w_scale = resolution / w
    h_scale = resolution / h
    # perform element-wise multiplication
    K = intrinsics.clone()
    K[:, 0] *= w_scale.unsqueeze(1)
    K[:, 1] *= h_scale.unsqueeze(1)
    return R, T, K


def render_forward_splat(
    src_imgs, src_depths, R_src, t_src, K_src, R_dst, t_dst, K_dst
):
    """3D render the image to the next viewpoint.
    The input transformation matrix should be in nerf space

    Returns:
      warp_feature: the rendered RGB feature map
      warp_disp: the rendered disparity
      warp_mask: the rendered mask
    """

    batch_size = src_imgs.shape[0]

    # TODO: This permute is so unnatual
    src_imgs = src_imgs.permute(0, 2, 3, 1)

    K_src_inv = K_src.inverse()
    R_dst_inv = R_dst.inverse()

    # convert nerf space to colmap space
    M = torch.eye(3).to(K_src.device)
    M[1, 1] = -1.0
    M[2, 2] = -1.0

    x = np.arange(src_imgs[0].shape[1])
    y = np.arange(src_imgs[0].shape[0])
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)
    coord = coord.astype(np.float32)
    coord = torch.as_tensor(coord, dtype=K_src.dtype, device=K_src.device)
    coord = coord[None, Ellipsis, None].repeat(batch_size, 1, 1, 1, 1)

    depth = src_depths[:, :, :, None, None]

    # from reference to target viewpoint
    pts_3d_ref = depth * K_src_inv[:, None, None, Ellipsis] @ coord
    pts_3d_ref = M[None, None, None, Ellipsis] @ pts_3d_ref
    pts_3d_tgt = R_dst_inv[:, None, None, Ellipsis] @ R_src[
        :, None, None, Ellipsis
    ] @ pts_3d_ref + R_dst_inv[:, None, None, Ellipsis] @ (
        t_src[:, None, None, :, None] - t_dst[:, None, None, :, None]
    )
    pts_3d_tgt = M[None, None, None, Ellipsis] @ pts_3d_tgt
    points = K_dst[:, None, None, Ellipsis] @ pts_3d_tgt
    points = points.squeeze(-1)

    new_z = points[:, :, :, [2]].clone().permute(0, 3, 1, 2)  # b,1,h,w
    points = points / torch.clamp(points[:, :, :, [2]], 1e-8, None)

    src_ims_ = src_imgs.permute(0, 3, 1, 2)
    num_channels = src_ims_.shape[1]

    flow = points - coord.squeeze(-1)
    flow = flow.permute(0, 3, 1, 2)[:, :2, Ellipsis]

    importance = 1.0 / (new_z)
    importance_min = importance.amin((1, 2, 3), keepdim=True)
    importance_max = importance.amax((1, 2, 3), keepdim=True)
    weights = (importance - importance_min) / (
        importance_max - importance_min + 1e-6
    ) * 20 - 10
    src_mask_ = torch.ones_like(new_z)

    input_data = torch.cat([src_ims_, (1.0 / (new_z)), src_mask_], 1)

    output_data = splatting_function("softmax", input_data, flow, weights.detach())

    warp_feature = output_data[:, 0:num_channels, Ellipsis]
    warp_disp = output_data[:, num_channels : num_channels + 1, Ellipsis]
    warp_mask = output_data[:, num_channels + 1 : num_channels + 2, Ellipsis]

    return warp_feature, warp_disp, warp_mask


def quaterion_to_rotation(q):
    w, x, y, z = q
    R = torch.tensor(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )
    return R


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def read_pfm(filename: str) -> Tuple[np.ndarray, float]:
    """Read a depth map from a .pfm file

    Args:
        filename: .pfm file path string

    Returns:
        data: array of shape (H, W, C) representing loaded depth map
        scale: float to recover actual depth map pixel values
    """
    file = open(filename, "rb")  # treat as binary and read-only

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf":  # depth is Pf
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width, 1)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale
