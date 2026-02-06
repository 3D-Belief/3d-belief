from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes

def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
):
    # Shift the scene so that the median Gaussian is at the origin.
    means = means - means.median(dim=0).values

    # Rescale the scene so that most Gaussians are within range [-1, 1].
    scale_factor = means.abs().quantile(0.95, dim=0).max()
    means = means / scale_factor
    scales = scales / scale_factor

    # Define a rotation that makes +Z be the world up vector.
    rotation = [
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ]
    rotation = torch.tensor(rotation, dtype=torch.float32, device=means.device)

    # The Polycam viewer seems to start at a 45 degree angle. Since we want to be
    # looking directly at the object, we compose a 45 degree rotation onto the above
    # rotation.
    adjustment = torch.tensor(
        R.from_rotvec([0, 0, -45], True).as_matrix(),
        dtype=torch.float32,
        device=means.device,
    )
    rotation = adjustment @ rotation

    # We also want to see the scene in camera space (as the default view). We therefore
    # compose the w2c rotation onto the above rotation.
    rotation = rotation @ extrinsics[:3, :3].inverse()

    # Apply the rotation to the means (Gaussian positions).
    means = einsum(rotation, means, "i j, ... j -> ... i")

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = rotation.detach().cpu().numpy() @ rotations
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since our axes are swizzled for the spherical harmonics, we only export the DC
    # band.
    harmonics_view_invariant = harmonics[..., 0]

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        harmonics_view_invariant.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)

def export_gaussians_to_ply(
    gaussians,                        # instance of Gaussians class
    extrinsics: torch.Tensor,         # shape (B,4,4)
    output_paths: Path | list[Path],  # single Path if B==1, else list of length B
):
    """
    - gaussians.means:       (B, N, 3)
    - gaussians.covariances: (B, N, 3, 3)
    - gaussians.harmonics:   (B, N, 3, d_sh)
    - gaussians.opacities:   (B, N)
    - extrinsics:            (B, 4, 4) world-to-camera matrices
    - output_paths:          Path or list of Paths, length B
    """
    B, N, _ = gaussians.means.shape

    # normalize output_paths to a list of length B
    if isinstance(output_paths, (str, Path)):
        if B != 1:
            raise ValueError("Single output_path only allowed if batch size is 1")
        paths = [Path(output_paths)]
    else:
        paths = list(output_paths)
        if len(paths) != B:
            raise ValueError(f"Need {B} output paths, got {len(paths)}")

    for b in range(B):
        means_b       = gaussians.means[b]                 # (N,3)
        covs_b        = gaussians.covariances[b]           # (N,3,3)
        harmonics_b   = gaussians.harmonics[b]             # (N,3,d_sh)
        opacities_b   = gaussians.opacities[b]             # (N,)

        # decompose each covariance into (scale, quaternion)
        scales_b      = torch.zeros((N, 3), device=means_b.device)
        rotations_b   = torch.zeros((N, 4), device=means_b.device)
        for i in range(N):
            # eigen‐decompose
            evals, evecs = torch.linalg.eigh(covs_b[i])
            scales_b[i]  = torch.sqrt(torch.clamp(evals, min=0.0))
            # convert rotation matrix to quaternion (x,y,z,w)
            qm = R.from_matrix(evecs.cpu().numpy()).as_quat()
            rotations_b[i] = torch.tensor(qm, device=means_b.device)

        export_ply(
            extrinsics[b],    # (4,4)
            means_b,          # (N,3)
            scales_b,         # (N,3)
            rotations_b,      # (N,4)
            harmonics_b,      # (N,3,d_sh)
            opacities_b,      # (N,)
            paths[b],         # Path to write .ply
        )