from math import isqrt

import torch
from e3nn.o3 import matrix_to_angles, wigner_D
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def rotate_sh(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype

    *_, n = sh_coefficients.shape
    # e3nn's matrix_to_angles asserts det(R) == 1; under bf16/fp16 autocast the
    # rotation matrices lose enough precision that the assertion fires (even
    # after casting back to fp32, the values are already perturbed). Force
    # this block to fp32 with autocast disabled AND re-orthonormalize R via
    # SVD so it's exactly a rotation. Same numerical result as the original
    # fp32 path when the input was already a clean rotation.
    with torch.amp.autocast(device_type="cuda", enabled=False):
        rotations_fp32 = rotations.float()
        # Project to nearest rotation: R = U V^T with sign of det(U V^T) folded
        # into the last column of V to guarantee det == +1.
        U, _, Vh = torch.linalg.svd(rotations_fp32)
        det = torch.det(U @ Vh)
        sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
        Vh = Vh.clone()
        Vh[..., -1, :] = Vh[..., -1, :] * sign.unsqueeze(-1)
        rotations_fp32 = U @ Vh
        alpha, beta, gamma = matrix_to_angles(rotations_fp32)
        result = []
        for degree in range(isqrt(n)):
            with torch.device(device):
                sh_rotations = wigner_D(degree, alpha, beta, gamma).type(dtype)
            sh_rotated = einsum(
                sh_rotations,
                sh_coefficients[..., degree**2 : (degree + 1) ** 2],
                "... i j, ... j -> ... i",
            )
            result.append(sh_rotated)

    return torch.cat(result, dim=-1)


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    from e3nn.o3 import spherical_harmonics
    from matplotlib import cm
    from scipy.spatial.transform.rotation import Rotation as R

    device = torch.device("cuda")

    # Generate random spherical harmonics coefficients.
    degree = 4
    coefficients = torch.rand((degree + 1) ** 2, dtype=torch.float32, device=device)

    def plot_sh(sh_coefficients, path: Path) -> None:
        phi = torch.linspace(0, torch.pi, 100, device=device)
        theta = torch.linspace(0, 2 * torch.pi, 100, device=device)
        phi, theta = torch.meshgrid(phi, theta, indexing="xy")
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        xyz = torch.stack([x, y, z], dim=-1)
        sh = spherical_harmonics(list(range(degree + 1)), xyz, True)
        result = einsum(sh, sh_coefficients, "... n, n -> ...")
        result = (result - result.min()) / (result.max() - result.min())

        # Set the aspect ratio to 1 so our sphere looks spherical
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            x.cpu().numpy(),
            y.cpu().numpy(),
            z.cpu().numpy(),
            rstride=1,
            cstride=1,
            facecolors=cm.seismic(result.cpu().numpy()),
        )
        # Turn off the axis planes
        ax.set_axis_off()
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(path)

    for i, angle in enumerate(torch.linspace(0, 2 * torch.pi, 30)):
        rotation = torch.tensor(
            R.from_euler("x", angle.item()).as_matrix(), device=device
        )
        plot_sh(rotate_sh(coefficients, rotation), Path(f"sh_rotation/{i:0>3}.png"))

    print("Done!")
