# # # # ─────────────────────────────────────────────────────────────
# # # #  inpainting/ns.py
# # # #  Navier-Stokes / Laplacian diffusion inpainting  (Bertalmio 2000)
# # # #  Pure Python + NumPy — no OpenCV
# # # # ─────────────────────────────────────────────────────────────
# # # #
# # # #  Algorithm outline
# # # #  -----------------
# # # #  The key idea: image smoothness (measured by the Laplacian) should
# # # #  be propagated *along* isophotes (lines of constant intensity) into
# # # #  the damaged region.
# # # #
# # # #  Simplified implementation
# # # #  -------------------------
# # # #  1. Initialise the masked region with the mean of known pixels.
# # # #  2. Iterate:
# # # #       a. Compute the Laplacian ∇²u over the whole image.
# # # #       b. Update only masked pixels:  u ← u + dt·∇²u
# # # #       c. Re-anchor known pixels so they never change.
# # # #  3. Return the result clamped to [0, 255].
# # # #
# # # #  This converges to the harmonic extension of the boundary values —
# # # #  equivalent to solving Laplace's equation inside the mask.
# # # # ─────────────────────────────────────────────────────────────

# # # import numpy as np

# # # from config import NS_ITERATIONS, NS_DT


# # # def ns_inpaint(
# # #     image_np:  np.ndarray,
# # #     mask:      np.ndarray,
# # #     num_iters: int   = NS_ITERATIONS,
# # #     dt:        float = NS_DT,
# # # ) -> np.ndarray:
# # #     """
# # #     Inpaint damaged pixels using Laplacian diffusion (Bertalmio 2000 style).

# # #     Parameters
# # #     ----------
# # #     image_np  : H×W×3 uint8 ndarray  (damaged image)
# # #     mask      : H×W   bool  ndarray  (True = pixel to fill)
# # #     num_iters : number of diffusion iterations
# # #     dt        : integration time-step  (must be < 0.25 for stability)

# # #     Returns
# # #     -------
# # #     H×W×3 uint8 ndarray  (inpainted image)
# # #     """
# # #     u     = image_np.astype(np.float64) / 255.0
# # #     H, W  = mask.shape
# # #     C     = u.shape[2]
# # #     fixed = ~mask          # pixels that stay fixed throughout

# # #     # ── Initialise masked region ──────────────────────────────
# # #     for c in range(C):
# # #         known_mean      = float(np.mean(u[:, :, c][fixed]))
# # #         u[:, :, c][mask] = known_mean

# # #     # ── Diffusion loop ────────────────────────────────────────
# # #     print(f"  [NS ] Diffusing for {num_iters} iterations …")
# # #     for it in range(num_iters):
# # #         # 2-D Laplacian (central differences, interior only)
# # #         lap = np.zeros_like(u)
# # #         lap[1:-1, 1:-1] = (
# # #             u[:-2, 1:-1] + u[2:, 1:-1] +
# # #             u[1:-1, :-2] + u[1:-1, 2:] -
# # #             4.0 * u[1:-1, 1:-1]
# # #         )

# # #         # Advance only masked pixels
# # #         u[mask] += dt * lap[mask]
# # #         u = np.clip(u, 0.0, 1.0)

# # #         # Re-anchor known pixels (numerical drift prevention)
# # #         for c in range(C):
# # #             u[:, :, c][fixed] = image_np[:, :, c][fixed] / 255.0

# # #         if (it + 1) % 100 == 0:
# # #             print(f"        iter {it + 1:>4d}/{num_iters}")

# # #     return (u * 255.0).astype(np.uint8)




# # # ─────────────────────────────────────────────────────────────
# # #  inpainting/ns.py
# # #  Navier-Stokes / Laplacian diffusion inpainting  (Bertalmio 2000)
# # #  Pure Python + NumPy — no OpenCV
# # #
# # #  v2 improvements
# # #  ─────────────────────────────────────────────────────────────
# # #  • Better initialisation: masked region seeded with bilinear
# # #    interpolation from the boundary pixels (not just global mean).
# # #    This gives diffusion a better starting point and converges faster.
# # #  • Iteration count auto-scales with mask size if num_iters=0.
# # # ─────────────────────────────────────────────────────────────

# # import numpy as np
# # from scipy.ndimage import distance_transform_edt

# # from config import NS_ITERATIONS, NS_DT


# # def _boundary_init(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
# #     """
# #     Initialise masked pixels using distance-weighted average of
# #     boundary pixels.  Much better starting point than global mean.
# #     """
# #     u = image.copy().astype(np.float64)
# #     H, W, C = u.shape

# #     # Distance of every masked pixel to the nearest known pixel
# #     dist = distance_transform_edt(mask)   # 0 outside mask, +ve inside

# #     # For each masked pixel, blend the nearest known pixel values
# #     # using a Gaussian-weighted average over boundary pixels
# #     for c in range(C):
# #         channel = u[:, :, c]
# #         # Simple seed: nearest-known-pixel via distance transform indices
# #         # scipy gives nearest-pixel coordinates
# #         _, idx = distance_transform_edt(mask, return_indices=True)
# #         # idx[0] = y-coords of nearest known pixel, idx[1] = x-coords
# #         iy, ix = idx
# #         # Fill masked region with nearest known pixel value
# #         channel[mask] = channel[iy[mask], ix[mask]]
# #         u[:, :, c] = channel

# #     return u


# # def ns_inpaint(
# #     image_np:  np.ndarray,
# #     mask:      np.ndarray,
# #     num_iters: int   = NS_ITERATIONS,
# #     dt:        float = NS_DT,
# # ) -> np.ndarray:
# #     """
# #     Inpaint using Laplacian diffusion (Bertalmio 2000 style).

# #     Parameters
# #     ----------
# #     image_np  : H×W×3 uint8 ndarray
# #     mask      : H×W   bool  ndarray  (True = fill)
# #     num_iters : diffusion iterations  (0 = auto-scale with mask size)
# #     dt        : time-step  (keep < 0.25)

# #     Returns
# #     -------
# #     H×W×3 uint8 ndarray
# #     """
# #     # Auto-scale iterations with mask area
# #     if num_iters == 0:
# #         pct       = 100.0 * mask.sum() / mask.size
# #         num_iters = max(200, min(800, int(pct * 40)))

# #     pct = 100.0 * mask.sum() / mask.size
# #     print(f"  [NS ] Diffusing {num_iters} iters (mask={pct:.1f}%) …")

# #     # Initialise with boundary-nearest-pixel (better than global mean)
# #     u     = _boundary_init(image_np, mask) / 255.0
# #     H, W  = mask.shape
# #     fixed = ~mask

# #     for it in range(num_iters):
# #         lap = np.zeros_like(u)
# #         lap[1:-1, 1:-1] = (
# #             u[:-2, 1:-1] + u[2:, 1:-1] +
# #             u[1:-1, :-2] + u[1:-1, 2:] -
# #             4.0 * u[1:-1, 1:-1]
# #         )

# #         u[mask] += dt * lap[mask]
# #         u = np.clip(u, 0.0, 1.0)

# #         # Re-anchor known pixels
# #         for c in range(u.shape[2]):
# #             u[:, :, c][fixed] = image_np[:, :, c][fixed] / 255.0

# #         if (it + 1) % 200 == 0:
# #             print(f"        iter {it+1:>4d}/{num_iters}")

# #     return (u * 255.0).astype(np.uint8)



# # ─────────────────────────────────────────────────────────────
# #  inpainting/ns.py  v5  —  stable Laplacian diffusion
# #
# #  The anisotropic (isophote-advection) approach was numerically
# #  unstable — large gradient values in the advection term caused
# #  oscillations that exploded into noise.
# #
# #  This version uses the proven stable approach:
# #    • Seed masked region with nearest-boundary pixel (not mean)
# #    • Iterate: u[mask] += dt * Laplacian(u)[mask]
# #    • Re-anchor known pixels every iteration
# #    • dt=0.1 is always stable for this stencil
# # ─────────────────────────────────────────────────────────────

# import numpy as np
# from scipy.ndimage import distance_transform_edt
# from config import NS_DT


# def ns_inpaint(image_np, mask, num_iters=0, dt=NS_DT):
#     """
#     Laplacian diffusion inpainting (stable, no noise).

#     Parameters
#     ----------
#     image_np  : H×W×3 uint8 ndarray
#     mask      : H×W   bool  ndarray  (True = fill)
#     num_iters : iterations  (0 = auto)
#     dt        : time-step   (0.1 is always stable)

#     Returns
#     -------
#     H×W×3 uint8 ndarray
#     """
#     pct = 100.0 * mask.sum() / mask.size
#     if num_iters == 0:
#         num_iters = max(200, min(600, int(pct * 50)))

#     print(f"  [NS ] Laplacian diffusion {num_iters} iters (mask={pct:.1f}%) …")

#     # ── Seed: each masked pixel ← nearest known pixel ─────────
#     u = image_np.astype(np.float64) / 255.0
#     _, idx = distance_transform_edt(mask, return_indices=True)
#     iy, ix = idx
#     for c in range(3):
#         ch = u[:, :, c]
#         ch[mask] = ch[iy[mask], ix[mask]]
#         u[:, :, c] = ch

#     fixed = ~mask

#     for it in range(num_iters):
#         # 5-point Laplacian (interior only to avoid border effects)
#         lap = np.zeros_like(u)
#         lap[1:-1, 1:-1] = (
#             u[:-2, 1:-1] + u[2:, 1:-1] +
#             u[1:-1, :-2] + u[1:-1, 2:] -
#             4.0 * u[1:-1, 1:-1]
#         )
#         u[mask] += dt * lap[mask]
#         np.clip(u, 0.0, 1.0, out=u)

#         # Re-anchor known pixels — must not drift
#         for c in range(3):
#             u[:, :, c][fixed] = image_np[:, :, c][fixed] / 255.0

#         if (it + 1) % 200 == 0:
#             print(f"        iter {it+1:>4d}/{num_iters}")

#     return (u * 255.0).astype(np.uint8)




# ─────────────────────────────────────────────────────────────
#  inpainting/ns.py  v5  —  stable Laplacian diffusion
#
#  The anisotropic (isophote-advection) approach was numerically
#  unstable — large gradient values in the advection term caused
#  oscillations that exploded into noise.
#
#  This version uses the proven stable approach:
#    • Seed masked region with nearest-boundary pixel (not mean)
#    • Iterate: u[mask] += dt * Laplacian(u)[mask]
#    • Re-anchor known pixels every iteration
#    • dt=0.1 is always stable for this stencil
# ─────────────────────────────────────────────────────────────

import numpy as np
from scipy.ndimage import distance_transform_edt
from config import NS_DT


def ns_inpaint(image_np, mask, num_iters=0, dt=NS_DT):
    """
    Laplacian diffusion inpainting (stable, no noise).

    Parameters
    ----------
    image_np  : H×W×3 uint8 ndarray
    mask      : H×W   bool  ndarray  (True = fill)
    num_iters : iterations  (0 = auto)
    dt        : time-step   (0.1 is always stable)

    Returns
    -------
    H×W×3 uint8 ndarray
    """
    pct = 100.0 * mask.sum() / mask.size
    if num_iters == 0:
        num_iters = max(200, min(600, int(pct * 50)))

    print(f"  [NS ] Laplacian diffusion {num_iters} iters (mask={pct:.1f}%) …")

    # ── Seed: each masked pixel ← nearest known pixel ─────────
    u = image_np.astype(np.float64) / 255.0
    _, idx = distance_transform_edt(mask, return_indices=True)
    iy, ix = idx
    for c in range(3):
        ch = u[:, :, c]
        ch[mask] = ch[iy[mask], ix[mask]]
        u[:, :, c] = ch

    fixed = ~mask

    for it in range(num_iters):
        # 5-point Laplacian (interior only to avoid border effects)
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            u[:-2, 1:-1] + u[2:, 1:-1] +
            u[1:-1, :-2] + u[1:-1, 2:] -
            4.0 * u[1:-1, 1:-1]
        )
        u[mask] += dt * lap[mask]
        np.clip(u, 0.0, 1.0, out=u)

        # Re-anchor known pixels — must not drift
        for c in range(3):
            u[:, :, c][fixed] = image_np[:, :, c][fixed] / 255.0

        if (it + 1) % 200 == 0:
            print(f"        iter {it+1:>4d}/{num_iters}")

    return (u * 255.0).astype(np.uint8)