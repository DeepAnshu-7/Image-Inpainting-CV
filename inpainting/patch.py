# ─────────────────────────────────────────────────────────────
#  inpainting/patch.py  v2
#  Exemplar-based patch inpainting
#
#  Why FMM/NS look blurry on photos
#  ─────────────────────────────────────────────────────────────
#  FMM and NS both compute WEIGHTED AVERAGES of surrounding
#  pixels.  Averaging destroys high-frequency texture (fur
#  stripes, fabric weaves, grass) because it mixes dark and
#  light pixels into a medium-grey blur.
#
#  This method instead COPIES small pixel blocks (patches) from
#  the undamaged part of the image, finding the block whose
#  visible neighbourhood best matches the target.  Real pixels
#  → real texture → no blur.
#
#  Algorithm
#  ─────────────────────────────────────────────────────────────
#  1. Build a library of patches sampled from undamaged pixels
#     in the ORIGINAL image (not an FMM-blurred copy).
#  2. Initialise the masked region with the nearest boundary
#     pixel (fast, avoids the "black hole" problem).
#  3. Process masked pixels in boundary-first order so that
#     already-filled neighbours contribute context.
#  4. For each pixel: randomly sample N_CAND candidates from
#     the library, compute SSD only over the KNOWN pixels in
#     the target patch, copy the centre colour of the winner.
# ─────────────────────────────────────────────────────────────

import numpy as np
from scipy.ndimage import distance_transform_edt


# ── tuneable defaults ─────────────────────────────────────────
PATCH_SIZE   = 9      # neighbourhood window (must be odd)
N_CANDIDATES = 500    # random candidates per pixel
MAX_SOURCE   = 25000  # max source patches to keep in library


def patch_inpaint(
    image_np:     np.ndarray,
    mask:         np.ndarray,
    patch_size:   int = PATCH_SIZE,
    n_candidates: int = N_CANDIDATES,
) -> np.ndarray:
    """
    Exemplar-based patch inpainting.

    Produces crisp, texture-preserving results on photos —
    especially better than FMM/NS on complex textures like fur,
    fabric, grass, or any region with repeating patterns.

    Parameters
    ----------
    image_np     : H×W×3 uint8 ndarray  (damaged image)
    mask         : H×W   bool  ndarray  (True = fill)
    patch_size   : patch window side length in pixels (default 9)
    n_candidates : candidate patches tested per pixel (default 500)

    Returns
    -------
    H×W×3 uint8 ndarray
    """
    H, W  = mask.shape
    C     = image_np.shape[2]
    HALF  = patch_size // 2
    flat  = patch_size * patch_size * C

    # ── 1. Build source library from ORIGINAL undamaged pixels ─
    print("  [PATCH] Building texture library from original image …")
    border      = HALF
    ky, kx      = np.where(~mask)
    valid       = ((ky >= border) & (ky < H - border) &
                   (kx >= border) & (kx < W - border))
    ky, kx      = ky[valid], kx[valid]

    if len(ky) == 0:
        print("  [PATCH] No valid source pixels — returning nearest-fill.")
        result = image_np.copy().astype(np.float64)
        _, idx = distance_transform_edt(mask, return_indices=True)
        iy, ix = idx
        for c in range(C):
            ch       = result[:, :, c]
            ch[mask] = ch[iy[mask], ix[mask]]
        return np.clip(result, 0, 255).astype(np.uint8)

    # Subsample library if too large
    N_src = min(MAX_SOURCE, len(ky))
    if N_src < len(ky):
        pick = np.random.choice(len(ky), N_src, replace=False)
        ky, kx = ky[pick], kx[pick]

    # Extract patches and centre colours from the ORIGINAL image
    # (key: use image_np, NOT a blurred copy)
    src_patches = np.empty((N_src, flat),  dtype=np.float32)
    src_colors  = np.empty((N_src, C),     dtype=np.float32)
    orig_f      = image_np.astype(np.float32)

    for i, (sy, sx) in enumerate(zip(ky, kx)):
        src_patches[i] = orig_f[sy - HALF: sy + HALF + 1,
                                 sx - HALF: sx + HALF + 1].ravel()
        src_colors[i]  = orig_f[sy, sx]

    print(f"  [PATCH] Library: {N_src} patches  "
          f"patch_size={patch_size}×{patch_size}  candidates={n_candidates}")

    # ── 2. Initialise masked region with nearest known pixel ───
    result    = image_np.astype(np.float64)
    _, idx    = distance_transform_edt(mask, return_indices=True)
    iy, ix    = idx
    for c in range(C):
        ch       = result[:, :, c]
        ch[mask] = ch[iy[mask], ix[mask]]
        result[:, :, c] = ch

    # ── 3. Process pixels boundary-first ──────────────────────
    dist       = distance_transform_edt(mask)
    ys, xs     = np.where(mask)
    order      = np.argsort(dist[ys, xs])   # nearest boundary first
    ys, xs     = ys[order], xs[order]
    total      = len(ys)
    step       = max(1, total // 10)
    work_mask  = mask.copy()
    n_cand     = min(n_candidates, N_src)

    print(f"  [PATCH] Filling {total} pixels …")

    for i, (y, x) in enumerate(zip(ys, xs)):
        if i % step == 0:
            print(f"          {i}/{total}  ({100 * i // total}%)")

        # Skip border pixels — nearest-fill from step 2 stays
        if y < HALF or y >= H - HALF or x < HALF or x >= W - HALF:
            work_mask[y, x] = False
            continue

        # Target patch and which pixels in it are already known
        t_patch    = result[y - HALF: y + HALF + 1,
                             x - HALF: x + HALF + 1]          # (ps,ps,3)
        t_known_2d = ~work_mask[y - HALF: y + HALF + 1,
                                 x - HALF: x + HALF + 1]      # (ps,ps) bool
        n_known    = int(t_known_2d.sum())

        if n_known == 0:
            work_mask[y, x] = False
            continue

        t_flat     = t_patch.ravel().astype(np.float32)       # (flat,)
        known_flat = np.repeat(t_known_2d.ravel(), C)         # (flat,) bool

        # Sample candidate patches from library
        samp  = np.random.choice(N_src, n_cand, replace=False)
        diff  = src_patches[samp] - t_flat                    # (n_cand, flat)
        diff[:, ~known_flat] = 0.0                            # ignore unknown
        ssd   = (diff * diff).sum(axis=1) / float(known_flat.sum())
        best  = samp[int(np.argmin(ssd))]

        result[y, x]     = src_colors[best]
        work_mask[y, x]  = False

    print("  [PATCH] Done.")
    return np.clip(result, 0, 255).astype(np.uint8)