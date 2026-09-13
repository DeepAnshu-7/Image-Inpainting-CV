
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import (binary_dilation, binary_closing,
                            median_filter, label as ndlabel,
                            distance_transform_edt)
from skimage.metrics import structural_similarity as ssim_fn

from inpainting import fmm_inpaint, ns_inpaint
from tools      import MaskDrawer


# ═════════════════════════════════════════════════════════════
#  MASK DETECTION
# ═════════════════════════════════════════════════════════════

def _clean(raw, min_comp=20, dilate=5):
    labeled, _ = ndlabel(raw)
    sizes = np.bincount(labeled.ravel())
    keep  = np.where(sizes >= min_comp)[0]; keep = keep[keep > 0]
    if len(keep) == 0: return raw
    clean = np.isin(labeled, keep)
    return binary_dilation(clean, iterations=dilate) if dilate > 0 else clean

def detect_smart(img, sensitivity=70, dilate=5):
    med = np.zeros_like(img, dtype=np.float64)
    for c in range(3):
        med[:,:,c] = median_filter(img[:,:,c].astype(float), size=31)
    dev = np.abs(img.astype(float) - med).max(axis=2)
    return _clean(dev > sensitivity, dilate=dilate)

def detect_dark(img, threshold=60, dilate=4):
    r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
    raw = (r.astype(int)<threshold)&(g.astype(int)<threshold)&(b.astype(int)<threshold)
    return _clean(raw, dilate=dilate)

def detect_light(img, threshold=220, dilate=4):
    r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
    raw = (r.astype(int)>threshold)&(g.astype(int)>threshold)&(b.astype(int)>threshold)
    return _clean(raw, dilate=dilate)

def detect_red(img, min_r=130, gap=60, dilate=5):
    r=img[:,:,0].astype(int); g=img[:,:,1].astype(int); b=img[:,:,2].astype(int)
    return _clean((r>min_r)&(r-g>gap)&(r-b>gap), dilate=dilate)

def detect_color(img, target=(0,0,0), tolerance=60, dilate=4):
    diff=img.astype(float)-np.array(target,dtype=float)
    dist=np.sqrt((diff**2).sum(axis=2))
    return _clean(dist<tolerance, dilate=dilate)


# ═════════════════════════════════════════════════════════════
#  MAKE "MARKED" IMAGE (original with red mask overlay)
# ═════════════════════════════════════════════════════════════

def make_marked_image(image_np, mask):
    """Show original image with red overlay where the mask is."""
    marked = image_np.copy()
    # Blend red over masked pixels  (70% red, 30% original)
    marked[mask, 0] = np.clip(0.3*image_np[mask,0] + 0.7*220, 0, 255).astype(np.uint8)
    marked[mask, 1] = np.clip(0.3*image_np[mask,1] + 0.7*0,   0, 255).astype(np.uint8)
    marked[mask, 2] = np.clip(0.3*image_np[mask,2] + 0.7*0,   0, 255).astype(np.uint8)
    return marked


# ═════════════════════════════════════════════════════════════
#  EVALUATION METRICS  (no ground truth needed)
# ═════════════════════════════════════════════════════════════

def evaluate(original, restored, mask):
    
    H, W = mask.shape
    border_out = binary_dilation(mask, iterations=4) & ~mask   # known pixels just outside
    border_in  = binary_dilation(~mask, iterations=4) & mask   # filled pixels just inside

    # ── 1. Border MAE ─────────────────────────────────────────
    if border_out.sum() > 0 and border_in.sum() > 0:
        mean_out = float(np.mean(restored[border_out].astype(float)))
        mean_in  = float(np.mean(restored[border_in].astype(float)))
        border_mae = abs(mean_in - mean_out)
    else:
        border_mae = float('nan')

    # ── 2. Gradient Consistency ───────────────────────────────
    # Compute image gradient magnitude, compare at boundary
    gray = restored.mean(axis=2).astype(float)
    gy   = np.zeros_like(gray); gx = np.zeros_like(gray)
    gy[1:-1,:]  = (gray[2:,:] - gray[:-2,:]) / 2.0
    gx[:,1:-1]  = (gray[:,2:] - gray[:,:-2]) / 2.0
    gmag = np.sqrt(gy*gy + gx*gx)

    boundary = binary_dilation(mask, iterations=2) & ~mask
    if boundary.sum() > 0:
        grad_consistency = float(np.mean(gmag[boundary]))
    else:
        grad_consistency = float('nan')

    # ── 3. Texture SSIM ───────────────────────────────────────
    # Take filled region and compare SSIM with surrounding known region
    # Use grayscale for SSIM
    gray_orig  = original.mean(axis=2).astype(np.float32) / 255.0
    gray_rest  = restored.mean(axis=2).astype(np.float32) / 255.0

    # Bounding box of mask
    ys, xs = np.where(mask)
    if len(ys) > 0:
        y0,y1 = max(0,ys.min()-10), min(H,ys.max()+10)
        x0,x1 = max(0,xs.min()-10), min(W,xs.max()+10)
        patch_h = y1-y0; patch_w = x1-x0
        if patch_h >= 7 and patch_w >= 7:
            win = min(7, patch_h if patch_h%2==1 else patch_h-1,
                         patch_w if patch_w%2==1 else patch_w-1)
            try:
                texture_ssim = float(ssim_fn(
                    gray_orig[y0:y1, x0:x1],
                    gray_rest[y0:y1, x0:x1],
                    win_size=win, data_range=1.0
                ))
            except Exception:
                texture_ssim = float('nan')
        else:
            texture_ssim = float('nan')
    else:
        texture_ssim = float('nan')

    return {
        "border_mae":        border_mae,
        "grad_consistency":  grad_consistency,
        "texture_ssim":      texture_ssim,
    }


def print_metrics(name, m):
    print(f"\n  ── {name} Metrics ──────────────────────────────")
    print(f"     Border MAE          : {m['border_mae']:.3f}   (lower = smoother seam)")
    print(f"     Gradient Consistency: {m['grad_consistency']:.3f}   (lower = less visible edge)")
    print(f"     Texture SSIM        : {m['texture_ssim']:.4f}  (higher = better texture match)")


# ═════════════════════════════════════════════════════════════
#  VISUALISATION  — 5 panels
# ═════════════════════════════════════════════════════════════

def _save(image_np, mask, results, metrics, out_dir, label):
    """
    Display and save a 5-panel comparison:
      Original | Marked (red overlay) | Mask | FMM | NS
    Plus a metrics bar chart below.
    """
    n   = len(results)   # 1 or 2
    fig = plt.figure(figsize=(5*(3+n), 9), layout="constrained")
    fig.suptitle(f"Damage Removal — {label}", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(2, 3+n, figure=fig,
                           height_ratios=[3, 1])

    marked = make_marked_image(image_np, mask)

    panels = [
        (image_np, "Original"),
        (marked,   "Marked (your strokes)"),
        ((mask*255).astype(np.uint8), "Mask"),
    ]
    for name, img in results.items():
        m = metrics.get(name, {})
        title = (f"{name.upper()}\n"
                 f"MAE={m.get('border_mae',float('nan')):.2f}  "
                 f"Grad={m.get('grad_consistency',float('nan')):.2f}  "
                 f"SSIM={m.get('texture_ssim',float('nan')):.3f}")
        panels.append((img, title))

    for col, (img, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # ── Metrics bar chart ──────────────────────────────────────
    if len(results) >= 1:
        ax_bar = fig.add_subplot(gs[1, :])
        names   = list(results.keys())
        mae_v   = [metrics[n]["border_mae"]       for n in names]
        grad_v  = [metrics[n]["grad_consistency"]  for n in names]
        ssim_v  = [metrics[n]["texture_ssim"]      for n in names]

        x    = np.arange(len(names))
        w    = 0.25
        bars = ax_bar.bar(x-w,   mae_v,  w, label="Border MAE (↓)",       color="#4c9be8", alpha=0.85)
        bars2= ax_bar.bar(x,     grad_v, w, label="Grad Consistency (↓)",  color="#e87c4c", alpha=0.85)
        bars3= ax_bar.bar(x+w,   ssim_v, w, label="Texture SSIM (↑)",     color="#5cb85c", alpha=0.85)

        for bar in list(bars)+list(bars2)+list(bars3):
            h = bar.get_height()
            ax_bar.text(bar.get_x()+bar.get_width()/2, h+0.002,
                        f"{h:.3f}", ha='center', va='bottom', fontsize=8)

        ax_bar.set_xticks(x); ax_bar.set_xticklabels([n.upper() for n in names])
        ax_bar.set_title("Evaluation Metrics Comparison", fontsize=10)
        ax_bar.legend(fontsize=8); ax_bar.set_ylim(bottom=0)

    comp = os.path.join(out_dir, f"comparison_{label}.png")
    plt.savefig(comp, dpi=150, bbox_inches="tight")
    print(f"\n  Comparison  → {comp}")
    plt.show()

    # Save individual result images
    Image.fromarray(marked).save(os.path.join(out_dir, f"marked_{label}.png"))
    for name, r in results.items():
        p = os.path.join(out_dir, f"cleaned_{name}_{label}.png")
        Image.fromarray(r).save(p)
        print(f"  {name.upper()} result  → {p}")
    Image.fromarray((mask*255).astype(np.uint8)).save(
        os.path.join(out_dir, f"mask_{label}.png"))

    # Save metrics text
    txt = os.path.join(out_dir, f"metrics_{label}.txt")
    with open(txt, "w") as f:
        f.write(f"Damage Removal Metrics — {label}\n")
        f.write(f"Mask pixels: {mask.sum()} ({100*mask.sum()/mask.size:.2f}%)\n\n")
        f.write(f"{'Metric':<25} {'FMM':>10} {'NS':>10}  Notes\n")
        f.write("-"*60+"\n")
        for key, note in [
            ("border_mae",       "lower is better"),
            ("grad_consistency",  "lower is better"),
            ("texture_ssim",      "higher is better"),
        ]:
            vals = {n: metrics[n].get(key, float('nan')) for n in results}
            row  = f"{key:<25}"
            for n in results:
                row += f" {vals[n]:>10.4f}"
            f.write(row + f"  {note}\n")
    print(f"  Metrics txt → {txt}")


# ═════════════════════════════════════════════════════════════
#  CORE PIPELINE
# ═════════════════════════════════════════════════════════════

def remove(
    image_path:  str,
    mode:        str   = "manual",
    method:      str   = "both",
    sensitivity: int   = 70,
    threshold:   int   = 60,
    target_rgb:  tuple = (0, 0, 0),
    tolerance:   int   = 60,
    dilate:      int   = 5,
    out_dir:     str   = "results_cleaned",
):
    os.makedirs(out_dir, exist_ok=True)

    image_np = np.array(Image.open(image_path).convert("RGB"))
    H, W     = image_np.shape[:2]
    tag      = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\n{'═'*55}")
    print(f"  DAMAGE REMOVAL  v5")
    print(f"  Image  : {os.path.basename(image_path)}  ({W}×{H})")
    print(f"  Mode   : {mode}    Method: {method}")
    print(f"{'═'*55}\n")

    # ── Build mask ────────────────────────────────────────────
    if mode == "manual":
        print("Opening mask painter …")
        print("  Tip: 2px brush, trace over damage. Press f to fill gaps.\n")
        mask = MaskDrawer(image_np).get_mask()
    elif mode == "smart":
        mask = detect_smart(image_np, sensitivity=sensitivity, dilate=dilate)
    elif mode == "dark":
        mask = detect_dark(image_np, threshold=threshold, dilate=dilate)
    elif mode == "light":
        mask = detect_light(image_np, threshold=threshold, dilate=dilate)
    elif mode == "red":
        mask = detect_red(image_np, dilate=dilate)
    elif mode == "color":
        mask = detect_color(image_np, target=target_rgb,
                            tolerance=tolerance, dilate=dilate)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Gap-close
    mask = binary_closing(mask, structure=np.ones((7,7), dtype=bool))
    mask = binary_dilation(mask, iterations=2).astype(bool)

    pct = 100.0 * mask.sum() / (H * W)
    print(f"  Masked : {mask.sum()} px ({pct:.2f}%)\n")

    if mask.sum() == 0:
        print("[!] Mask is empty."); return {}

    # ── Inpaint ───────────────────────────────────────────────
    results = {}
    metrics  = {}

    if method in ("fmm", "both", "all"):
        print("── FMM ──")
        t0 = time.perf_counter()
        results["fmm"] = fmm_inpaint(image_np, mask)
        t = time.perf_counter()-t0
        metrics["fmm"] = evaluate(image_np, results["fmm"], mask)
        metrics["fmm"]["time"] = t
        print_metrics("FMM", metrics["fmm"])
        print(f"     Time: {t:.1f}s")

    if method in ("ns", "both", "all"):
        print("\n── NS ──")
        t0 = time.perf_counter()
        results["ns"] = ns_inpaint(image_np, mask)
        t = time.perf_counter()-t0
        metrics["ns"] = evaluate(image_np, results["ns"], mask)
        metrics["ns"]["time"] = t
        print_metrics("NS", metrics["ns"])
        print(f"     Time: {t:.1f}s")

    _save(image_np, mask, results, metrics, out_dir, tag)
    print(f"\nDone. Results saved → {out_dir}/")
    return results


# ═════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Remove marker/pen from photos with evaluation metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python remove_damage.py --image photo.jpg
  python remove_damage.py --image photo.jpg --auto smart
  python remove_damage.py --image photo.jpg --auto dark --threshold 50
  python remove_damage.py --image dog.jpg   --auto red
  python remove_damage.py --image photo.jpg --method both --out results/
""",
    )
    p.add_argument("--image",       "-i", required=True)
    p.add_argument("--auto",        "-a", default=None,
                   choices=["smart","dark","light","red","color"])
    p.add_argument("--sensitivity", "-s", type=int, default=70)
    p.add_argument("--threshold",   "-t", type=int, default=60)
    p.add_argument("--color",       "-c", type=int, nargs=3,
                   metavar=("R","G","B"), default=[0,0,0])
    p.add_argument("--tolerance",   "-T", type=int, default=60)
    p.add_argument("--dilate",      "-d", type=int, default=5)
    p.add_argument("--method",      "-m", default="both",
                   choices=["fmm","ns","both","all"])
    p.add_argument("--out",         "-o", default="results_cleaned")
    args = p.parse_args()

    if not os.path.isfile(args.image):
        print(f"[ERROR] Not found: {args.image}"); sys.exit(1)

    remove(
        image_path  = args.image,
        mode        = args.auto if args.auto else "manual",
        method      = args.method,
        sensitivity = args.sensitivity,
        threshold   = args.threshold,
        target_rgb  = tuple(args.color),
        tolerance   = args.tolerance,
        dilate      = args.dilate,
        out_dir     = args.out,
    )

if __name__ == "__main__":
    main()