
# ── FMM settings ─────────────────────────────────────────────
FMM_RADIUS = 5          
# ── NS / Diffusion settings ───────────────────────────────────
NS_ITERATIONS = 400     
NS_DT         = 0.1     

# ── Damage creation defaults ──────────────────────────────────
SCRATCH_COUNT     = 15
SCRATCH_MAX_WIDTH = 4
TEXT_WATERMARK    = "WATERMARK"
TEXT_FONT_SIZE    = 48
BLOCK_SIZE        = 60

# ── Metric targets (from project spec) ───────────────────────
PSNR_TARGET = 25.0      # dB
SSIM_TARGET = 0.60

# ── Output ────────────────────────────────────────────────────
DEFAULT_OUT_DIR = "results"

# ── Fallback font paths (tried in order) ─────────────────────
FONT_PATHS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/Arial.ttf",
]