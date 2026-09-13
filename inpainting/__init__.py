import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inpainting.fmm   import fmm_inpaint
from inpainting.ns    import ns_inpaint
from inpainting.patch import patch_inpaint

__all__ = ["fmm_inpaint", "ns_inpaint", "patch_inpaint"]