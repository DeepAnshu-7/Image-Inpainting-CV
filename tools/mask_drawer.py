# # ─────────────────────────────────────────────────────────────
# #  tools/mask_drawer.py  v4
# #
# #  Key fix: stroke INTERPOLATION — positions are connected
# #  between mouse events so you draw solid lines, not blobs.
# #
# #  Controls
# #  Left-drag   paint mask (red)    Right-drag  erase mask
# #  +/=         bigger brush        -           smaller brush
# #  u           undo                r           reset
# #  f           fill gaps           close       done
# # ─────────────────────────────────────────────────────────────

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import binary_closing, binary_dilation


# class MaskDrawer:
#     DEFAULT_BRUSH = 2

#     def __init__(self, image_np: np.ndarray):
#         self.image      = image_np
#         self.mask       = np.zeros(image_np.shape[:2], dtype=bool)
#         self._history   = []
#         self.drawing    = False
#         self.erasing    = False
#         self.brush      = self.DEFAULT_BRUSH
#         self._last_pos  = None          # ← needed for interpolation

#         self.fig, (self.ax_ref, self.ax_draw) = plt.subplots(1, 2, figsize=(14, 6))
#         self.fig.patch.set_facecolor("#1a1a2e")
#         self.fig.suptitle(
#             "MASK PAINTER  ·  "
#             "LEFT=paint  RIGHT=erase  +/-=brush  "
#             "u=undo  r=reset  f=fill-gaps  close=done",
#             fontsize=9, color="#e0e0e0",
#         )
#         self.ax_ref.imshow(image_np)
#         self.ax_ref.set_title("Reference (unchanged)", fontsize=9, color="#aaaaaa")
#         self.ax_ref.axis("off")
#         self.ax_draw.axis("off")
#         self.ax_draw.set_facecolor("#1a1a2e")
#         self._render()

#         c = self.fig.canvas
#         c.mpl_connect("button_press_event",   self._on_press)
#         c.mpl_connect("button_release_event", self._on_release)
#         c.mpl_connect("motion_notify_event",  self._on_move)
#         c.mpl_connect("key_press_event",      self._on_key)

#     # ── Render ────────────────────────────────────────────────

#     def _render(self):
#         disp = self.image.copy()
#         disp[self.mask, 0] = 235
#         disp[self.mask, 1] = 25
#         disp[self.mask, 2] = 25
#         if hasattr(self, "im_draw"):
#             self.im_draw.set_data(disp)
#         else:
#             self.im_draw = self.ax_draw.imshow(disp)
#         pct = 100.0 * self.mask.sum() / max(1, self.mask.size)
#         self.ax_draw.set_title(
#             f"Brush: {self.brush}px  ·  Masked: {self.mask.sum()}px ({pct:.1f}%)  ·  f=fill-gaps",
#             fontsize=8, color="#cccccc",
#         )
#         self.fig.canvas.draw_idle()

#     # ── Core: paint a filled circle at one point ──────────────

#     def _stamp(self, cx: int, cy: int, paint: bool):
#         H, W   = self.mask.shape
#         r      = self.brush
#         ys, xs = np.ogrid[-r: r + 1, -r: r + 1]
#         disc   = xs * xs + ys * ys <= r * r
#         y0 = max(0, cy - r);  y1 = min(H, cy + r + 1)
#         x0 = max(0, cx - r);  x1 = min(W, cx + r + 1)
#         cy0 = y0 - (cy - r);  cy1 = y1 - (cy - r)
#         cx0 = x0 - (cx - r);  cx1 = x1 - (cx - r)
#         if paint:
#             self.mask[y0:y1, x0:x1] |= disc[cy0:cy1, cx0:cx1]
#         else:
#             self.mask[y0:y1, x0:x1] &= ~disc[cy0:cy1, cx0:cx1]

#     # ── Interpolated stroke: fills the gap between events ─────

#     def _stroke(self, event, paint: bool):
#         """
#         Paint/erase along the LINE from the last recorded position to
#         the current one.  This turns disconnected event-dots into a
#         solid, continuous brush stroke.
#         """
#         if event.inaxes != self.ax_draw or event.xdata is None:
#             return

#         cx, cy = int(event.xdata), int(event.ydata)

#         if self._last_pos is not None:
#             lx, ly = self._last_pos
#             # Number of steps = Chebyshev distance (covers diagonal moves)
#             steps = max(1, max(abs(cx - lx), abs(cy - ly)))
#             for i in range(steps + 1):
#                 t  = i / steps
#                 ix = int(round(lx + t * (cx - lx)))
#                 iy = int(round(ly + t * (cy - ly)))
#                 self._stamp(ix, iy, paint)
#         else:
#             self._stamp(cx, cy, paint)

#         self._last_pos = (cx, cy)
#         self._render()

#     # ── Events ────────────────────────────────────────────────

#     def _on_press(self, e):
#         self._save_history()
#         self._last_pos = None            # fresh stroke start
#         if e.button == 1:
#             self.drawing = True;  self._stroke(e, True)
#         elif e.button == 3:
#             self.erasing = True;  self._stroke(e, False)

#     def _on_release(self, e):
#         self.drawing  = False
#         self.erasing  = False
#         self._last_pos = None            # reset so next press starts clean

#     def _on_move(self, e):
#         if self.drawing:   self._stroke(e, True)
#         elif self.erasing: self._stroke(e, False)

#     def _on_key(self, e):
#         if   e.key in ("+", "="): self.brush = min(60, self.brush + 2)
#         elif e.key == "-":         self.brush = max(1,  self.brush - 2)
#         elif e.key == "u":         self._undo()
#         elif e.key == "r":
#             self._save_history()
#             self.mask[:] = False
#             self._last_pos = None
#             print("  Mask reset.")
#         elif e.key == "f":         self._fill_gaps()
#         self._render()

#     # ── History / undo ────────────────────────────────────────

#     def _save_history(self):
#         self._history.append(self.mask.copy())
#         if len(self._history) > 40:
#             self._history.pop(0)

#     def _undo(self):
#         if self._history:
#             self.mask = self._history.pop()
#             self._last_pos = None
#             print("  Undo.")
#         else:
#             print("  Nothing to undo.")

#     # ── Gap filler ────────────────────────────────────────────

#     def _fill_gaps(self):
#         self._save_history()
#         struct    = np.ones((9, 9), dtype=bool)
#         closed    = binary_closing(self.mask, structure=struct)
#         self.mask = binary_dilation(closed, iterations=2).astype(bool)
#         print(f"  Gaps filled → {self.mask.sum()} px")

#     # ── Public ────────────────────────────────────────────────

#     def get_mask(self) -> np.ndarray:
#         plt.show()
#         print(f"  Mask done: {self.mask.sum()} px ({100.0*self.mask.sum()/self.mask.size:.2f}%)")
#         return self.mask



# ─────────────────────────────────────────────────────────────
#  tools/mask_drawer.py  v5
#
#  Controls
#  Left-drag   paint (red)    Right-drag  erase
#  +/=         bigger brush   -           smaller
#  u           undo           r           reset
#  f           fill gaps      close       done
# ─────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_dilation


class MaskDrawer:
    DEFAULT_BRUSH = 1          # ← very small default

    def __init__(self, image_np: np.ndarray):
        self.image     = image_np
        self.mask      = np.zeros(image_np.shape[:2], dtype=bool)
        self._history  = []
        self.drawing   = False
        self.erasing   = False
        self.brush     = self.DEFAULT_BRUSH
        self._last_pos = None

        self.fig, (self.ax_ref, self.ax_draw) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.patch.set_facecolor("#1a1a2e")
        self.fig.suptitle(
            "MASK PAINTER  ·  "
            "LEFT=paint  RIGHT=erase  +/-=brush(2px default)  "
            "u=undo  r=reset  f=fill-gaps  close=done",
            fontsize=9, color="#e0e0e0",
        )
        self.ax_ref.imshow(image_np)
        self.ax_ref.set_title("Reference", fontsize=9, color="#aaaaaa")
        self.ax_ref.axis("off")
        self.ax_draw.axis("off")
        self.ax_draw.set_facecolor("#1a1a2e")
        self._render()

        c = self.fig.canvas
        c.mpl_connect("button_press_event",   self._on_press)
        c.mpl_connect("button_release_event", self._on_release)
        c.mpl_connect("motion_notify_event",  self._on_move)
        c.mpl_connect("key_press_event",      self._on_key)

    def _render(self):
        disp = self.image.copy()
        disp[self.mask, 0] = 235
        disp[self.mask, 1] = 25
        disp[self.mask, 2] = 25
        if hasattr(self, "im_draw"):
            self.im_draw.set_data(disp)
        else:
            self.im_draw = self.ax_draw.imshow(disp)
        pct = 100.0 * self.mask.sum() / max(1, self.mask.size)
        self.ax_draw.set_title(
            f"Brush: {self.brush}px  ·  Masked: {self.mask.sum()}px ({pct:.1f}%)  ·  f=fill-gaps",
            fontsize=8, color="#cccccc",
        )
        self.fig.canvas.draw_idle()

    def _stamp(self, cx, cy, paint):
        H, W   = self.mask.shape
        r      = self.brush
        ys, xs = np.ogrid[-r: r + 1, -r: r + 1]
        disc   = xs * xs + ys * ys <= r * r
        y0 = max(0, cy - r);  y1 = min(H, cy + r + 1)
        x0 = max(0, cx - r);  x1 = min(W, cx + r + 1)
        cy0 = y0-(cy-r); cy1 = y1-(cy-r)
        cx0 = x0-(cx-r); cx1 = x1-(cx-r)
        if paint:
            self.mask[y0:y1, x0:x1] |= disc[cy0:cy1, cx0:cx1]
        else:
            self.mask[y0:y1, x0:x1] &= ~disc[cy0:cy1, cx0:cx1]

    def _stroke(self, event, paint):
        if event.inaxes != self.ax_draw or event.xdata is None:
            return
        cx, cy = int(event.xdata), int(event.ydata)
        if self._last_pos is not None:
            lx, ly = self._last_pos
            steps  = max(1, max(abs(cx - lx), abs(cy - ly)))
            for i in range(steps + 1):
                t  = i / steps
                self._stamp(int(round(lx + t*(cx-lx))),
                            int(round(ly + t*(cy-ly))), paint)
        else:
            self._stamp(cx, cy, paint)
        self._last_pos = (cx, cy)
        self._render()

    def _on_press(self, e):
        self._save_history(); self._last_pos = None
        if e.button == 1:   self.drawing = True;  self._stroke(e, True)
        elif e.button == 3: self.erasing = True;  self._stroke(e, False)

    def _on_release(self, e):
        self.drawing = False; self.erasing = False; self._last_pos = None

    def _on_move(self, e):
        if self.drawing:   self._stroke(e, True)
        elif self.erasing: self._stroke(e, False)

    def _on_key(self, e):
        if   e.key in ("+","="): self.brush = min(60, self.brush + 1)
        elif e.key == "-":       self.brush = max(1,  self.brush - 1)
        elif e.key == "u":       self._undo()
        elif e.key == "r":
            self._save_history(); self.mask[:] = False; self._last_pos = None
        elif e.key == "f":       self._fill_gaps()
        self._render()

    def _save_history(self):
        self._history.append(self.mask.copy())
        if len(self._history) > 40: self._history.pop(0)

    def _undo(self):
        if self._history:
            self.mask = self._history.pop(); self._last_pos = None
        else:
            print("  Nothing to undo.")

    def _fill_gaps(self):
        self._save_history()
        struct    = np.ones((9, 9), dtype=bool)
        closed    = binary_closing(self.mask, structure=struct)
        self.mask = binary_dilation(closed, iterations=2).astype(bool)
        print(f"  Gaps filled → {self.mask.sum()} px")

    def get_mask(self):
        plt.show()
        print(f"  Mask: {self.mask.sum()} px ({100.0*self.mask.sum()/self.mask.size:.2f}%)")
        return self.mask