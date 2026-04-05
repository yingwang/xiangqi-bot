#!/usr/bin/env python3
"""
Xiangqi Bot - Auto-play 天天象棋 using Pikafish engine.

Usage:
  1. Open 天天象棋 in WeChat, start a game (initial position)
  2. python3 /tmp/xiangqi_bot.py
  3. Follow the calibration prompts (move mouse to 2 corners)
  4. Bot auto-plays! Ctrl+C to stop.
"""

import subprocess
import sys
import time
import os
import numpy as np
import cv2
import pyautogui
import Quartz

pyautogui.FAILSAFE = True  # Move mouse to corner to abort

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIKAFISH = "/tmp/pikafish-src/src/pikafish"
PIKAFISH_DIR = "/tmp/pikafish-src/src"
TEMPLATE_DIR = os.path.join(_SCRIPT_DIR, "templates")
SCREENSHOT_PATH = os.path.join(_SCRIPT_DIR, "screen.png")
CALIB_PATH = os.path.join(_SCRIPT_DIR, "calib.json")
MOVE_TIME_MS = 500  # Fast for real games with time pressure

# Initial board layouts (screen space)
INIT_RED = [
    ['r','n','b','a','k','a','b','n','r'],
    [None]*9,
    [None,'c',None,None,None,None,None,'c',None],
    ['p',None,'p',None,'p',None,'p',None,'p'],
    [None]*9, [None]*9,
    ['P',None,'P',None,'P',None,'P',None,'P'],
    [None,'C',None,None,None,None,None,'C',None],
    [None]*9,
    ['R','N','B','A','K','A','B','N','R'],
]
INIT_BLACK = [
    ['R','N','B','A','K','A','B','N','R'],
    [None]*9,
    [None,'C',None,None,None,None,None,'C',None],
    ['P',None,'P',None,'P',None,'P',None,'P'],
    [None]*9, [None]*9,
    ['p',None,'p',None,'p',None,'p',None,'p'],
    [None,'c',None,None,None,None,None,'c',None],
    [None]*9,
    ['r','n','b','a','k','a','b','n','r'],
]


class Bot:
    def __init__(self):
        self.cols_logical = []  # 9 x-coords in logical screen space
        self.rows_logical = []  # 10 y-coords in logical screen space
        self.cell_w = 0
        self.cell_h = 0
        self.templates = {}
        self.patch_size = 0
        self.playing_red = True
        self.retina_scale = 2.0
        self.win_id = None
        self.win_x = 0
        self.win_y = 0
        self.cnn = None  # CNN classifier (loaded on demand)

    # --- Window & Screenshot ---

    def find_window(self):
        import Quartz
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
        for w in windows:
            if 'WeChat' in w.get('kCGWindowOwnerName', '') and \
               '天天象棋' in w.get('kCGWindowName', ''):
                self.win_id = w['kCGWindowNumber']
                b = w['kCGWindowBounds']
                self.win_x, self.win_y = int(b['X']), int(b['Y'])
                lw = int(b['Width'])
                print(f"  Window: id={self.win_id} pos=({self.win_x},{self.win_y}) "
                      f"size={lw}x{int(b['Height'])}")
                return
        raise RuntimeError("天天象棋 window not found!")

    def screenshot_for_processing(self):
        """Capture window, return image. Uses unique filename each time."""
        self._ss_counter = getattr(self, '_ss_counter', 0) + 1
        path = os.path.join(_SCRIPT_DIR, f"ss_{self._ss_counter % 3}.png")
        subprocess.run(['screencapture', '-x', '-o', '-l', str(self.win_id),
                        path], capture_output=True, check=True)
        full = cv2.imread(path)
        if full is None:
            raise RuntimeError("Screenshot failed!")
        # Determine retina scale from window capture
        import Quartz
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
        for w in windows:
            if w.get('kCGWindowNumber') == self.win_id:
                lw = int(w['kCGWindowBounds']['Width'])
                self.retina_scale = full.shape[1] / lw
                break
        return full

    def logical_to_pixel(self, lx, ly):
        """Convert logical screen coords to full-res pixel coords in window capture."""
        # Logical coords are absolute screen coords
        # Window capture starts at (win_x, win_y) in logical space
        px = (lx - self.win_x) * self.retina_scale
        py = (ly - self.win_y) * self.retina_scale
        return int(px), int(py)

    # --- Calibration ---

    def calibrate(self):
        """Calibration: user points mouse to 2 corner pieces (countdown, no Enter needed)."""
        print("\n=== CALIBRATION ===")
        print("Move mouse to TOP-LEFT corner piece (leftmost piece on top rank)")
        for i in range(5, 0, -1):
            print(f"  Capturing in {i}...", end="\r")
            time.sleep(1)
        x1, y1 = pyautogui.position()
        print(f"  Top-left: ({x1}, {y1})        ")

        print("\nNow move mouse to BOTTOM-RIGHT corner piece (rightmost piece on bottom rank)")
        for i in range(5, 0, -1):
            print(f"  Capturing in {i}...", end="\r")
            time.sleep(1)
        x2, y2 = pyautogui.position()
        print(f"  Bottom-right: ({x2}, {y2})        ")

        self.cell_w = (x2 - x1) / 8.0
        self.cell_h = (y2 - y1) / 9.0

        self.cols_logical = [x1 + i * self.cell_w for i in range(9)]
        self.rows_logical = [y1 + j * self.cell_h for j in range(10)]

        print(f"\n  Cell size: {self.cell_w:.1f} x {self.cell_h:.1f} logical pixels")
        print(f"  Grid: x=[{x1:.0f}..{x2:.0f}] y=[{y1:.0f}..{y2:.0f}]")

        # Save calibration as relative to window (survives resize)
        import json
        win_w = self._get_window_width()
        win_h = self._get_window_height()
        with open(CALIB_PATH, 'w') as f:
            json.dump({
                'rx1': (x1 - self.win_x) / win_w,
                'ry1': (y1 - self.win_y) / win_h,
                'rx2': (x2 - self.win_x) / win_w,
                'ry2': (y2 - self.win_y) / win_h,
            }, f)
        print(f"  Saved to {CALIB_PATH}")

    def _get_window_width(self):
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
        for w in windows:
            if w.get('kCGWindowNumber') == self.win_id:
                return int(w['kCGWindowBounds']['Width'])
        return 1628  # fallback

    def _get_window_height(self):
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
        for w in windows:
            if w.get('kCGWindowNumber') == self.win_id:
                return int(w['kCGWindowBounds']['Height'])
        return 960  # fallback

    def load_calibration(self):
        """Load calibration, auto-adapt to current window size/position."""
        import json
        if not os.path.exists(CALIB_PATH):
            return False
        try:
            with open(CALIB_PATH) as f:
                d = json.load(f)

            # Support both relative (new) and absolute (old) formats
            if 'rx1' in d:
                win_w = self._get_window_width()
                win_h = self._get_window_height()
                x1 = self.win_x + d['rx1'] * win_w
                y1 = self.win_y + d['ry1'] * win_h
                x2 = self.win_x + d['rx2'] * win_w
                y2 = self.win_y + d['ry2'] * win_h
            else:
                x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
                if d.get('win_x') != self.win_x or d.get('win_y') != self.win_y:
                    dx = self.win_x - d.get('win_x', 0)
                    dy = self.win_y - d.get('win_y', 0)
                    x1 += dx; y1 += dy; x2 += dx; y2 += dy

            self.cell_w = (x2 - x1) / 8.0
            self.cell_h = (y2 - y1) / 9.0
            self.cols_logical = [x1 + i * self.cell_w for i in range(9)]
            self.rows_logical = [y1 + j * self.cell_h for j in range(10)]
            print(f"  Loaded calibration: cell={self.cell_w:.1f}x{self.cell_h:.1f}")
            return True
        except:
            return False

    # --- Orientation & Templates ---

    def detect_orientation(self, img):
        """Check if top-row pieces are red (→ user plays BLACK)."""
        ps = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.6)
        red_total = 0
        for ci in [0, 4, 8]:
            px, py = self.logical_to_pixel(self.cols_logical[ci], self.rows_logical[0])
            h, w = img.shape[:2]
            patch = img[max(0,py-ps):min(h,py+ps), max(0,px-ps):min(w,px+ps)]
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255))
            m2 = cv2.inRange(hsv, (168, 60, 60), (180, 255, 255))
            red_total += (cv2.countNonZero(m1) + cv2.countNonZero(m2)) / max(1, patch.size//3)
        self.playing_red = red_total < 0.03
        print(f"  You play: {'RED' if self.playing_red else 'BLACK'}")

    def capture_templates(self, img):
        """Capture templates from ALL initial positions (multiple per piece)."""
        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        init = INIT_RED if self.playing_red else INIT_BLACK
        ps = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.7)
        self.patch_size = ps
        self.templates = {}  # piece -> [tmpl1, tmpl2, ...]

        for r in range(10):
            for c in range(9):
                piece = init[r][c]
                if piece is None:
                    continue
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                patch = self._extract(img, px, py, ps)
                if patch is not None:
                    if piece not in self.templates:
                        self.templates[piece] = []
                    self.templates[piece].append(patch)

        # Empty cells from many positions (including edges)
        self.templates['_'] = []
        for er in [1, 4, 5, 8]:  # rows that are fully empty
            for ec in range(9):
                px, py = self.logical_to_pixel(self.cols_logical[ec], self.rows_logical[er])
                ep = self._extract(img, px, py, ps)
                if ep is not None:
                    self.templates['_'].append(ep)

        total = sum(len(v) for k, v in self.templates.items() if k != '_')
        print(f"  Templates: {len(self.templates)} types, {total} variants")

    def _extract(self, img, cx, cy, ps):
        h, w = img.shape[:2]
        x1, y1 = max(0, cx-ps), max(0, cy-ps)
        x2, y2 = min(w, cx+ps), min(h, cy+ps)
        p = img[y1:y2, x1:x2]
        return p if p.shape[0] >= ps and p.shape[1] >= ps else None

    # --- Color-based Piece Classifier (v2) ---

    def _extract_piece_center(self, img, px, py, radius=None):
        """Extract the circular piece region centered at (px, py)."""
        if radius is None:
            radius = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.28)
        h, w = img.shape[:2]
        x1 = max(0, px - radius)
        y1 = max(0, py - radius)
        x2 = min(w, px + radius)
        y2 = min(h, py + radius)
        patch = img[y1:y2, x1:x2]
        if patch.shape[0] < radius or patch.shape[1] < radius:
            return None
        return patch

    def _find_piece_circle(self, img, px, py):
        """Detect the circular piece token using Hough circles.

        Returns (cx, cy, cr) in image coordinates if found, or None.
        The circle center is in ABSOLUTE image coordinates.
        """
        scale = min(self.cell_w, self.cell_h) * self.retina_scale
        search_r = int(scale * 0.55)
        piece_r_min = int(scale * 0.22)
        piece_r_max = int(scale * 0.42)
        center_tol = int(scale * 0.25)

        h, w = img.shape[:2]
        x1 = max(0, px - search_r)
        y1 = max(0, py - search_r)
        x2 = min(w, px + search_r)
        y2 = min(h, py + search_r)
        patch = img[y1:y2, x1:x2]

        if patch.shape[0] < 30 or patch.shape[1] < 30:
            return None

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        ph, pw = gray.shape[:2]

        # Try multiple blur sizes and Hough sensitivity levels
        for blur_k in [7, 5, 9]:
            blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 2.0)
            for p1, p2 in [(70, 30), (60, 25), (50, 22), (80, 35)]:
                circles = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                    minDist=search_r,
                    param1=p1, param2=p2,
                    minRadius=piece_r_min,
                    maxRadius=piece_r_max
                )
                if circles is not None:
                    # Find circle closest to patch center
                    best_circ = None
                    best_dist = float('inf')
                    for circ in circles[0]:
                        cx, cy, cr = circ
                        dist = np.sqrt((cx - pw / 2) ** 2 + (cy - ph / 2) ** 2)
                        if dist < center_tol and dist < best_dist:
                            best_dist = dist
                            best_circ = circ
                    if best_circ is not None:
                        abs_cx = int(best_circ[0] + x1)
                        abs_cy = int(best_circ[1] + y1)
                        abs_cr = int(best_circ[2])
                        return (abs_cx, abs_cy, abs_cr)
        return None

    def _has_piece_v2(self, img, px, py):
        """Check if the cell at (px, py) has a piece.

        Uses Hough circle detection as primary method, with a color-based
        fallback that checks for the characteristic piece ring color.
        """
        # Primary: Hough circle detection
        circle = self._find_piece_circle(img, px, py)
        if circle is not None:
            return True

        # Fallback: check for red/dark ring pixels in an annular region
        # This catches pieces that Hough misses (edge cases, partial visibility)
        scale = min(self.cell_w, self.cell_h) * self.retina_scale
        outer_r = int(scale * 0.35)
        inner_r = int(scale * 0.22)

        h, w = img.shape[:2]
        x1 = max(0, px - outer_r)
        y1 = max(0, py - outer_r)
        x2 = min(w, px + outer_r)
        y2 = min(h, py + outer_r)
        patch = img[y1:y2, x1:x2]
        if patch.shape[0] < outer_r or patch.shape[1] < outer_r:
            return False

        ph, pw = patch.shape[:2]
        pcx, pcy = pw // 2, ph // 2

        # Create annular mask (ring where piece border would be)
        mask_outer = np.zeros((ph, pw), dtype=np.uint8)
        mask_inner = np.zeros((ph, pw), dtype=np.uint8)
        cv2.circle(mask_outer, (pcx, pcy), outer_r, 255, -1)
        cv2.circle(mask_inner, (pcx, pcy), inner_r, 255, -1)
        ring_mask = cv2.subtract(mask_outer, mask_inner)

        # Check for red ring pixels (red piece borders)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, (0, 80, 100), (12, 255, 200))
        red_mask2 = cv2.inRange(hsv, (168, 80, 100), (180, 255, 200))
        red_ring = cv2.bitwise_and(cv2.bitwise_or(red_mask1, red_mask2), ring_mask)
        red_count = cv2.countNonZero(red_ring)

        # Check for dark ring pixels (black piece borders)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        dark_ring_vals = gray[ring_mask > 0]
        dark_count = int(np.sum(dark_ring_vals < 100)) if len(dark_ring_vals) > 0 else 0

        ring_total = max(1, cv2.countNonZero(ring_mask))
        red_ratio = red_count / ring_total
        dark_ratio = dark_count / ring_total

        # A piece border needs both red/dark ring AND bright center (piece token is convex)
        # Be strict to avoid false positives from board borders and adjacent piece edges
        center_mask = np.zeros((ph, pw), dtype=np.uint8)
        cv2.circle(center_mask, (pcx, pcy), int(scale * 0.10), 255, -1)
        center_vals = gray[center_mask > 0]
        center_brightness = float(np.mean(center_vals)) if len(center_vals) > 0 else 0
        ring_brightness = float(np.mean(dark_ring_vals)) if len(dark_ring_vals) > 0 else 0

        has_ring = red_ratio > 0.15 or dark_ratio > 0.25
        has_bright_center = center_brightness > 140 and center_brightness > ring_brightness + 5

        return has_ring and has_bright_center

    def _classify_color_v2(self, img, px, py):
        """Classify a piece as RED or BLACK based on text/ring color.

        Red pieces have red-colored Chinese characters and ring on a tan background.
        Black pieces have dark/black characters and ring on a tan background.

        Returns 'red' or 'black'.
        """
        # Use the piece center for color analysis
        radius = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.22)

        # Try to use actual circle center if found
        circle = self._find_piece_circle(img, px, py)
        if circle is not None:
            px, py = circle[0], circle[1]

        h, w = img.shape[:2]
        x1 = max(0, px - radius)
        y1 = max(0, py - radius)
        x2 = min(w, px + radius)
        y2 = min(h, py + radius)
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            return 'unknown'

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        # Create circular mask
        ph, pw = patch.shape[:2]
        mask = np.zeros((ph, pw), dtype=np.uint8)
        cv2.circle(mask, (pw // 2, ph // 2), min(ph, pw) // 2, 255, -1)

        # Count red pixels (text/ring of red pieces)
        red_mask1 = cv2.inRange(hsv, (0, 70, 80), (12, 255, 255))
        red_mask2 = cv2.inRange(hsv, (168, 70, 80), (180, 255, 255))
        red_mask = cv2.bitwise_and(cv2.bitwise_or(red_mask1, red_mask2), mask)
        red_count = cv2.countNonZero(red_mask)

        # Count dark pixels (text/ring of black pieces)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        dark_vals = gray[mask > 0]
        dark_count = int(np.sum(dark_vals < 80)) if len(dark_vals) > 0 else 0

        total = max(1, cv2.countNonZero(mask))
        red_ratio = red_count / total
        dark_ratio = dark_count / total

        if red_ratio > 0.03:
            return 'red'
        elif dark_ratio > 0.03:
            return 'black'
        else:
            return 'red' if red_ratio > dark_ratio else 'black'

    def _compute_feature_vector(self, img, px, py, grid_size=5):
        """Compute a structural feature vector from the character on a piece.

        Uses multiple complementary features:
        1. Stroke density in a grid (captures overall character shape)
        2. Horizontal and vertical edge density (captures stroke orientations)
        3. Multi-scale: coarse (4x4) and fine (6x6) grids

        Returns a normalized feature vector.
        """
        radius = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.18)

        # Use actual circle center if found
        circle = self._find_piece_circle(img, px, py)
        if circle is not None:
            px, py = circle[0], circle[1]

        h, w = img.shape[:2]
        x1 = max(0, px - radius)
        y1 = max(0, py - radius)
        x2 = min(w, px + radius)
        y2 = min(h, py + radius)
        patch = img[y1:y2, x1:x2]
        if patch.shape[0] < radius or patch.shape[1] < radius:
            return None

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ph, pw = gray.shape

        # Create circular mask
        mask = np.zeros((ph, pw), dtype=np.float32)
        cv2.circle(mask, (pw // 2, ph // 2), min(ph, pw) // 2, 1.0, -1)

        # Normalize brightness within mask
        masked_pixels = gray[mask > 0]
        if len(masked_pixels) == 0:
            return None
        local_mean = np.mean(masked_pixels)
        local_std = np.std(masked_pixels)
        if local_std < 5:
            return None

        # Normalized grayscale (mean-subtracted, std-normalized)
        norm_gray = (gray - local_mean) / local_std
        norm_gray = norm_gray * mask

        # Compute edge maps for orientation features
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * mask
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * mask

        features = []

        # Feature set 1: normalized brightness in grid cells (captures character structure)
        for gs in [4, 6]:  # multi-scale
            for gi in range(gs):
                for gj in range(gs):
                    y_start = int(gi * ph / gs)
                    y_end = int((gi + 1) * ph / gs)
                    x_start = int(gj * pw / gs)
                    x_end = int((gj + 1) * pw / gs)
                    region = norm_gray[y_start:y_end, x_start:x_end]
                    region_mask = mask[y_start:y_end, x_start:x_end]
                    mask_sum = np.sum(region_mask)
                    if mask_sum > 0:
                        features.append(float(np.sum(region) / mask_sum))
                    else:
                        features.append(0.0)

        # Feature set 2: horizontal vs vertical edge balance in grid cells
        for gi in range(4):
            for gj in range(4):
                y_start = int(gi * ph / 4)
                y_end = int((gi + 1) * ph / 4)
                x_start = int(gj * pw / 4)
                x_end = int((gj + 1) * pw / 4)

                sx = np.abs(sobel_x[y_start:y_end, x_start:x_end])
                sy = np.abs(sobel_y[y_start:y_end, x_start:x_end])
                region_mask = mask[y_start:y_end, x_start:x_end]
                mask_sum = np.sum(region_mask)
                if mask_sum > 0:
                    # Ratio of vertical to horizontal edges
                    sx_sum = float(np.sum(sx * region_mask))
                    sy_sum = float(np.sum(sy * region_mask))
                    total_edge = sx_sum + sy_sum + 1e-8
                    features.append(sx_sum / total_edge)  # horizontal dominance
                else:
                    features.append(0.5)

        vec = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec

    def _cosine_similarity(self, v1, v2):
        """Compute cosine similarity between two vectors."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def capture_feature_vectors(self, img):
        """Capture feature vectors from known piece positions in the initial layout.

        This builds self.piece_features: dict mapping piece code -> list of feature vectors
        and self.piece_color_map: dict mapping piece code -> 'red' or 'black'
        """
        init = INIT_RED if self.playing_red else INIT_BLACK
        self.piece_features = {}  # piece -> [vec1, vec2, ...]
        self.piece_color_map = {}

        for r in range(10):
            for c in range(9):
                piece = init[r][c]
                if piece is None:
                    continue
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                vec = self._compute_feature_vector(img, px, py)
                if vec is not None:
                    if piece not in self.piece_features:
                        self.piece_features[piece] = []
                    self.piece_features[piece].append(vec)
                    # Map piece to color
                    if piece.isupper():
                        self.piece_color_map[piece] = 'red' if self.playing_red else 'black'
                    else:
                        self.piece_color_map[piece] = 'black' if self.playing_red else 'red'

        total = sum(len(v) for v in self.piece_features.values())
        print(f"  Feature vectors: {len(self.piece_features)} types, {total} vectors")

    def capture_feature_vectors_from_board(self, img, board):
        """Rebuild feature vectors from current known board state.

        Uses only cells where we confidently know what piece is there.
        """
        self.piece_features = {}
        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece is None:
                    continue
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                vec = self._compute_feature_vector(img, px, py)
                if vec is not None:
                    if piece not in self.piece_features:
                        self.piece_features[piece] = []
                    self.piece_features[piece].append(vec)

    def identify_v2(self, img, px, py):
        """Identify the piece at pixel position (px, py) using color-based classification.

        Returns piece code (e.g. 'R', 'n', 'K') or None if empty.

        Algorithm:
        1. Check if cell has a piece (Hough circle + edge density fallback)
        2. Classify red vs black via HSV analysis
        3. Match character structure via stroke-density feature vector similarity
        """
        # Step 1: Is there a piece here?
        if not self._has_piece_v2(img, px, py):
            return None

        # Step 2: Red or Black?
        color = self._classify_color_v2(img, px, py)

        # Step 3: Match type via feature vector
        vec = self._compute_feature_vector(img, px, py)
        if vec is None:
            return None

        if not hasattr(self, 'piece_features') or not self.piece_features:
            return None

        # Filter candidates by color
        candidates = {}
        for piece, vecs in self.piece_features.items():
            piece_color = self.piece_color_map.get(piece, 'unknown')
            if piece_color == color or color == 'unknown':
                candidates[piece] = vecs

        if not candidates:
            # Fall back to all pieces
            candidates = self.piece_features

        # Find best match by cosine similarity
        best_piece = None
        best_sim = -1.0

        for piece, vecs in candidates.items():
            for ref_vec in vecs:
                sim = self._cosine_similarity(vec, ref_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_piece = piece

        # Threshold check
        if best_sim < 0.60:
            return None

        return best_piece

    def parse_board_v2(self, img):
        """Parse the full board using identify_v2."""
        board = []
        for r in range(10):
            row = []
            for c in range(9):
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                p = self.identify_v2(img, px, py)
                row.append(p)
            board.append(row)
        return board

    # --- Move validation helpers ---

    def _valid_destinations(self, piece, from_row, from_col, board):
        """Return set of (row, col) reachable positions for a piece.

        Uses basic Chinese chess movement rules. Not a full legal move generator
        (doesn't check for checks etc.) but validates piece-type movement patterns.
        """
        dests = set()
        is_upper = piece.isupper()
        p = piece.upper()

        # Determine which half of the board is "own side"
        # In screen coords: rows 0-4 are top, 5-9 are bottom
        # If playing red: red is bottom (5-9), black is top (0-4)
        # If playing black: red is top (0-4), black is bottom (5-9)
        if self.playing_red:
            own_top = 5 if is_upper else 0  # uppercase=red=bottom, lower=black=top
            own_bottom = 9 if is_upper else 4
            enemy_top = 0 if is_upper else 5
            enemy_bottom = 4 if is_upper else 9
        else:
            own_top = 0 if is_upper else 5
            own_bottom = 4 if is_upper else 9
            enemy_top = 5 if is_upper else 0
            enemy_bottom = 9 if is_upper else 4

        r, c = from_row, from_col

        if p == 'R':  # Rook: straight lines
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                while 0 <= nr <= 9 and 0 <= nc <= 8:
                    dests.add((nr, nc))
                    if board[nr][nc] is not None:
                        break
                    nr += dr
                    nc += dc

        elif p == 'N':  # Knight
            for dr, dc, br, bc in [
                (-2, -1, -1, 0), (-2, 1, -1, 0),
                (2, -1, 1, 0), (2, 1, 1, 0),
                (-1, -2, 0, -1), (-1, 2, 0, 1),
                (1, -2, 0, -1), (1, 2, 0, 1)
            ]:
                # Check blocking piece
                block_r, block_c = r + br, c + bc
                if 0 <= block_r <= 9 and 0 <= block_c <= 8:
                    if board[block_r][block_c] is not None:
                        continue
                nr, nc = r + dr, c + dc
                if 0 <= nr <= 9 and 0 <= nc <= 8:
                    dests.add((nr, nc))

        elif p == 'B':  # Bishop/Elephant: diagonal 2 steps, own half only
            for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                # Check blocking piece at midpoint
                mr, mc = r + dr // 2, c + dc // 2
                if 0 <= mr <= 9 and 0 <= mc <= 8:
                    if board[mr][mc] is not None:
                        continue
                nr, nc = r + dr, c + dc
                if 0 <= nr <= 9 and 0 <= nc <= 8:
                    if own_top <= nr <= own_bottom:
                        dests.add((nr, nc))

        elif p == 'A':  # Advisor: diagonal 1 step within palace
            # Palace: top 3 rows if own side is top, bottom 3 if own side is bottom
            if own_top == 0:
                palace_r_min, palace_r_max = 0, 2
            else:
                palace_r_min, palace_r_max = 7, 9
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 3 <= nc <= 5 and palace_r_min <= nr <= palace_r_max:
                    dests.add((nr, nc))

        elif p == 'K':  # King: orthogonal 1 step within palace
            if own_top == 0:
                palace_r_min, palace_r_max = 0, 2
            else:
                palace_r_min, palace_r_max = 7, 9
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 3 <= nc <= 5 and palace_r_min <= nr <= palace_r_max:
                    dests.add((nr, nc))
            # Flying general (face-to-face kings) - can capture opposing king
            for dr in [1, -1]:
                nr = r + dr
                while 0 <= nr <= 9:
                    if board[nr][c] is not None:
                        if board[nr][c].upper() == 'K':
                            dests.add((nr, c))
                        break
                    nr += dr

        elif p == 'C':  # Cannon: straight lines, jump capture
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                jumped = False
                while 0 <= nr <= 9 and 0 <= nc <= 8:
                    if not jumped:
                        if board[nr][nc] is None:
                            dests.add((nr, nc))
                        else:
                            jumped = True
                    else:
                        if board[nr][nc] is not None:
                            dests.add((nr, nc))
                            break
                    nr += dr
                    nc += dc

        elif p == 'P':  # Pawn
            # Determine forward direction
            if self.playing_red:
                fwd = -1 if is_upper else 1
            else:
                fwd = 1 if is_upper else -1

            nr = r + fwd
            if 0 <= nr <= 9:
                dests.add((nr, c))

            # After crossing river, can move sideways
            crossed = (r <= own_top - 1) or (r >= own_bottom + 1)
            # More precise: pawn has crossed if it's in enemy half
            if enemy_top <= r <= enemy_bottom:
                crossed = True
            if crossed:
                for dc in [-1, 1]:
                    nc = c + dc
                    if 0 <= nc <= 8:
                        dests.add((r, nc))

        return dests

    def detect_move_v2(self, img_before, img_after, board_before):
        """Detect opponent's move using identify_v2 and move validation.

        Algorithm:
        1. Find cells that changed visually (pixel diff)
        2. Use identify_v2 to classify what's at each changed cell
        3. Determine source (cell that lost a piece) and dest (cell that gained/changed)
        4. Validate against piece movement rules
        5. Fall back to heuristics if needed
        """
        board_after = [row[:] for row in board_before]
        changed = []

        # Step 1: Find changed cells
        for r in range(10):
            for c in range(9):
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                ps = self.patch_size
                h, w = img_before.shape[:2]
                x1, y1 = max(0, px - ps), max(0, py - ps)
                x2, y2 = min(w, px + ps), min(h, py + ps)
                p1 = img_before[y1:y2, x1:x2]
                p2 = img_after[y1:y2, x1:x2]
                if p1.shape != p2.shape:
                    continue
                diff = cv2.absdiff(p1, p2)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                change = np.count_nonzero(gray_diff > 30) / max(1, gray_diff.size)
                if change > 0.03:
                    changed.append((r, c, change))

        if not changed:
            print("    detect_move_v2: no changed cells")
            return board_before

        # Refresh feature vectors from unchanged cells in the new image
        self.capture_feature_vectors_from_board(img_after, board_before)

        # Step 2: Identify what's at each changed cell now
        cell_info = []
        for r, c, change_amt in changed:
            px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
            had_piece = board_before[r][c]
            has_piece_now = self._has_piece_v2(img_after, px, py)
            now_piece = self.identify_v2(img_after, px, py) if has_piece_now else None
            cell_info.append({
                'row': r, 'col': c,
                'change': change_amt,
                'had': had_piece,
                'has_now': has_piece_now,
                'now_piece': now_piece,
            })

        # Debug output
        for ci in cell_info:
            print(f"    Changed ({ci['row']},{ci['col']}): "
                  f"had={ci['had']} now={ci['now_piece']} "
                  f"has_piece={ci['has_now']} change={ci['change']:.3f}")

        # Step 3: Find source and destination
        # Source: cell that had an opponent piece and now is empty or has different piece
        # Destination: cell that now has a piece that wasn't there before, or has a different piece

        sources = []  # Cells that lost a piece
        dests = []    # Cells that gained/changed a piece

        for ci in cell_info:
            if ci['had'] is not None and not ci['has_now']:
                sources.append(ci)
            elif ci['had'] is None and ci['has_now']:
                dests.append(ci)
            elif ci['had'] is not None and ci['has_now']:
                # Piece was here and still a piece here - could be capture destination
                # or could be a highlight/selection artifact
                if ci['now_piece'] is not None and ci['now_piece'] != ci['had']:
                    # Different piece arrived - this is a capture destination
                    dests.append(ci)
                elif ci['change'] > 0.15:
                    # Large change with piece still there - might be dest of a capture
                    dests.append(ci)

        # Handle case: exactly one source, one or zero dests
        if len(sources) == 1 and len(dests) == 1:
            src = sources[0]
            dst = dests[0]
            moving_piece = src['had']

            # Step 4: Validate move
            valid_dests = self._valid_destinations(
                moving_piece, src['row'], src['col'], board_before)

            if (dst['row'], dst['col']) in valid_dests:
                board_after[src['row']][src['col']] = None
                board_after[dst['row']][dst['col']] = moving_piece
                print(f"    Move: {moving_piece} ({src['row']},{src['col']}) → "
                      f"({dst['row']},{dst['col']}) [validated]")
                return board_after
            else:
                # Move doesn't match rules for this piece type - still apply it
                # but warn (could be wrong identification)
                board_after[src['row']][src['col']] = None
                board_after[dst['row']][dst['col']] = moving_piece
                print(f"    Move: {moving_piece} ({src['row']},{src['col']}) → "
                      f"({dst['row']},{dst['col']}) [UNVALIDATED - may be wrong piece type]")
                return board_after

        elif len(sources) == 1 and len(dests) == 0:
            # Piece left but no dest found - might be a piece that moved to a cell
            # with similar appearance. Check all changed cells with piece present.
            src = sources[0]
            moving_piece = src['had']
            valid_dests = self._valid_destinations(
                moving_piece, src['row'], src['col'], board_before)

            # Check which changed cells with a piece are valid destinations
            for ci in cell_info:
                if ci is src:
                    continue
                if ci['has_now'] and (ci['row'], ci['col']) in valid_dests:
                    board_after[src['row']][src['col']] = None
                    board_after[ci['row']][ci['col']] = moving_piece
                    print(f"    Move: {moving_piece} ({src['row']},{src['col']}) → "
                          f"({ci['row']},{ci['col']}) [rule-validated dest]")
                    return board_after

            print(f"    Piece {moving_piece} left ({src['row']},{src['col']}) but no valid dest")

        elif len(sources) >= 2 and len(dests) >= 1:
            # Multiple changes - try to find valid move pairs
            # This can happen with move highlight effects
            best_move = None
            best_score = -1

            for src in sources:
                moving_piece = src['had']
                valid = self._valid_destinations(
                    moving_piece, src['row'], src['col'], board_before)
                for dst in dests:
                    if (dst['row'], dst['col']) in valid:
                        # Score by change amount (higher = more likely real change)
                        score = src['change'] + dst['change']
                        if score > best_score:
                            best_score = score
                            best_move = (src, dst, moving_piece)

            if best_move:
                src, dst, moving_piece = best_move
                board_after[src['row']][src['col']] = None
                board_after[dst['row']][dst['col']] = moving_piece
                print(f"    Move: {moving_piece} ({src['row']},{src['col']}) → "
                      f"({dst['row']},{dst['col']}) [best valid from multiple]")
                return board_after

        # Step 5: Fallback - use the two most-changed cells
        changed_sorted = sorted(cell_info, key=lambda x: x['change'], reverse=True)
        if len(changed_sorted) >= 2:
            # Assume most changed are source and dest
            c1, c2 = changed_sorted[0], changed_sorted[1]
            # Source is the one that had a piece and lost it
            if c1['had'] is not None and not c1['has_now']:
                src, dst = c1, c2
            elif c2['had'] is not None and not c2['has_now']:
                src, dst = c2, c1
            elif c1['had'] is not None:
                src, dst = c1, c2
            else:
                src, dst = c2, c1

            if src['had'] is not None:
                board_after[src['row']][src['col']] = None
                board_after[dst['row']][dst['col']] = src['had']
                print(f"    Move: {src['had']} ({src['row']},{src['col']}) → "
                      f"({dst['row']},{dst['col']}) [fallback: most changed]")
                return board_after

        print("    detect_move_v2: could not determine move")
        return board_before

    # --- Board Parsing (v1 - template matching) ---

    def _masked_corr(self, patch, tmpl, mask):
        """Compute normalized correlation within circular mask."""
        p = patch[mask > 0].astype(np.float32)
        t = tmpl[mask > 0].astype(np.float32)
        if len(p) == 0 or p.std() < 1 or t.std() < 1:
            return 0.0
        pn = (p - p.mean()) / p.std()
        tn = (t - t.mean()) / t.std()
        return float(np.mean(pn * tn))

    def identify(self, img, px, py):
        patch = self._extract(img, px, py, self.patch_size)
        if patch is None:
            return None
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if gray.std() < 5:
            return None

        h, w = patch.shape[:2]
        # Circular mask — only compare inside the piece, ignore background
        mask = np.zeros((h, w), dtype=np.uint8)
        r = int(min(h, w) * 0.32)
        cv2.circle(mask, (w//2, h//2), r, 255, -1)
        # 3-channel mask for color images
        mask3 = cv2.merge([mask, mask, mask])

        best_piece, best_piece_sc = None, 0.0
        best_empty_sc = 0.0

        for pc, tmpls in self.templates.items():
            for tmpl in tmpls:
                t = cv2.resize(tmpl, (w, h))
                sc = self._masked_corr(patch, t, mask)
                if pc == '_':
                    best_empty_sc = max(best_empty_sc, sc)
                elif sc > best_piece_sc:
                    best_piece_sc, best_piece = sc, pc

        if best_empty_sc > best_piece_sc:
            return None
        if best_piece_sc - best_empty_sc < 0.08:
            return None
        return best_piece if best_piece_sc > 0.3 else None

    def parse_board(self, img):
        board = []
        for r in range(10):
            row = []
            for c in range(9):
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                p = self.identify(img, px, py)
                row.append(p)
            board.append(row)
        return board

    def _cell_has_piece_now(self, img, col, row):
        """Check if a cell currently has a piece using brightness."""
        px, py = self.logical_to_pixel(self.cols_logical[col], self.rows_logical[row])
        ps = self.patch_size // 2  # smaller region for center check
        h, w = img.shape[:2]
        x1, y1 = max(0, px-ps), max(0, py-ps)
        x2, y2 = min(w, px+ps), min(h, py+ps)
        patch = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Pieces are brighter circles with text = higher std
        return gray.std() > 25

    def get_legal_moves(self, fen):
        """Get all legal moves from pikafish for the given position."""
        proc = subprocess.Popen(
            [PIKAFISH], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=PIKAFISH_DIR)
        proc.stdin.write(f"uci\nisready\nposition fen {fen}\ngo perft 1\n")
        proc.stdin.flush()
        moves = []
        t0 = time.time()
        while time.time() - t0 < 5:
            line = proc.stdout.readline().strip()
            if not line: continue
            if ':' in line and len(line.split(':')[0].strip()) == 4:
                moves.append(line.split(':')[0].strip())
            if line.startswith('Nodes'):
                break
        try: proc.stdin.write("quit\n"); proc.stdin.flush()
        except: pass
        try: proc.wait(timeout=2)
        except: proc.kill()
        return moves

    def _find_move(self, old_fen, new_fen):
        """Find the UCI move that transforms old_fen → new_fen."""
        opp_turn = 'b' if self.playing_red else 'w'
        legal = self.get_legal_moves(f"{old_fen} {opp_turn} - - 0 1")
        for move in legal:
            # Simulate this move on a temp board
            src, dst = self.uci_to_screen_cells(move)
            # Quick check: just try to match FEN
            # (We already know the result board, so just find which legal move matches)
            fc, fr = ord(move[0]) - ord('a'), int(move[1])
            tc, tr = ord(move[2]) - ord('a'), int(move[3])
            if self.playing_red:
                sr1, sc1 = 9 - fr, fc
                sr2, sc2 = 9 - tr, tc
            else:
                sr1, sc1 = fr, 8 - fc
                sr2, sc2 = tr, 8 - tc
            # Parse old_fen to board, apply move, check if result matches new_fen
            # For speed, just return the first move that could explain the change
            temp = self._fen_to_board(old_fen)
            if temp and temp[sr1][sc1] is not None:
                temp[sr2][sc2] = temp[sr1][sc1]
                temp[sr1][sc1] = None
                if self.board_to_fen(temp) == new_fen:
                    return move
        return None

    def _fen_to_board(self, fen):
        """Convert FEN string to 10x9 board array."""
        rows = fen.split('/')
        if len(rows) != 10:
            return None
        board = []
        for row_str in rows:
            row = []
            for ch in row_str:
                if ch.isdigit():
                    row.extend([None] * int(ch))
                else:
                    row.append(ch)
            if len(row) != 9:
                return None
            board.append(row)
        if not self.playing_red:
            board = [row[::-1] for row in reversed(board)]
        return board

    def uci_to_screen_cells(self, move):
        """Convert UCI move to screen (row, col) pairs."""
        fc, fr = ord(move[0]) - ord('a'), int(move[1])
        tc, tr = ord(move[2]) - ord('a'), int(move[3])
        if self.playing_red:
            return (9-fr, fc), (9-tr, tc)
        else:
            return (fr, 8-fc), (tr, 8-tc)

    def _cell_change(self, img_before, img_after, r, c):
        """Compute pixel change at a specific cell (centered, no overlap)."""
        px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
        # 35% of cell size as half-width — fits inside cell, no neighbor overlap
        # (old 0.7 caused 218px patch vs 156px cell spacing = massive overlap!)
        hs = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.35)
        h, w = img_before.shape[:2]
        x1, y1 = max(0, px-hs), max(0, py-hs)
        x2, y2 = min(w, px+hs), min(h, py+hs)
        p1 = img_before[y1:y2, x1:x2]
        p2 = img_after[y1:y2, x1:x2]
        if p1.shape != p2.shape or p1.size == 0:
            return 0
        return cv2.absdiff(p1, p2).mean()

    def detect_move_perft(self, img_before, img_after, board_before, fen_before):
        """Detect opponent's move: score each legal move by pixel change at src+dst."""
        board_after = [row[:] for row in board_before]

        opp_turn = 'b' if self.playing_red else 'w'
        opp_fen = f"{fen_before} {opp_turn} - - 0 1"
        legal_moves = self.get_legal_moves(opp_fen)

        if not legal_moves:
            print(f"    No legal moves")
            return board_after

        # Score every legal move by how much its src+dst cells changed
        scored = []
        for move in legal_moves:
            src, dst = self.uci_to_screen_cells(move)
            sc = self._cell_change(img_before, img_after, src[0], src[1])
            dc = self._cell_change(img_before, img_after, dst[0], dst[1])
            scored.append((move, src, dst, sc + dc))

        scored.sort(key=lambda x: -x[3])
        best_move, best_src, best_dst, best_score = scored[0]

        # Debug: show top 3
        top3 = [(m, f"{s:.1f}") for m, _, _, s in scored[:3]]
        print(f"    Top: {top3}")

        # Correct detections score 50-100+. Below 20 is always wrong.
        if best_score < 20:
            print(f"    ⚠ Score too low (Δ={best_score:.1f}), need > 20")
            return None
        if best_score < 40 and len(scored) > 1 and best_score < scored[1][3] * 2:
            print(f"    ⚠ Low confidence (Δ={best_score:.1f}), gap too small")
            return None

        piece = board_before[best_src[0]][best_src[1]]
        if piece:
            board_after[best_src[0]][best_src[1]] = None
            board_after[best_dst[0]][best_dst[1]] = piece
            print(f"    Move: {best_move} ({piece} {best_src}→{best_dst}) [Δ={best_score:.1f}]")
        else:
            print(f"    Move: {best_move} but no piece at {best_src}")
            return None

        return board_after

    # --- Highlight-based detection (most reliable) ---

    def _cell_highlight_score(self, img, r, c):
        """Detect green/yellow highlight at a cell. Returns highlight pixel ratio."""
        px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
        hs = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.45)
        h, w = img.shape[:2]
        x1, y1 = max(0, px - hs), max(0, py - hs)
        x2, y2 = min(w, px + hs), min(h, py + hs)
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        total = patch.shape[0] * patch.shape[1]
        # Green highlight: H 35-85
        green = cv2.inRange(hsv, (35, 40, 80), (85, 255, 255))
        # Yellow-green: H 20-35
        yellow = cv2.inRange(hsv, (20, 40, 80), (35, 255, 255))
        return (cv2.countNonZero(green) + cv2.countNonZero(yellow)) / total

    def detect_move_highlight(self, img, board_after_our_move, fen_after_our_move,
                              our_move=None):
        """Detect opponent's move by finding highlighted cells (green/yellow markers).

        天天象棋 highlights the source and destination of the last move.
        We find those cells and match against legal opponent moves.
        """
        opp_turn = 'b' if self.playing_red else 'w'
        opp_fen = f"{fen_after_our_move} {opp_turn} - - 0 1"
        legal_moves = self.get_legal_moves(opp_fen)
        if not legal_moves:
            return None

        # Compute highlight score for every cell
        hl = {}
        for r in range(10):
            for c in range(9):
                hl[(r, c)] = self._cell_highlight_score(img, r, c)

        # Filter out our own move's highlight cells
        our_cells = set()
        if our_move:
            s, d = self.uci_to_screen_cells(our_move)
            our_cells = {(s[0], s[1]), (d[0], d[1])}

        # Score each legal move by highlight at src+dst
        scored = []
        for move in legal_moves:
            src, dst = self.uci_to_screen_cells(move)
            sk, dk = (src[0], src[1]), (dst[0], dst[1])
            # Skip if this move's cells overlap with our move's highlight
            if sk in our_cells or dk in our_cells:
                scored.append((move, src, dst, 0.0))
                continue
            scored.append((move, src, dst, hl[sk] + hl[dk]))

        scored.sort(key=lambda x: -x[3])
        best = scored[0]

        # Debug: show top highlights and top moves
        top_cells = sorted(hl.items(), key=lambda x: -x[1])[:5]
        top_moves = [(m, f"{s:.3f}") for m, _, _, s in scored[:3]]
        print(f"    Hl cells: {[(f'{r},{c}', f'{s:.3f}') for (r,c),s in top_cells]}")
        print(f"    Hl moves: {top_moves}")

        if best[3] > 0.02:  # At least 2% highlight pixels across both cells
            move, src, dst, score = best
            piece = board_after_our_move[src[0]][src[1]]
            if piece:
                board_result = [row[:] for row in board_after_our_move]
                board_result[src[0]][src[1]] = None
                board_result[dst[0]][dst[1]] = piece
                print(f"    HlMove: {move} ({piece}) [score={score:.3f}]")
                return board_result

        return None

    # --- Single-image occupancy detection (no before/after needed) ---

    def _cell_feature(self, img, r, c):
        """Brightness std at cell center — high for pieces, low for empty cells."""
        px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
        radius = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.25)
        h, w = img.shape[:2]
        x1, y1 = max(0, px - radius), max(0, py - radius)
        x2, y2 = min(w, px + radius), min(h, py + radius)
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    def detect_move_occupancy(self, img, board_after_our_move, fen_after_our_move):
        """Detect opponent's move from a SINGLE image using occupancy analysis.

        Uses brightness std to distinguish pieces (high variance: bright center +
        dark text) from empty cells (low variance: uniform board color).
        Calibrates threshold on-the-fly from known cells.
        """
        opp_turn = 'b' if self.playing_red else 'w'
        opp_fen = f"{fen_after_our_move} {opp_turn} - - 0 1"
        legal_moves = self.get_legal_moves(opp_fen)

        if not legal_moves:
            return None

        # Compute brightness std for every cell
        features = {}
        for r in range(10):
            for c in range(9):
                features[(r, c)] = self._cell_feature(img, r, c)

        # Collect cells that are src/dst of any legal move
        src_cells = set()
        dst_cells = set()
        for move in legal_moves:
            src, dst = self.uci_to_screen_cells(move)
            src_cells.add((src[0], src[1]))
            dst_cells.add((dst[0], dst[1]))

        # Safe references: cells not involved in any legal move
        occ_ref = [features[rc] for rc in
                   [(r, c) for r in range(10) for c in range(9)
                    if board_after_our_move[r][c] is not None and (r, c) not in src_cells]]
        emp_ref = [features[rc] for rc in
                   [(r, c) for r in range(10) for c in range(9)
                    if board_after_our_move[r][c] is None and (r, c) not in dst_cells]]

        if len(occ_ref) < 3 or len(emp_ref) < 3:
            print("    Occ: insufficient ref cells")
            return None

        occ_med = float(np.median(occ_ref))
        emp_med = float(np.median(emp_ref))
        threshold = (occ_med + emp_med) / 2

        if occ_med - emp_med < 5:
            print(f"    Occ: weak separation ({occ_med:.1f} vs {emp_med:.1f})")
            return None

        print(f"    Occ: occ={occ_med:.1f} emp={emp_med:.1f} thr={threshold:.1f}")

        # Score each legal move
        scored = []
        for move in legal_moves:
            src, dst = self.uci_to_screen_cells(move)
            sf = features[(src[0], src[1])]
            df = features[(dst[0], dst[1])]
            # After move: src empty (low std), dst has piece (high std)
            src_empty = max(0, threshold - sf)
            dst_piece = max(0, df - threshold)
            score = src_empty * 2 + dst_piece  # src leaving is clearest signal
            scored.append((move, src, dst, score, sf, df))

        scored.sort(key=lambda x: -x[3])
        top3 = [(m, f"{s:.1f}") for m, _, _, s, _, _ in scored[:3]]
        print(f"    Occ top: {top3}")

        best = scored[0]
        margin_ok = len(scored) < 2 or best[3] > scored[1][3] * 1.3
        if best[3] > 3 and margin_ok:
            move, src, dst, _, sf, df = best
            piece = board_after_our_move[src[0]][src[1]]
            if piece:
                board_result = [row[:] for row in board_after_our_move]
                board_result[src[0]][src[1]] = None
                board_result[dst[0]][dst[1]] = piece
                print(f"    OccMove: {move} ({piece}) sf={sf:.1f} df={df:.1f}")
                return board_result

        print(f"    Occ: no confident move (best={best[3]:.1f})")
        return None

    def detect_move(self, img_before, img_after, board_before):
        """Detect opponent's move by finding which cells changed."""
        board_after = [row[:] for row in board_before]
        changed = []

        for r in range(10):
            for c in range(9):
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                ps = self.patch_size
                h, w = img_before.shape[:2]
                x1, y1 = max(0, px-ps), max(0, py-ps)
                x2, y2 = min(w, px+ps), min(h, py+ps)
                p1 = img_before[y1:y2, x1:x2]
                p2 = img_after[y1:y2, x1:x2]
                if p1.shape != p2.shape:
                    continue
                diff = cv2.absdiff(p1, p2)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                change = np.count_nonzero(gray_diff > 30) / max(1, gray_diff.size)
                if change > 0.03:
                    had_piece = board_before[r][c] is not None
                    has_piece_now = self._cell_has_piece_now(img_after, c, r)
                    changed.append((r, c, change, had_piece, has_piece_now))

        # Re-capture templates from KNOWN piece positions in the new image,
        # then re-parse the full board with fresh templates.
        fresh_templates = {}
        ps = self.patch_size
        for r in range(10):
            for c in range(9):
                piece = board_before[r][c]
                if piece is None:
                    continue
                # Only use pieces that DIDN'T change (still in original position)
                in_changed = any(cr == r and cc == c for cr, cc, *_ in changed)
                if in_changed:
                    continue
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                patch = self._extract(img_after, px, py, ps)
                if patch is not None:
                    if piece not in fresh_templates:
                        fresh_templates[piece] = []
                    fresh_templates[piece].append(patch)

        # Also add empty templates from known empty positions
        fresh_templates['_'] = []
        for r in range(10):
            for c in range(9):
                if board_before[r][c] is not None:
                    continue
                in_changed = any(cr == r and cc == c for cr, cc, *_ in changed)
                if in_changed:
                    continue
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                patch = self._extract(img_after, px, py, ps)
                if patch is not None:
                    fresh_templates['_'].append(patch)
                    if len(fresh_templates['_']) >= 10:
                        break
            if len(fresh_templates.get('_', [])) >= 10:
                break

        # Save old templates, use fresh ones for re-parse
        old_templates = self.templates
        self.templates = fresh_templates

        # Re-parse all cells
        new_board = self.parse_board(img_after)

        # Restore original templates
        self.templates = old_templates

        # Find what changed
        src = dst = None
        for r in range(10):
            for c in range(9):
                old = board_before[r][c]
                new = new_board[r][c]
                if old != new:
                    if old is not None and new is None:
                        src = (r, c, old)
                    elif old is None and new is not None:
                        dst = (r, c, new)
                    elif old is not None and new is not None:
                        src = (r, c, old)
                        dst = (r, c, new)

        if src:
            board_after[src[0]][src[1]] = None
        if dst and src:
            board_after[dst[0]][dst[1]] = src[2]  # moving piece
            print(f"    Move: {src[2]} ({src[0]},{src[1]}) → ({dst[0]},{dst[1]})")
        elif src:
            print(f"    Piece left ({src[0]},{src[1]}) but no dest found")
        else:
            print(f"    No move detected via fresh re-parse")
            # Fallback: just return the fresh parse
            return new_board

        return board_after

    def board_to_fen(self, board):
        if self.playing_red:
            fb = board
        else:
            fb = [row[::-1] for row in reversed(board)]
        parts = []
        for row in fb:
            s, e = "", 0
            for p in row:
                if p is None:
                    e += 1
                else:
                    if e: s += str(e); e = 0
                    s += p
            if e: s += str(e)
            parts.append(s)
        return "/".join(parts)

    # --- Move Execution ---

    def uci_to_logical(self, move):
        fc, fr = ord(move[0]) - ord('a'), int(move[1])
        tc, tr = ord(move[2]) - ord('a'), int(move[3])
        if self.playing_red:
            s = [(fc, 9-fr), (tc, 9-tr)]
        else:
            s = [(8-fc, fr), (8-tc, tr)]
        return [(self.cols_logical[c], self.rows_logical[r]) for c, r in s]

    def load_cnn(self):
        """Load CNN piece classifier if available."""
        try:
            from xiangqi_cnn import PieceClassifierCNN, MODEL_PATH
            if os.path.exists(MODEL_PATH):
                self.cnn = PieceClassifierCNN(MODEL_PATH)
                print("  CNN model loaded!")
                return True
        except Exception as e:
            print(f"  CNN not available: {e}")
        return False

    def parse_board_cnn(self, img):
        """Parse entire board using CNN. Returns 10x9 board array."""
        if not self.cnn:
            return None
        return self.cnn.parse_board(
            img, self.cols_logical, self.rows_logical,
            self.retina_scale, self.win_x, self.win_y,
            self.cell_w, self.cell_h)

    def detect_move_cnn(self, img, board_before, fen_before):
        """Detect opponent's move by CNN board parsing + diff with tracked state.

        Parses the entire board from a single screenshot, compares with
        the tracked board state, and matches differences against legal moves.
        """
        if not self.cnn:
            return None

        parsed = self.parse_board_cnn(img)
        if not parsed:
            return None

        # Find cells that differ between tracked and CNN-parsed board
        diffs = []
        for r in range(10):
            for c in range(9):
                old = board_before[r][c]
                new = parsed[r][c]
                if old != new:
                    diffs.append((r, c, old, new))

        if not diffs:
            return None  # No change detected

        # Match against legal opponent moves
        opp_turn = 'b' if self.playing_red else 'w'
        opp_fen = f"{fen_before} {opp_turn} - - 0 1"
        legal_moves = self.get_legal_moves(opp_fen)

        best_move = None
        best_match = 0
        for move in legal_moves:
            src, dst = self.uci_to_screen_cells(move)
            sr, sc = src[0], src[1]
            dr, dc = dst[0], dst[1]
            match = 0
            # Check if src cell changed from piece to empty/different
            for r, c, old, new in diffs:
                if r == sr and c == sc and old is not None:
                    match += 1
                if r == dr and c == dc:
                    match += 1
            if match > best_match:
                best_match = match
                best_move = move

        if best_move and best_match >= 1:
            src, dst = self.uci_to_screen_cells(best_move)
            piece = board_before[src[0]][src[1]]
            if piece:
                board_result = [row[:] for row in board_before]
                board_result[src[0]][src[1]] = None
                board_result[dst[0]][dst[1]] = piece
                print(f"    CNN: {best_move} ({piece}) [diffs={len(diffs)}, match={best_match}]")
                return board_result

        print(f"    CNN: {len(diffs)} diffs but no legal move match")
        return None

    def collect_cnn_data(self, img, board):
        """Save cell patches for CNN training (auto-labeled from tracked board)."""
        try:
            from xiangqi_cnn import collect_from_screenshot
            session = getattr(self, '_cnn_session', 0)
            n = collect_from_screenshot(
                img, self.cols_logical, self.rows_logical, board,
                self.retina_scale, self.win_x, self.win_y,
                self.cell_w, self.cell_h, session)
            self._cnn_session += 1
        except Exception:
            pass  # Don't let data collection break the game

    def activate_window(self):
        """Bring WeChat to front and focus the specific mini-program window."""
        subprocess.run(['osascript', '-e',
            'tell application "WeChat" to activate'],
            capture_output=True, timeout=2)
        time.sleep(0.3)
        # Click the title bar of the 天天象棋 window to ensure THIS window
        # (not the main WeChat window) has keyboard/mouse focus.
        # Window is at (win_x, win_y), size ~1628x960.
        # Title bar is at y=win_y, roughly 25px tall. Click center.
        title_x = self.win_x + 814
        title_y = self.win_y + 5  # Just inside the title bar
        self._cgevent_click(title_x, title_y)
        time.sleep(0.2)

    def click(self, lx, ly):
        """Click using CGEvent (bypasses pyautogui, more reliable for WebViews)."""
        self._cgevent_click(int(lx), int(ly))

    def _cgevent_click(self, x, y):
        """Low-level click using Quartz CGEvent API."""
        point = Quartz.CGPointMake(x, y)
        # Move mouse to position first
        move = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventMouseMoved, point, 0)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, move)
        time.sleep(0.08)
        # Mouse down
        down = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventLeftMouseDown, point,
            Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
        time.sleep(0.08)
        # Mouse up
        up = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventLeftMouseUp, point,
            Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)

    # --- Pikafish ---

    def pikafish(self, fen, move_history=None, excluded=None):
        proc = subprocess.Popen(
            [PIKAFISH], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=PIKAFISH_DIR)
        if move_history:
            pos_cmd = f"position fen {fen} moves {' '.join(move_history)}\n"
        else:
            pos_cmd = f"position fen {fen}\n"
        # Get all legal moves, exclude repetition moves
        go_cmd = f"go movetime {MOVE_TIME_MS}"
        if excluded:
            legal = self.get_legal_moves(fen)
            allowed = [m for m in legal if m not in excluded]
            if allowed:
                go_cmd = f"go movetime {MOVE_TIME_MS} searchmoves {' '.join(allowed)}"
        try:
            proc.stdin.write(f"uci\nisready\n{pos_cmd}{go_cmd}\n")
            proc.stdin.flush()
        except BrokenPipeError:
            err = proc.stderr.read()
            print(f"  Pikafish crash! stderr: {err[:200]}")
            proc.kill()
            return None, ""
        best, info = None, ""
        t0 = time.time()
        while time.time() - t0 < MOVE_TIME_MS/1000 + 5:
            line = proc.stdout.readline().strip()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            if line.startswith('bestmove'):
                best = line.split()[1] if len(line.split()) > 1 else None
                break
            if 'score' in line:
                info = line
        try: proc.stdin.write("quit\n"); proc.stdin.flush()
        except: pass
        try: proc.wait(timeout=2)
        except: proc.kill()
        return best, info

    def score_str(self, info):
        if 'score cp' in info:
            p = info.split()
            try: return f"{int(p[p.index('cp')+1])/100:+.1f}"
            except: pass
        if 'score mate' in info:
            p = info.split()
            try: return f"M{p[p.index('mate')+1]}"
            except: pass
        return "?"

    # --- Main ---

    def crop_board_region(self, img):
        """Crop the board grid area from the full capture for fast pixel comparison."""
        px0, py0 = self.logical_to_pixel(self.cols_logical[0], self.rows_logical[0])
        px8, py9 = self.logical_to_pixel(self.cols_logical[8], self.rows_logical[9])
        margin = 10
        return img[max(0,py0-margin):py9+margin, max(0,px0-margin):px8+margin].copy()

    def images_changed(self, img1, img2):
        """Compare two cropped board images. Returns True if significantly different."""
        if img1 is None or img2 is None:
            return True
        if img1.shape != img2.shape:
            return True
        diff = cv2.absdiff(img1, img2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
        # Count pixels that changed by more than 30 intensity levels
        changed_pixels = np.count_nonzero(gray_diff > 30)
        total_pixels = gray_diff.size
        change_ratio = changed_pixels / total_pixels
        # If more than 0.5% of pixels changed significantly, board changed
        return change_ratio > 0.005

    def run(self):
        print("=== Xiangqi Bot (Pikafish) ===\n")
        if not os.path.exists(PIKAFISH):
            print(f"ERROR: {PIKAFISH} not found"); sys.exit(1)

        print("[1] Finding window...")
        self.find_window()

        print("[2] Calibration...")
        if not self.load_calibration():
            self.calibrate()

        print("[3] Taking screenshot...")
        img = self.screenshot_for_processing()
        print(f"  Image: {img.shape[1]}x{img.shape[0]}, retina={self.retina_scale:.2f}x")

        print("[4] Loading CNN / detecting orientation...")
        if self.load_cnn():
            board = self.parse_board_cnn(img)
            # Detect orientation from king positions (works for any board state)
            k_row = None
            for r in range(10):
                for c in range(9):
                    if board[r][c] == 'K':
                        k_row = r
            if k_row is not None and k_row <= 4:
                self.playing_red = False
                print(f"  You play: BLACK (red K at row {k_row})")
            else:
                self.playing_red = True
                print(f"  You play: RED (red K at row {k_row})")
        else:
            self.detect_orientation(img)
            self.capture_templates(img)
            board = self.parse_board(img)
            print("  Board parsed by template matching (no CNN)")

        fen = self.board_to_fen(board)
        print(f"\n  FEN: {fen}")

        # Print board
        for r in range(10):
            line = " "
            for c in range(9):
                p = board[r][c]
                line += f" {p}" if p else " ."
            print(line)

        # === CNN-driven game loop ===
        # Simple: poll board with CNN, detect changes, react
        turn = "w" if self.playing_red else "b"
        n = 0
        last_fen = fen
        self._cnn_session = int(time.time())

        fen_history = {}  # fen -> last_move, to avoid repetition
        excluded_moves = []  # moves to exclude if position repeats

        print(f"\n--- Game loop (playing {'RED' if self.playing_red else 'BLACK'}) ---\n")

        while True:
            try:
                # Step 1: Ask pikafish for best move
                full_fen = f"{fen} {turn} - - 0 1"

                # Check for repetition
                if fen in fen_history:
                    excluded_moves.append(fen_history[fen])
                    excl_str = ' '.join(set(excluded_moves))
                    print(f"  Repeat! Excluding: {excl_str}")

                print(f"  FEN → Pikafish: {full_fen}")
                best, info = self.pikafish(full_fen, excluded=excluded_moves)

                if not best or best == '(none)':
                    # Try without exclusions
                    if excluded_moves:
                        print("  No move with exclusions, trying without...")
                        excluded_moves = []
                        best, info = self.pikafish(full_fen)

                if not best or best == '(none)':
                    print(f"  No move! Re-parsing...")
                    time.sleep(2)
                    img = self.screenshot_for_processing()
                    board = self.parse_board_cnn(img) if self.cnn else self.parse_board(img)
                    fen = self.board_to_fen(board)
                    continue

                fen_history[fen] = best  # Track this move for this position
                n += 1
                sc = self.score_str(info)
                print(f"[{n}] {best} ({sc})")

                # Step 2: Click our move (with retry)
                pts = self.uci_to_logical(best)
                before_crop = self.crop_board_region(self.screenshot_for_processing())

                click_ok = False
                for click_try in range(3):
                    self.activate_window()
                    time.sleep(0.3)
                    self.click(pts[0][0], pts[0][1])
                    time.sleep(0.8)
                    self.click(pts[1][0], pts[1][1])
                    time.sleep(1.0)
                    check = self.crop_board_region(self.screenshot_for_processing())
                    if self.images_changed(before_crop, check):
                        click_ok = True
                        break
                    if click_try < 2:
                        print(f"  Click retry {click_try+1}...")

                if not click_ok:
                    # Click failed — might not be our turn or illegal move
                    # Wait for board to change, then re-parse
                    print("  Click failed — waiting for board change...")
                    ref = self.crop_board_region(self.screenshot_for_processing())
                    for wi in range(60):
                        time.sleep(0.5)
                        curr = self.screenshot_for_processing()
                        if self.images_changed(ref, self.crop_board_region(curr)):
                            time.sleep(1.5)
                            break
                        if wi % 20 == 0 and wi > 0:
                            sys.stdout.write(".")
                            sys.stdout.flush()
                    # Re-parse entire board with CNN
                    img = self.screenshot_for_processing()
                    board = self.parse_board_cnn(img) if self.cnn else self.parse_board(img)
                    fen = self.board_to_fen(board)
                    print(f"  Re-parsed → {fen}")
                    continue

                # Step 3: Wait until board fully settles
                # (our animation + opponent response + their animation)
                # Don't separate phases — just wait for stable board
                print("  Waiting...", end="", flush=True)
                time.sleep(1.0)  # Brief initial wait for our click to register
                last_change = time.time()
                prev_check = self.crop_board_region(self.screenshot_for_processing())
                while time.time() - last_change < 30:  # Max 30s since last change
                    time.sleep(0.4)
                    curr_check = self.crop_board_region(self.screenshot_for_processing())
                    if self.images_changed(prev_check, curr_check):
                        last_change = time.time()  # Reset timer on any change
                        sys.stdout.write("~")
                    else:
                        # Board stable — but wait at least 2s since last change
                        if time.time() - last_change >= 2.0:
                            break
                    prev_check = curr_check
                    sys.stdout.flush()
                print(" done")

                # Step 4: Re-parse entire board with CNN (no tracking needed!)
                img = self.screenshot_for_processing()
                board = self.parse_board_cnn(img) if self.cnn else self.parse_board(img)
                fen = self.board_to_fen(board)
                print(f"  Board → {fen}")

            except KeyboardInterrupt:
                print("\nStopped."); break
            except Exception as e:
                print(f"\nErr: {e}")
                import traceback; traceback.print_exc()
                time.sleep(2)


if __name__ == '__main__':
    Bot().run()
