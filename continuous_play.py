#!/usr/bin/env python3
"""
continuous_play.py — Robust multi-game supervisor for the xiangqi bot.

Wraps the existing `Bot` from xiangqi_bot.py in a headless CLI supervisor
that plays games back-to-back:

  1. Recover to a playable board (tiered fallback: templates → blind coords → ESC → re-find window)
  2. Run one game via `Bot.run()` in a thread
  3. Detect end-of-game via OR'd signals (pikafish (none), FEN stale, king missing, board invisible)
  4. Loop to step 1

Does NOT modify xiangqi_bot.py or app.py. If this script misbehaves,
fall back to:
  python3 xiangqi_bot.py    # single-game CLI
  python3 app.py            # single-game GUI

Subcommands:
  python3 continuous_play.py play [--max-games N] [--recovery-timeout S]
  python3 continuous_play.py test-templates
  python3 continuous_play.py detect-end
  python3 continuous_play.py diag
  python3 continuous_play.py crop-templates   (manual Preview.app workflow printed)
"""

import argparse
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import Quartz

from xiangqi_bot import Bot

# ============================================================
# Constants
# ============================================================

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(REPO_DIR, "templates")
DEBUG_ROOT = os.path.join(REPO_DIR, "continuous_play_debug")

RECOVERY_TIMEOUT_S = 120
GAME_TIMEOUT_S = 30 * 60
POLL_INTERVAL_S = 2.0
GRACE_AFTER_START_S = 15.0

OPPONENT_STUCK_S = 150.0  # last-resort safety net — primary detection uses templates
GAME_END_TEMPLATE_THRESHOLD = 0.95  # strict for detection to avoid mid-game false positives

BUTTON_COOLDOWN_S = 30.0

TEMPLATE_THRESHOLDS = (0.92, 0.88)  # stricter — avoid false positives on start/play screens
TEMPLATE_SCALES = (1.0, 0.9, 1.1, 0.85, 1.15, 0.95, 1.05)

ESC_KEYCODE = 53  # macOS Escape virtual keycode

# Blind-click relative coordinates used as Tier-2 fallback.
# Format: (label, rel_x, rel_y, settle_seconds).
# Inherited from app.py:_recovery_click_pass (which ships in the current bot)
# plus new coordinates for 再来一局 and 切换对手 (which the original list missed).
BLIND_CLICKS = [
    # Ordered by most-likely-to-succeed first.
    ("play_again(right_col_bottom)", 0.584, 0.842, 0.6),
    ("switch_opponent(left_col)",    0.442, 0.842, 0.6),
    ("popup_close_x_in_popup",       0.672, 0.272, 0.6),
    ("popup_close_x(top_right)",     0.93, 0.11, 0.5),
    ("popup_close_x(top_left)",      0.08, 0.10, 0.5),
    ("main_button(bottom_center)",   0.50, 0.84, 0.6),
    ("sub_button(mid_bottom)",       0.50, 0.74, 0.6),
    ("continue(mid)",                0.50, 0.64, 0.6),
    ("confirm(lower)",               0.50, 0.91, 0.6),
]

# Template inventory. Priority order inside a recovery pass.
# Script works fine with missing templates — each is optional.
# IMPORTANT: btn_start_game is FIRST because it's the most specific match
# (1.00 on start screens, near-zero on others). Catching it first
# short-circuits false positives from other templates on start-screen
# backgrounds.
TEMPLATE_ACTIONS = [
    # (template_name, label_for_log, settle_s)
    # btn_start_game FIRST so it short-circuits false positives from the
    # smaller/more generic templates on the start-screen background.
    ("btn_start_game",       "开始",         1.5),   # puzzle/endgame start
    ("btn_confirm",          "确定",         1.5),   # level-up popup (恭喜升级)
    ("btn_play_again",       "再来一局",     1.2),   # result dialog + ticket refund
    ("btn_close_x_top",      "X close",     2.0),   # 54x54 tight; popups close animations are slow
    ("btn_switch_opponent",  "切换对手",     1.0),   # fallback if 再来一局 blocked
]

# Detection-only templates (presence is a signal, never clicked).
DETECTION_TEMPLATES = [
    "popup_end_banner",
    "popup_ad_banner",
]


# ============================================================
# Helpers
# ============================================================

def log(tag, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}", flush=True)


def _send_escape_key():
    """Post ESC key down + up via Quartz."""
    try:
        down = Quartz.CGEventCreateKeyboardEvent(None, ESC_KEYCODE, True)
        up = Quartz.CGEventCreateKeyboardEvent(None, ESC_KEYCODE, False)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
        time.sleep(0.05)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)
    except Exception as e:
        log("key", f"ESC send failed: {e}")


def _quiet(fn, *args, **kwargs):
    """Run fn with stdout suppressed. Returns its value or None on error."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None
    finally:
        sys.stdout = old


def _board_visible(bot):
    """Return (visible: bool, board | None, piece_count, red_K, black_k, avg_conf).

    Equivalent to app.py:_board_state — re-implemented freshly here.
    """
    if not bot.cols_logical or not bot.rows_logical:
        return False, None, 0, 0, 0, 0.0
    try:
        img = bot.screenshot_for_processing()
        board = _quiet(bot.parse_board_cnn, img)
    except Exception:
        return False, None, 0, 0, 0, 0.0
    if not board:
        return False, None, 0, 0, 0, 0.0

    pieces = [p for row in board for p in row if p is not None]
    pc = len(pieces)
    rk = sum(1 for p in pieces if p == 'K')
    bk = sum(1 for p in pieces if p == 'k')

    confs = []
    cell_probs = getattr(bot.cnn, "_cell_probs", None)
    if cell_probs:
        for r in range(10):
            for c in range(9):
                if board[r][c] is None or cell_probs[r][c] is None:
                    continue
                confs.append(float(cell_probs[r][c].max()))
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0

    visible = (pc >= 6 and rk == 1 and bk == 1 and avg_conf >= 0.55)
    return visible, board, pc, rk, bk, avg_conf


def _ensure_bot_ready(bot):
    """Bring the bot up to the point where it can read the board. Quiet."""
    try:
        bot.find_window()
    except Exception as e:
        log("ready", f"find_window failed: {e}")
        return False
    if not _quiet(bot.load_cnn):
        log("ready", "load_cnn failed")
        return False
    if bot.cols_logical and bot.rows_logical:
        return True
    if _quiet(bot.load_calibration):
        return True
    # Try auto-calibrate
    try:
        img = bot.screenshot_for_processing()
        return bool(_quiet(bot.auto_calibrate, img))
    except Exception as e:
        log("ready", f"auto_calibrate failed: {e}")
        return False


# ============================================================
# Template matching
# ============================================================

class TemplateMatcher:
    """Multi-scale, stepped-threshold matcher. Returns retina-pixel coords.

    Matching is done at HALF resolution (0.5x downscale) for speed — on a
    3748x2198 screenshot, full-res matching takes ~100-200ms per template-scale;
    half-res is ~25-50ms. Coordinates are scaled back to full resolution
    before returning.
    """

    _missing_warned = set()  # class-level to warn once per file per process
    MATCH_DOWNSCALE = 0.35    # ← key speedup knob (0.35 ~= 8x faster than full)

    def __init__(self, templates_dir):
        self.templates_dir = templates_dir
        self._cache_full = {}  # name -> full-res grayscale ndarray
        self._cache_small = {} # name -> downscaled grayscale ndarray

    def _load(self, name):
        """Load template at both full and downscaled resolution."""
        if name in self._cache_full:
            return self._cache_full[name]
        path = os.path.join(self.templates_dir, f"{name}.png")
        if not os.path.exists(path):
            if name not in TemplateMatcher._missing_warned:
                log("tm", f"template missing: {name}.png (ok — graceful)")
                TemplateMatcher._missing_warned.add(name)
            self._cache_full[name] = None
            self._cache_small[name] = None
            return None
        tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            log("tm", f"cv2.imread failed for {path}")
            self._cache_full[name] = None
            self._cache_small[name] = None
            return None
        self._cache_full[name] = tpl
        # Pre-compute downscaled version
        nh = max(1, int(tpl.shape[0] * self.MATCH_DOWNSCALE))
        nw = max(1, int(tpl.shape[1] * self.MATCH_DOWNSCALE))
        self._cache_small[name] = cv2.resize(tpl, (nw, nh), interpolation=cv2.INTER_AREA)
        return tpl

    def _downscale_gray(self, img_bgr):
        """Convert BGR → gray → downscaled gray. Slow step, do it once per scan."""
        gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ds = self.MATCH_DOWNSCALE
        small_w = int(gray_full.shape[1] * ds)
        small_h = int(gray_full.shape[0] * ds)
        return cv2.resize(gray_full, (small_w, small_h), interpolation=cv2.INTER_AREA)

    def find(self, img_bgr, name,
             thresholds=TEMPLATE_THRESHOLDS,
             scales=TEMPLATE_SCALES,
             precomputed_gray_small=None):
        """Return (px_center, py_center, score, scale) in RETINA pixel coords.

        `precomputed_gray_small` lets the caller pass a pre-downscaled
        grayscale image to avoid redundant work across multiple find() calls
        on the same screenshot.
        """
        if self._load(name) is None:
            return None
        if img_bgr is None and precomputed_gray_small is None:
            return None
        tpl_small = self._cache_small[name]

        if precomputed_gray_small is not None:
            gray_small = precomputed_gray_small
        else:
            gray_small = self._downscale_gray(img_bgr)

        best = None  # (score, scale, x, y, w, h) — in DOWNSCALED space
        for scale in scales:
            if scale == 1.0:
                t = tpl_small
            else:
                nh = max(1, int(tpl_small.shape[0] * scale))
                nw = max(1, int(tpl_small.shape[1] * scale))
                if nh > gray_small.shape[0] or nw > gray_small.shape[1]:
                    continue
                t = cv2.resize(tpl_small, (nw, nh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(gray_small, t, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
            if best is None or maxv > best[0]:
                best = (maxv, scale, maxloc[0], maxloc[1], t.shape[1], t.shape[0])
            # Early exit: if we already have a very high score, don't
            # bother checking other scales — saves ~6x template matches.
            if maxv >= 0.97:
                break
        if best is None:
            return None
        # Apply stepped thresholds
        score, scale, x, y, w, h = best
        accepted = False
        for thr in thresholds:
            if score >= thr:
                accepted = True
                break
        if not accepted:
            return None
        # Scale coordinates back to full resolution
        scale_back = 1.0 / self.MATCH_DOWNSCALE
        cx = int((x + w / 2) * scale_back)
        cy = int((y + h / 2) * scale_back)
        return (cx, cy, float(score), float(scale))

    def match_to_logical(self, bot, match):
        """Convert retina-pixel center to logical screen coords for bot.click()."""
        cx_px, cy_px, _, _ = match
        lx = bot.win_x + cx_px / max(bot.retina_scale, 0.0001)
        ly = bot.win_y + cy_px / max(bot.retina_scale, 0.0001)
        return lx, ly


# ============================================================
# Click verifier
# ============================================================

class ClickVerifier:
    def __init__(self, bot):
        self.bot = bot

    def click_and_verify(self, lx, ly, label, settle_s=0.6):
        """Click at (lx, ly), return True iff the board crop looks different."""
        try:
            before = self.bot.screenshot_for_processing()
        except Exception as e:
            log("click", f"before-screenshot failed: {e}")
            return False
        try:
            # Refresh window before clicking — handles any window movement
            self.bot.find_window()
            self.bot.activate_window()
            time.sleep(0.2)
            self.bot.click(lx, ly)
        except Exception as e:
            log("click", f"click call failed: {e}")
            return False
        # Split settle into two chunks: a short one for immediate check,
        # then a longer one if no change yet (popup close animations can
        # take 1.5+ seconds to complete).
        time.sleep(min(settle_s, 0.6))
        try:
            after = self.bot.screenshot_for_processing()
        except Exception:
            return False
        changed = self.bot.images_changed(before, after)
        if not changed and settle_s > 0.6:
            # Give the animation more time and re-check
            time.sleep(settle_s - 0.6)
            try:
                after = self.bot.screenshot_for_processing()
            except Exception:
                return False
            changed = self.bot.images_changed(before, after)
        if changed:
            log("click", f"{label} at ({int(lx)},{int(ly)}) → screen changed ✓")
        else:
            log("click", f"{label} at ({int(lx)},{int(ly)}) → no change ✗")
        return changed


# ============================================================
# Game-end detector
# ============================================================

class GameEndDetector:
    """Detect end of game via template matching (primary) + stuck-timer fallback.

    Uses strict 0.95 threshold on template matches to minimize mid-game false
    positives. The old FEN/king/board-visible signals were too CNN-dependent
    and fired during long opponent thinks or board-rendering hiccups.
    """

    # Templates whose presence indicates a game-end screen.
    END_TEMPLATES = ("popup_end_banner", "popup_level_up", "popup_badge_earned",
                     "btn_play_again", "btn_close_x_top", "btn_switch_opponent",
                     "btn_confirm")

    def __init__(self, tm):
        self.tm = tm
        self.reset()

    def reset(self):
        self.last_pikafish_best = None
        self.not_my_turn_since = None  # timestamp when opponent's turn started

    def mark_pikafish_result(self, best_move):
        self.last_pikafish_best = best_move

    def update(self, bot):
        """Poll current state. Return (ended: bool, reason: str)."""
        now = time.time()

        # Signal 1: Pikafish returned (none) — mate or stalemate, definitive.
        if self.last_pikafish_best == "(none)":
            return True, "pikafish_none"

        # Signal 2: template-based popup detection.
        # STRICT 0.95 threshold — we need high confidence to abort a live
        # game. Any of the end-screen templates matching strongly is
        # effectively a certainty that we're looking at a result popup.
        try:
            img = bot.screenshot_for_processing()
            gray_small = self.tm._downscale_gray(img)
            for name in self.END_TEMPLATES:
                m = self.tm.find(img, name,
                                 thresholds=(GAME_END_TEMPLATE_THRESHOLD,),
                                 precomputed_gray_small=gray_small)
                if m is not None:
                    return True, f"tpl:{name}({m[2]:.2f})"
        except Exception:
            pass

        # Signal 3: last-resort safety net — opponent hasn't moved in 5 min.
        # 天天象棋 games rarely have >5 min opponent thinks outside of
        # disconnects / timeouts.
        try:
            mine = bot.is_my_turn()
        except Exception:
            mine = True  # on error, assume it's our turn (no false-positive)
        if not mine:
            if self.not_my_turn_since is None:
                self.not_my_turn_since = now
            elapsed = now - self.not_my_turn_since
            if elapsed > OPPONENT_STUCK_S:
                return True, f"opponent_stuck({int(elapsed)}s)"
        else:
            self.not_my_turn_since = None

        return False, ""


# ============================================================
# Debug dumper
# ============================================================

class DebugDumper:
    def __init__(self):
        self.base = DEBUG_ROOT

    def save(self, bot, reason, extra=""):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ddir = os.path.join(self.base, ts)
        os.makedirs(ddir, exist_ok=True)
        try:
            img = bot.screenshot_for_processing()
            cv2.imwrite(os.path.join(ddir, "screen.png"), img)
        except Exception as e:
            extra += f"\nscreenshot_failed: {e}"
        try:
            with open(os.path.join(ddir, "state.log"), "w") as f:
                f.write(f"reason: {reason}\n")
                f.write(f"win_x={getattr(bot, 'win_x', '?')} "
                        f"win_y={getattr(bot, 'win_y', '?')} "
                        f"retina={getattr(bot, 'retina_scale', '?')}\n")
                f.write(f"cols_logical={getattr(bot, 'cols_logical', None)}\n")
                f.write(f"rows_logical={getattr(bot, 'rows_logical', None)}\n")
                if extra:
                    f.write("\n--- extra ---\n")
                    f.write(extra)
        except Exception as e:
            log("dump", f"state.log write failed: {e}")
        log("dump", f"saved {ddir}")
        return ddir


# ============================================================
# Recovery orchestrator
# ============================================================

class RecoveryOrchestrator:
    def __init__(self, bot, tm, verifier, dumper, should_stop=None):
        self.bot = bot
        self.tm = tm
        self.verifier = verifier
        self.dumper = dumper
        self.should_stop = should_stop or (lambda: False)
        self._cooldown = {}  # label -> expiry_ts

    def _cooled(self, label):
        now = time.time()
        # Clean expired
        for k in [k for k, v in self._cooldown.items() if v <= now]:
            self._cooldown.pop(k, None)
        return label in self._cooldown

    def _cool(self, label, seconds=BUTTON_COOLDOWN_S):
        self._cooldown[label] = time.time() + seconds

    def _tier1_template_clicks(self, min_score=None):
        """Try each template in priority order, click the FIRST match
        and return.  Taking one screenshot and then clicking multiple
        buttons is dangerous because later templates may false-match on
        the same frame — we break as soon as we successfully click.
        If min_score is given, use it as a stricter single threshold
        (for pre-game safety to avoid false positives on a live board).
        """
        if self.should_stop(): return False
        # Refresh window geometry — the window may have moved since last pass.
        try:
            self.bot.find_window()
        except Exception as e:
            log("tier1", f"find_window failed: {e}")
            return False
        try:
            img = self.bot.screenshot_for_processing()
        except Exception as e:
            log("tier1", f"screenshot failed: {e}")
            return False
        # Pre-downscale the screenshot once — ALL template matches in this
        # pass share the same downscaled grayscale. Huge speedup.
        gray_small = self.tm._downscale_gray(img)
        thresholds = (min_score,) if min_score is not None else TEMPLATE_THRESHOLDS
        # Score everything first, then pick the highest-priority match.
        for name, label, settle in TEMPLATE_ACTIONS:
            if self.should_stop(): return False
            if self._cooled(label):
                continue
            m = self.tm.find(img, name, thresholds=thresholds,
                             precomputed_gray_small=gray_small)
            if m is None:
                continue
            lx, ly = self.tm.match_to_logical(self.bot, m)
            log("tier1", f"match {name} score={m[2]:.2f} scale={m[3]:.2f} → click")
            ok = self.verifier.click_and_verify(lx, ly, f"tpl:{label}", settle_s=settle)
            if ok:
                # Success means the screen changed. Clear cooldowns so
                # previously-blocked templates (e.g. btn_play_again behind
                # an overlay) get a fresh chance on the next pass.
                self._cooldown.clear()
                return True
            self._cool(label)
        return False

    def pregame_template_scan(self, max_passes=5, min_score=0.92):
        """Safe pre-game template scan. Runs Tier 1 only (no blind clicks,
        no ESC, no refind) with a stricter threshold so false positives on
        a live board are near-zero.

        IMPORTANT: Template scan runs FIRST on every pass. `_board_visible()`
        alone is not trustworthy here — the CNN may happily parse a puzzle
        background into a "visible" board while the 开始 overlay is still
        blocking play. We only declare ready when no template matches.
        Happy path (nothing visible to click): 1 pass, ~0.5s.
        """
        log("pregame", f"starting (max_passes={max_passes}, min_score={min_score})")
        self._cooldown.clear()
        for i in range(max_passes):
            if self.should_stop(): return False
            clicked = self._tier1_template_clicks(min_score=min_score)
            if clicked:
                log("pregame", f"pass {i+1}: clicked, waiting for UI transition")
                for _ in range(15):
                    if self.should_stop(): return False
                    time.sleep(0.1)
                continue  # re-scan; another overlay may appear
            # Nothing matched — happy path. Return immediately without
            # verifying _board_visible (too slow on a noisy puzzle
            # background, and bot.run() does its own stabilization anyway).
            log("pregame", f"pass {i+1}: no high-confidence matches, proceed")
            return True
        log("pregame", f"exhausted {max_passes} passes")
        return True

    def _tier2_blind_clicks(self):
        """Click the hardcoded relative coords. Return True if any actually changed screen."""
        if self.should_stop(): return False
        # Refresh window geometry each pass
        try:
            self.bot.find_window()
        except Exception as e:
            log("tier2", f"find_window failed: {e}")
            return False
        ww = max(1, int(self._window_width()))
        wh = max(1, int(self._window_height()))
        clicked_any = False
        for label, rx, ry, settle in BLIND_CLICKS:
            if self.should_stop(): return clicked_any
            if self._cooled(label):
                continue
            lx = self.bot.win_x + rx * ww
            ly = self.bot.win_y + ry * wh
            ok = self.verifier.click_and_verify(lx, ly, f"blind:{label}", settle_s=settle)
            if ok:
                clicked_any = True
                # Early check — if board came back we don't need to spam more clicks
                visible, *_ = _board_visible(self.bot)
                if visible:
                    return True
            else:
                self._cool(label, seconds=15)  # shorter cooldown for blind
        return clicked_any

    def _tier3_escape(self):
        for i in range(3):
            _send_escape_key()
            time.sleep(0.4)
        time.sleep(0.6)
        log("tier3", "ESC × 3 sent")
        return True

    def _tier4_refind(self):
        try:
            self.bot.find_window()
            self.bot.activate_window()
            log("tier4", "find_window + activate_window OK")
            return True
        except Exception as e:
            log("tier4", f"failed: {e}")
            return False

    def _window_width(self):
        try:
            windows = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
            for w in windows:
                if w.get("kCGWindowNumber") == self.bot.win_id:
                    return int(w["kCGWindowBounds"]["Width"])
        except Exception:
            pass
        return 1628

    def _window_height(self):
        try:
            windows = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
            for w in windows:
                if w.get("kCGWindowNumber") == self.bot.win_id:
                    return int(w["kCGWindowBounds"]["Height"])
        except Exception:
            pass
        return 960

    def recover_to_next_game(self, deadline_ts):
        """Tiered recovery. Return True if board becomes visible in time."""
        log("recover", "starting recovery")
        # Short settle delay — the popup animation usually fades in within
        # a few hundred ms. Keep it short so game start is fast.
        for _ in range(5):
            if self.should_stop(): return False
            time.sleep(0.1)

        # First: make sure we can even read the board
        if self.should_stop(): return False
        if not _ensure_bot_ready(self.bot):
            if self.should_stop(): return False
            if not self._tier4_refind():
                time.sleep(1.5)
            _ensure_bot_ready(self.bot)

        visible, *_ = _board_visible(self.bot)
        if visible:
            log("recover", "board already visible — nothing to do")
            return True

        round_no = 0
        while time.time() < deadline_ts:
            if self.should_stop():
                log("recover", "stop requested, exiting recovery")
                return False
            round_no += 1
            log("recover", f"round {round_no}")

            # Tier 1: templates
            if self._tier1_template_clicks():
                if self._ensure_and_check(): return True
            if self.should_stop(): return False

            # Tier 2: blind
            if self._tier2_blind_clicks():
                if self._ensure_and_check(): return True
            if self.should_stop(): return False

            # Tier 3: ESC
            self._tier3_escape()
            if self._ensure_and_check(): return True
            if self.should_stop(): return False

            # Tier 4: re-find window
            self._tier4_refind()
            if self._ensure_and_check(): return True
            if self.should_stop(): return False

            # Sleep in small chunks so Ctrl+C is responsive
            for _ in range(10):
                if self.should_stop(): return False
                time.sleep(0.1)

        self.dumper.save(self.bot, "recovery_timeout",
                         extra=f"cooldown={self._cooldown}")
        return False

    def _ensure_and_check(self):
        _ensure_bot_ready(self.bot)
        visible, *_ = _board_visible(self.bot)
        if visible:
            log("recover", "board visible ✓")
        return visible


# ============================================================
# Continuous supervisor
# ============================================================

class ContinuousSupervisor:
    def __init__(self, max_games=None, recovery_timeout_s=RECOVERY_TIMEOUT_S):
        self.running = True
        self.max_games = max_games
        self.recovery_timeout_s = recovery_timeout_s
        self.current_bot = None
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, signum, frame):
        if self.running:
            log("sig", "SIGINT received, stopping — auto force-exit in 4s")
            self.running = False
            if self.current_bot is not None:
                self.current_bot.stop_flag = True
            # Hard-force exit after 4 seconds regardless of where we're stuck.
            # A daemon thread sleeping then calling os._exit is immune to any
            # subprocess blocks or stuck loops in the main thread.
            def _force_exit():
                time.sleep(4.0)
                print("[force exit] still running — os._exit(130)", flush=True)
                os._exit(130)
            threading.Thread(target=_force_exit, daemon=True).start()
        else:
            # Second Ctrl+C: immediate force-exit
            print("[sig] second SIGINT — force exit", flush=True)
            os._exit(130)

    def run(self):
        log("sup", "continuous supervisor starting")
        log("sup", f"templates dir: {TEMPLATES_DIR}")
        log("sup", f"debug dir: {DEBUG_ROOT}")
        os.makedirs(DEBUG_ROOT, exist_ok=True)

        game_no = 0
        while self.running:
            if self.max_games is not None and game_no >= self.max_games:
                log("sup", f"reached --max-games={self.max_games}, stopping")
                break

            game_no += 1
            log("sup", f"=== game {game_no} ===")

            bot = Bot()
            self.current_bot = bot
            tm = TemplateMatcher(TEMPLATES_DIR)
            verifier = ClickVerifier(bot)
            dumper = DebugDumper()
            orch = RecoveryOrchestrator(
                bot, tm, verifier, dumper,
                should_stop=lambda: not self.running,
            )
            detector = GameEndDetector(tm)

            # Recovery phase.
            # On the FIRST game we run only the safe template-only scan
            # (pregame_template_scan) — it clicks 开始/再来一局 if visible,
            # but won't do blind clicks or ESC that could disrupt a live
            # game.  On subsequent games we run the full tiered recovery.
            if game_no > 1:
                deadline = time.time() + self.recovery_timeout_s
                if not orch.recover_to_next_game(deadline):
                    log("sup", f"game {game_no}: recovery failed, sleeping 3s")
                    time.sleep(3.0)
                    continue
            else:
                log("sup", "first game: pre-game template scan")
                orch.pregame_template_scan(max_passes=3, min_score=0.92)

            if not self.running:
                break

            # Play phase
            self._play_one_game(bot, detector, dumper, game_no)

            self.current_bot = None
            if not self.running:
                break
            time.sleep(1.0)

        log("sup", "supervisor exit")

    def _play_one_game(self, bot, detector, dumper, game_no):
        bot.stop_flag = False
        game_start = time.time()
        end_flag = {"ended": False, "reason": ""}

        # Run bot.run() in a thread. We capture stdout through a pipe so the
        # existing print-based debug logs don't spam terminal; we still forward.
        def _target():
            try:
                bot.run()
            except Exception as e:
                log("game", f"bot.run() raised: {e}")
                traceback.print_exc()

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        log("game", f"game {game_no} thread started")

        grace_until = time.time() + GRACE_AFTER_START_S

        while self.running and t.is_alive():
            # Interruptible sleep
            for _ in range(int(POLL_INTERVAL_S * 10)):
                if not self.running or not t.is_alive():
                    break
                time.sleep(0.1)
            if not self.running or not t.is_alive():
                break
            if time.time() < grace_until:
                continue
            if time.time() - game_start > GAME_TIMEOUT_S:
                log("game", f"game {game_no} watchdog: > {GAME_TIMEOUT_S}s, forcing stop")
                dumper.save(bot, "game_watchdog_timeout")
                bot.stop_flag = True
                break
            ended, reason = detector.update(bot)
            if ended:
                log("game", f"game {game_no} ended: {reason}")
                # Auto-save the end-screen screenshot for future template cropping
                try:
                    snap_dir = os.path.join(REPO_DIR, "end_snapshots")
                    os.makedirs(snap_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_reason = reason.replace("/", "_").replace(" ", "_")
                    path = os.path.join(snap_dir, f"end_{ts}_{safe_reason}.png")
                    cv2.imwrite(path, bot.screenshot_for_processing())
                    log("game", f"saved end-screen snapshot: {path}")
                except Exception as e:
                    log("game", f"snap save failed: {e}")
                bot.stop_flag = True
                break

        # Wait for bot thread to exit cleanly
        t.join(timeout=10.0)
        if t.is_alive():
            log("game", f"game {game_no} thread did not exit — abandoning Bot instance")
            dumper.save(bot, "game_thread_stuck")
        else:
            log("game", f"game {game_no} finished")


# ============================================================
# Subcommands
# ============================================================

def cmd_play(args):
    sup = ContinuousSupervisor(
        max_games=args.max_games,
        recovery_timeout_s=args.recovery_timeout,
    )
    sup.run()
    return 0


def cmd_test_templates(args):
    bot = Bot()
    if not _ensure_bot_ready(bot):
        print("ERROR: bot not ready. Make sure WeChat 天天象棋 window is open.")
        return 2
    img = bot.screenshot_for_processing()
    tm = TemplateMatcher(TEMPLATES_DIR)
    names = [n for n, _, _ in TEMPLATE_ACTIONS] + DETECTION_TEMPLATES
    print(f"Testing {len(names)} templates against current screen:")
    for name in names:
        m = tm.find(img, name,
                    thresholds=(0.50,),   # report any score
                    scales=TEMPLATE_SCALES)
        if m is None:
            # Could be missing file, or genuinely nothing on screen.
            path = os.path.join(TEMPLATES_DIR, f"{name}.png")
            if not os.path.exists(path):
                print(f"  {name:25s}  (template file missing)")
            else:
                print(f"  {name:25s}  no match")
        else:
            cx, cy, score, scale = m
            lx, ly = tm.match_to_logical(bot, m)
            tag = "✓" if score >= 0.82 else "?"
            print(f"  {name:25s}  {tag} score={score:.2f} scale={scale:.2f} "
                  f"px=({cx},{cy}) logical=({int(lx)},{int(ly)})")
    return 0


def cmd_detect_end(args):
    bot = Bot()
    if not _ensure_bot_ready(bot):
        print("ERROR: bot not ready.")
        return 2
    tm = TemplateMatcher(TEMPLATES_DIR)
    detector = GameEndDetector(tm)
    # Run two updates so the stale-signal counters bump
    ended, reason = detector.update(bot)
    time.sleep(1.0)
    ended, reason = detector.update(bot)

    visible, board, pc, rk, bk, conf = _board_visible(bot)
    print(f"board visible: {visible}")
    print(f"pieces={pc} red_K={rk} black_k={bk} avg_conf={conf:.2f}")
    print(f"end-detector: ended={ended} reason={reason or '(none)'}")

    # Also test detection templates one-shot
    try:
        img = bot.screenshot_for_processing()
        for name in DETECTION_TEMPLATES:
            m = tm.find(img, name)
            print(f"  {name}: {'match ' + str(m) if m else 'no match'}")
    except Exception as e:
        print(f"screenshot failed: {e}")
    return 0


def cmd_diag(args):
    bot = Bot()
    try:
        bot.find_window()
    except Exception as e:
        print(f"find_window failed: {e}")
        return 2
    print(f"win_id={bot.win_id}")
    print(f"win_x={bot.win_x} win_y={bot.win_y}")
    try:
        img = bot.screenshot_for_processing()
    except Exception as e:
        print(f"screenshot failed: {e}")
        return 2
    print(f"screenshot shape: {img.shape}")
    print(f"retina_scale: {bot.retina_scale}")
    os.makedirs(DEBUG_ROOT, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ddir = os.path.join(DEBUG_ROOT, f"diag_{ts}")
    os.makedirs(ddir, exist_ok=True)
    cv2.imwrite(os.path.join(ddir, "screen.png"), img)
    with open(os.path.join(ddir, "info.json"), "w") as f:
        json.dump({
            "win_id": bot.win_id,
            "win_x": bot.win_x,
            "win_y": bot.win_y,
            "retina_scale": bot.retina_scale,
            "shape": list(img.shape),
        }, f, indent=2)
    print(f"dumped → {ddir}")

    ok_cnn = _quiet(bot.load_cnn)
    print(f"load_cnn: {ok_cnn}")
    ok_cal = _quiet(bot.load_calibration)
    print(f"load_calibration: {ok_cal}")
    if ok_cnn and ok_cal:
        visible, board, pc, rk, bk, conf = _board_visible(bot)
        print(f"board visible: {visible} (pc={pc} rk={rk} bk={bk} conf={conf:.2f})")
    return 0


def cmd_crop_templates(args):
    print("=" * 60)
    print("Manual cropping workflow (Preview.app)")
    print("=" * 60)
    print()
    print("The script auto-detects templates in:")
    print(f"  {TEMPLATES_DIR}")
    print()
    print("To create them manually (5 minutes total):")
    print()
    print("For each template below:")
    print("  1. Open the source PNG in Preview.app")
    print("  2. Press K to get rectangular selection")
    print("  3. Drag a box around the button (tight, 2-3px padding)")
    print("  4. Cmd+C to copy")
    print("  5. Cmd+N to make new image from clipboard")
    print("  6. Cmd+Shift+S to save, pick PNG format,")
    print("     save to the templates directory above with the exact name below")
    print()
    inventory = [
        ("btn_play_again.png",      "end.png",         "右下 '再来一局' 按钮"),
        ("btn_close_x_top.png",     "end.png",         "弹窗右上角的白色 X"),
        ("popup_end_banner.png",    "end.png",         "顶部 '财源广进' 整条横幅 (检测用)"),
        ("btn_close_popup.png",     "after_close.png", "右上角 '关闭' 文字按钮"),
        ("btn_switch_opponent.png", "end.png",         "'切换对手' (可选)"),
        ("btn_esc_dialog.png",      "esc.png",         "可取消的对话框按钮 (可选)"),
        ("popup_ad_banner.png",     "ad.png",          "广告横幅 (可选, 检测用)"),
    ]
    print(f"{'Filename':<30s} {'Source':<18s} {'What to crop'}")
    print("-" * 80)
    for fn, src, desc in inventory:
        marker = "★" if "可选" not in desc else " "
        print(f"{marker} {fn:<28s} {src:<18s} {desc}")
    print()
    print("★ = required for best results; unmarked = optional")
    print()
    print("After cropping, verify with:")
    print("  python3 continuous_play.py test-templates")
    print()
    print("Note: the script runs fine without ANY templates — it just falls")
    print("back to blind coordinates + ESC. Templates improve reliability.")
    return 0


# ============================================================
# CLI entry
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Continuous multi-game supervisor for xiangqi-bot")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("play", help="main supervisor loop")
    pp.add_argument("--max-games", type=int, default=None,
                    help="stop after N games (default: unlimited)")
    pp.add_argument("--recovery-timeout", type=float, default=RECOVERY_TIMEOUT_S,
                    help=f"seconds allowed for recovery (default: {RECOVERY_TIMEOUT_S})")
    pp.set_defaults(func=cmd_play)

    pt = sub.add_parser("test-templates",
                        help="match all templates against current screen")
    pt.set_defaults(func=cmd_test_templates)

    pd = sub.add_parser("detect-end",
                        help="one-shot GameEndDetector report")
    pd.set_defaults(func=cmd_detect_end)

    pdiag = sub.add_parser("diag", help="dump screenshot + window info, no clicking")
    pdiag.set_defaults(func=cmd_diag)

    pc = sub.add_parser("crop-templates",
                        help="print the manual cropping workflow")
    pc.set_defaults(func=cmd_crop_templates)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
