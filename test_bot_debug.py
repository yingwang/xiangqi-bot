#!/usr/bin/env python3
"""
Debug test for Xiangqi Bot - does ONE complete cycle:
  a. Screenshot + parse board
  b. Pikafish suggests a move
  c. Clicks the move
  d. Waits 3 seconds
  e. Takes another screenshot
  f. Compares before/after for change detection
  g. Parses new board and prints FEN

Usage:
  python3 /tmp/test_bot_debug.py [--dry-run]

  --dry-run: Skip actual clicks, only test parsing and change detection.
"""

import subprocess
import sys
import time
import os
import json
import numpy as np
import cv2
import pyautogui
import Quartz

pyautogui.FAILSAFE = True

DRY_RUN = '--dry-run' in sys.argv

PIKAFISH = "/tmp/pikafish-src/src/pikafish"
PIKAFISH_DIR = "/tmp/pikafish-src/src"
SCREENSHOT_PATH = "/tmp/xiangqi_debug_screen.png"
CALIB_PATH = "/tmp/xiangqi_calib.json"
WIN_ID = 18991

# ---------------------------------------------------------------
# Bot helpers (extracted from xiangqi_bot.py for standalone use)
# ---------------------------------------------------------------

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


def load_calibration():
    with open(CALIB_PATH) as f:
        d = json.load(f)
    x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
    cw = (x2 - x1) / 8.0
    ch = (y2 - y1) / 9.0
    cols = [x1 + i * cw for i in range(9)]
    rows = [y1 + j * ch for j in range(10)]
    win_x, win_y = d.get('win_x', 0), d.get('win_y', 25)
    return cols, rows, cw, ch, win_x, win_y


def take_screenshot(path=SCREENSHOT_PATH):
    subprocess.run(['screencapture', '-x', '-o', '-l', str(WIN_ID), path],
                   capture_output=True, check=True)
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read screenshot at {path}")
    return img


def logical_to_pixel(lx, ly, win_x, win_y, retina):
    px = (lx - win_x) * retina
    py = (ly - win_y) * retina
    return int(px), int(py)


def extract_patch(img, cx, cy, ps):
    h, w = img.shape[:2]
    x1, y1 = max(0, cx - ps), max(0, cy - ps)
    x2, y2 = min(w, cx + ps), min(h, cy + ps)
    p = img[y1:y2, x1:x2]
    return p if p.shape[0] >= ps and p.shape[1] >= ps else None


def capture_templates(img, cols, rows, cw, ch, win_x, win_y, retina, playing_red=True):
    init = INIT_RED
    ps = int(min(cw, ch) * retina * 0.7)
    templates = {}

    for r in range(10):
        for c in range(9):
            piece = init[r][c]
            if piece is None:
                continue
            px, py = logical_to_pixel(cols[c], rows[r], win_x, win_y, retina)
            patch = extract_patch(img, px, py, ps)
            if patch is not None:
                if piece not in templates:
                    templates[piece] = []
                templates[piece].append(patch)

    templates['_'] = []
    for er in [1, 4, 5, 8]:
        for ec in range(9):
            px, py = logical_to_pixel(cols[ec], rows[er], win_x, win_y, retina)
            ep = extract_patch(img, px, py, ps)
            if ep is not None:
                templates['_'].append(ep)

    total = sum(len(v) for k, v in templates.items() if k != '_')
    print(f"  Templates: {len(templates)} types, {total} piece variants, "
          f"{len(templates.get('_', []))} empty variants")
    return templates, ps


def identify(img, px, py, ps, templates):
    patch = extract_patch(img, px, py, ps)
    if patch is None:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if gray.std() < 5:
        return None

    best_piece, best_piece_sc = None, 0.0
    best_empty_sc = 0.0

    for pc, tmpls in templates.items():
        for tmpl in tmpls:
            t = cv2.resize(tmpl, (patch.shape[1], patch.shape[0]))
            sc = cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED).max()
            if pc == '_':
                best_empty_sc = max(best_empty_sc, sc)
            elif sc > best_piece_sc:
                best_piece_sc, best_piece = sc, pc

    if best_empty_sc > best_piece_sc:
        return None
    return best_piece if best_piece_sc > 0.4 else None


def parse_board(img, cols, rows, win_x, win_y, retina, ps, templates):
    board = []
    for r in range(10):
        row = []
        for c in range(9):
            px, py = logical_to_pixel(cols[c], rows[r], win_x, win_y, retina)
            p = identify(img, px, py, ps, templates)
            row.append(p)
        board.append(row)
    return board


def board_to_fen(board, playing_red=True):
    if playing_red:
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
                if e:
                    s += str(e)
                    e = 0
                s += p
            if e:
                pass
        if e:
            s += str(e)
        parts.append(s)
    return "/".join(parts)


def print_board(board):
    for r in range(10):
        line = f"  {r}: "
        for c in range(9):
            p = board[r][c]
            line += f" {p}" if p else " ."
        print(line)


def pikafish(fen, movetime_ms=1000):
    proc = subprocess.Popen(
        [PIKAFISH], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, cwd=PIKAFISH_DIR)
    try:
        proc.stdin.write(f"uci\nisready\nposition fen {fen}\ngo movetime {movetime_ms}\n")
        proc.stdin.flush()
    except BrokenPipeError:
        err = proc.stderr.read()
        print(f"  Pikafish crash! stderr: {err[:200]}")
        proc.kill()
        return None, ""
    best, info = None, ""
    t0 = time.time()
    while time.time() - t0 < movetime_ms / 1000 + 5:
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
    try:
        proc.stdin.write("quit\n")
        proc.stdin.flush()
    except:
        pass
    try:
        proc.wait(timeout=2)
    except:
        proc.kill()
    return best, info


def uci_to_logical(move, cols, rows, playing_red=True):
    fc, fr = ord(move[0]) - ord('a'), int(move[1])
    tc, tr = ord(move[2]) - ord('a'), int(move[3])
    if playing_red:
        s = [(fc, 9 - fr), (tc, 9 - tr)]
    else:
        s = [(8 - fc, fr), (8 - tc, tr)]
    return [(cols[c], rows[r]) for c, r in s]


def activate_and_click(lx, ly):
    """Activate WeChat and click at logical coordinates using CGEvent."""
    # Activate WeChat app
    subprocess.run(['osascript', '-e',
        'tell application "WeChat" to activate'],
        capture_output=True, timeout=3)
    time.sleep(0.3)

    # Click the title bar of the 天天象棋 window to ensure it has focus
    # Window is at (0, 25), size 1628x960
    # Title bar is roughly at y=25 (the top of the window)
    # Click center of title bar
    title_point = Quartz.CGPointMake(814, 30)
    move_evt = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, title_point, 0)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, move_evt)
    time.sleep(0.05)
    down_evt = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseDown, title_point, Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_evt)
    time.sleep(0.05)
    up_evt = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseUp, title_point, Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_evt)
    time.sleep(0.3)

    # Now click at the actual target
    do_click(lx, ly)


def do_click(lx, ly):
    """Click at logical coordinates using CGEvent (most reliable on macOS)."""
    point = Quartz.CGPointMake(lx, ly)

    # Move mouse
    move_evt = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, point, 0)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, move_evt)
    time.sleep(0.08)

    # Mouse down
    down_evt = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseDown, point, Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_evt)
    time.sleep(0.08)

    # Mouse up
    up_evt = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseUp, point, Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_evt)


def crop_board_region(img, cols, rows, win_x, win_y, retina):
    px0, py0 = logical_to_pixel(cols[0], rows[0], win_x, win_y, retina)
    px8, py9 = logical_to_pixel(cols[8], rows[9], win_x, win_y, retina)
    margin = 10
    return img[max(0, py0 - margin):py9 + margin,
               max(0, px0 - margin):px8 + margin].copy()


def images_changed(img1, img2):
    if img1 is None or img2 is None:
        return True
    if img1.shape != img2.shape:
        return True
    diff = cv2.absdiff(img1, img2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
    changed_pixels = np.count_nonzero(gray_diff > 30)
    total_pixels = gray_diff.size
    change_ratio = changed_pixels / total_pixels
    return change_ratio > 0.005


# ---------------------------------------------------------------
# Main test
# ---------------------------------------------------------------

def main():
    print("=" * 60)
    print("Xiangqi Bot Debug Test")
    print("=" * 60)
    if DRY_RUN:
        print("*** DRY RUN: will NOT click ***")
    print()

    # Step 1: Load calibration
    print("[1] Loading calibration...")
    cols, rows, cw, ch, win_x, win_y = load_calibration()
    print(f"  Cell: {cw:.1f}x{ch:.1f}, window: ({win_x},{win_y})")

    # Step 2: Take screenshot and compute retina scale
    print("[2] Taking screenshot (before)...")
    img1 = take_screenshot("/tmp/debug_before.png")
    retina = img1.shape[1] / 1628
    print(f"  Image: {img1.shape[1]}x{img1.shape[0]}, retina={retina:.2f}x")

    # Step 3: Capture templates from initial position
    print("[3] Capturing templates...")
    templates, ps = capture_templates(img1, cols, rows, cw, ch, win_x, win_y, retina)

    # Step 4: Parse board
    print("[4] Parsing board...")
    t0 = time.time()
    board1 = parse_board(img1, cols, rows, win_x, win_y, retina, ps, templates)
    dt = time.time() - t0
    fen1 = board_to_fen(board1)
    print(f"  FEN: {fen1}  (parsed in {dt:.2f}s)")
    print_board(board1)

    expected = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
    if fen1 == expected:
        print("  OK: Initial position detected!")
    else:
        print(f"  WARN: Not initial position (expected {expected})")

    # Step 5: Pikafish
    print("[5] Consulting Pikafish...")
    full_fen = f"{fen1} w - - 0 1"
    print(f"  FEN: {full_fen}")
    t0 = time.time()
    best, info = pikafish(full_fen)
    dt = time.time() - t0
    print(f"  Best move: {best} ({dt:.1f}s)")
    if info:
        print(f"  Info: {info[:120]}")

    if not best or best == '(none)':
        print("  ERROR: Pikafish returned no move!")
        sys.exit(1)

    # Step 6: Calculate click coordinates
    pts = uci_to_logical(best, cols, rows)
    src_x, src_y = pts[0]
    dst_x, dst_y = pts[1]
    print(f"[6] Move {best}: click ({src_x:.0f},{src_y:.0f}) -> ({dst_x:.0f},{dst_y:.0f})")

    # Verify click coords are within the window
    print(f"  Window bounds: x=[{win_x}, {win_x + 1628}] y=[{win_y}, {win_y + 960}]")
    for label, x, y in [("source", src_x, src_y), ("dest", dst_x, dst_y)]:
        in_x = win_x <= x <= win_x + 1628
        in_y = win_y <= y <= win_y + 960
        status = "OK" if (in_x and in_y) else "OUT OF BOUNDS!"
        print(f"  {label} ({x:.0f},{y:.0f}): {status}")

    # Save board region before click
    snap_before = crop_board_region(img1, cols, rows, win_x, win_y, retina)
    print(f"  Board crop size: {snap_before.shape[1]}x{snap_before.shape[0]}")

    if DRY_RUN:
        print("\n[DRY RUN] Skipping clicks. Taking second screenshot to test change detection...")
        time.sleep(1)
        img2 = take_screenshot("/tmp/debug_after.png")
    else:
        # Step 7: Execute clicks
        print("[7] Executing clicks...")
        print(f"  Activating window and clicking source piece...")
        activate_and_click(src_x, src_y)
        time.sleep(1.0)  # Wait for selection highlight

        # Take intermediate screenshot to verify selection
        img_sel = take_screenshot("/tmp/debug_selected.png")
        snap_selected = crop_board_region(img_sel, cols, rows, win_x, win_y, retina)
        sel_changed = images_changed(snap_before, snap_selected)
        print(f"  After source click: changed={sel_changed}")
        if not sel_changed:
            print("  WARNING: No visual change after clicking source piece!")
            print("  The click may not have registered.")

        print(f"  Clicking destination...")
        do_click(dst_x, dst_y)

        # Step 8: Wait and take screenshot
        print("[8] Waiting 3 seconds for animation to settle...")
        time.sleep(3)
        img2 = take_screenshot("/tmp/debug_after.png")

    # Step 9: Compare before/after
    print("[9] Comparing screenshots...")
    snap_after = crop_board_region(img2, cols, rows, win_x, win_y, retina)

    diff = cv2.absdiff(snap_before, snap_after)
    if snap_before.shape == snap_after.shape:
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
        changed_pixels = np.count_nonzero(gray_diff > 30)
        total_pixels = gray_diff.size
        change_ratio = changed_pixels / total_pixels
        print(f"  Changed pixels (>30): {changed_pixels}/{total_pixels} = {change_ratio*100:.3f}%")
        print(f"  Threshold: 0.5%")
        print(f"  images_changed() would return: {change_ratio > 0.005}")

        if change_ratio > 0.005:
            print("  --> CHANGE DETECTED: Board state changed!")
        else:
            print("  --> NO CHANGE: The click did not affect the board.")

        # Save diff image for visual inspection
        diff_vis = cv2.normalize(gray_diff, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite("/tmp/debug_diff.png", diff_vis)
        print(f"  Diff image saved: /tmp/debug_diff.png")
    else:
        print(f"  Shape mismatch: {snap_before.shape} vs {snap_after.shape}")

    # Step 10: Parse new board
    print("[10] Parsing new board...")
    board2 = parse_board(img2, cols, rows, win_x, win_y, retina, ps, templates)
    fen2 = board_to_fen(board2)
    print(f"  FEN after: {fen2}")
    print_board(board2)

    if fen1 == fen2:
        print("\n  RESULT: Board did NOT change. Click failed to register.")
    else:
        print(f"\n  RESULT: Board changed!")
        print(f"    Before: {fen1}")
        print(f"    After:  {fen2}")

    # Summary
    print("\n" + "=" * 60)
    print("Files created:")
    print("  /tmp/debug_before.png  - Screenshot before click")
    if not DRY_RUN:
        print("  /tmp/debug_selected.png - Screenshot after selecting piece")
    print("  /tmp/debug_after.png   - Screenshot after move")
    print("  /tmp/debug_diff.png    - Visual diff of board region")
    print("=" * 60)


if __name__ == '__main__':
    main()
