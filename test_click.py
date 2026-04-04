#!/usr/bin/env python3
"""
Standalone click test for 天天象棋 bot.

Tests that:
1. The WeChat window can be activated and focused
2. Mouse moves to the correct position (visible to user)
3. Clicks register in the WeChat mini-program

Usage:
  python3 /tmp/test_click.py [strategy]

  strategy: pyautogui | cgevent (default) | applescript

The script will:
  - Activate the WeChat window
  - Move mouse to (652, 641) - red pawn at c3 (col 2, row 6)
  - Wait 2 seconds so you can visually verify position
  - Click to select the piece
  - Wait 1 second
  - Click at (652, 560) to move pawn to c4 (col 2, row 5)
  - Compare before/after screenshots to verify click registered
"""

import subprocess
import sys
import time


def activate_wechat_basic():
    """Strategy 1: Basic AppleScript activate."""
    print("  [activate] tell application WeChat to activate")
    subprocess.run(['osascript', '-e',
        'tell application "WeChat" to activate'],
        capture_output=True, timeout=3)


def activate_wechat_raise_window():
    """Strategy 2: Click the window title bar to focus this specific window."""
    import pyautogui
    # The window is at (0, 25), size 1628x960
    # Click near the middle of the title bar to ensure THIS window gets focus
    print("  [activate] Clicking window title bar area at (814, 35)")
    pyautogui.click(814, 35)
    time.sleep(0.1)


def click_pyautogui(x, y):
    """Strategy A: Standard pyautogui click."""
    import pyautogui
    print(f"  [pyautogui] moveTo({x}, {y})")
    pyautogui.moveTo(x, y, duration=0.15)
    time.sleep(0.1)
    print(f"  [pyautogui] click()")
    pyautogui.click()


def click_cgevent(x, y):
    """Strategy B: Direct CGEvent posting (lower-level, more reliable)."""
    import Quartz
    print(f"  [CGEvent] click at ({x}, {y})")
    point = Quartz.CGPointMake(x, y)

    # Move mouse first
    move = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, point, 0)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, move)
    time.sleep(0.05)

    # Mouse down
    down = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseDown, point, Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
    time.sleep(0.05)

    # Mouse up
    up = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseUp, point, Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)


def click_applescript(x, y):
    """Strategy C: Click via AppleScript + System Events (needs Accessibility)."""
    print(f"  [AppleScript] click at ({x}, {y})")
    result = subprocess.run(['osascript', '-e', f'''
        tell application "System Events"
            click at {{{x}, {y}}}
        end tell
    '''], capture_output=True, text=True, timeout=3)
    if result.stderr:
        print(f"    stderr: {result.stderr.strip()}")
    return result.returncode == 0


def take_screenshot(label=""):
    """Take a window screenshot."""
    path = f"/tmp/test_click_{label}.png" if label else "/tmp/test_click.png"
    subprocess.run(['screencapture', '-x', '-o', '-l', '18991', path],
                   capture_output=True, check=True)
    print(f"  Screenshot saved: {path}")
    return path


def main():
    import pyautogui
    pyautogui.FAILSAFE = True

    src_x, src_y = 652, 641   # Red pawn at c3 (col 2, row 6)
    dst_x, dst_y = 652, 560   # Destination c4 (col 2, row 5)

    print("=" * 60)
    print("Xiangqi Bot Click Test")
    print("=" * 60)
    print(f"Screen size: {pyautogui.size()}")
    print(f"Source: ({src_x}, {src_y}) - should be red pawn at c3")
    print(f"Destination: ({dst_x}, {dst_y}) - should be c4")
    print()

    print("Starting in 3 seconds... (Ctrl+C to abort)")
    print("(Move mouse to corner to trigger FAILSAFE)")
    time.sleep(3)

    # Screenshot BEFORE
    print("\n--- Before clicks ---")
    take_screenshot("before")

    # Strategy selection
    strategy = "cgevent"
    if len(sys.argv) > 1:
        strategy = sys.argv[1]
    print(f"\nUsing click strategy: {strategy}")

    # Activate window
    print(f"\n--- Activating window ---")
    activate_wechat_basic()
    time.sleep(0.3)
    # Also click title bar to ensure this specific window has focus
    activate_wechat_raise_window()
    time.sleep(0.3)

    # Move to source position
    print(f"\n--- Moving to source ({src_x}, {src_y}) ---")
    print("  (Watch the cursor - it should be on a red pawn)")
    pyautogui.moveTo(src_x, src_y, duration=0.3)
    print(f"  Mouse now at: {pyautogui.position()}")
    print("  Waiting 2 seconds for visual verification...")
    time.sleep(2)

    # Click source
    print(f"\n--- Clicking source ({src_x}, {src_y}) ---")
    if strategy == "cgevent":
        click_cgevent(src_x, src_y)
    elif strategy == "applescript":
        click_applescript(src_x, src_y)
    else:
        click_pyautogui(src_x, src_y)

    # Screenshot after first click
    time.sleep(0.5)
    take_screenshot("after_select")

    # Wait before destination click
    print(f"\n  Waiting 1 second before destination click...")
    time.sleep(1)

    # Click destination
    print(f"\n--- Clicking destination ({dst_x}, {dst_y}) ---")
    if strategy == "cgevent":
        click_cgevent(dst_x, dst_y)
    elif strategy == "applescript":
        click_applescript(dst_x, dst_y)
    else:
        click_pyautogui(dst_x, dst_y)

    # Screenshot AFTER
    time.sleep(1.5)
    print(f"\n--- After clicks ---")
    take_screenshot("after")

    # Compare before and after
    print(f"\n--- Comparing screenshots ---")
    import cv2
    import numpy as np
    before = cv2.imread("/tmp/test_click_before.png")
    after = cv2.imread("/tmp/test_click_after.png")
    if before is not None and after is not None and before.shape == after.shape:
        diff = cv2.absdiff(before, after)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed = np.count_nonzero(gray_diff > 30)
        total = gray_diff.size
        pct = changed / total * 100
        print(f"  Changed pixels (>30 intensity): {changed}/{total} = {pct:.3f}%")
        if pct > 0.5:
            print("  --> BOARD CHANGED! Click likely worked.")
        elif pct > 0.01:
            print("  --> Minor change detected (cursor/highlight?).")
        else:
            print("  --> NO CHANGE. Click probably did NOT register.")

        # Also compare just the after_select to check if piece was highlighted
        after_sel = cv2.imread("/tmp/test_click_after_select.png")
        if after_sel is not None and after_sel.shape == before.shape:
            diff2 = cv2.absdiff(before, after_sel)
            gray_diff2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)
            changed2 = np.count_nonzero(gray_diff2 > 30)
            pct2 = changed2 / total * 100
            print(f"\n  After first click vs before: {changed2}/{total} = {pct2:.3f}%")
            if pct2 > 0.01:
                print("  --> First click caused a change (piece selection highlight?)")
            else:
                print("  --> First click caused NO change (click not registering?)")
    else:
        print("  Could not compare (different sizes or missing files)")

    print(f"\n--- Summary ---")
    print(f"Strategy used: {strategy}")
    print("Try different strategies if this one didn't work:")
    print("  python3 /tmp/test_click.py pyautogui")
    print("  python3 /tmp/test_click.py cgevent")
    print("  python3 /tmp/test_click.py applescript")
    print("\nDone!")


if __name__ == '__main__':
    main()
