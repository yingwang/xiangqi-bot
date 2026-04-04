#!/usr/bin/env python3
"""
Test the identify_v2 color-based piece classifier.

This script:
1. Takes a screenshot of the current board (mid-game after 18 moves)
2. Loads calibration from /tmp/xiangqi_calib.json
3. Captures feature vectors from known piece positions
4. Tests identification on the current board state against ground truth
5. Reports accuracy
"""

import subprocess
import sys
import os
import json
import time
import numpy as np
import cv2

# Import the bot
sys.path.insert(0, '/tmp')
from xiangqi_bot import Bot, INIT_RED, INIT_BLACK, SCREENSHOT_PATH, CALIB_PATH

INITIAL_SCREENSHOT = "/tmp/xiangqi_initial.png"
TEST_OUTPUT_DIR = "/tmp/xiangqi_test_output"

# Ground truth for the current mid-game position (verified by visual inspection)
GROUND_TRUTH = [
    [None, 'r', None, None, 'k', 'a', 'b', None, 'r'],
    [None, None, None, None, 'a', None, None, 'R', None],
    [None, 'c', 'n', None, None, None, None, None, 'c'],
    ['p', None, 'p', None, 'p', None, None, None, 'p'],
    [None]*9,
    [None, None, 'P', None, None, None, None, None, None],
    ['P', None, None, None, 'P', None, 'n', None, 'P'],
    [None, None, None, None, 'C', None, 'N', None, None],
    [None]*9,
    [None, 'R', 'B', 'A', 'K', 'A', 'B', None, None],
]


def find_wechat_window():
    """Find the WeChat window dynamically."""
    import Quartz
    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
    for w in windows:
        owner = w.get('kCGWindowOwnerName', '')
        name = w.get('kCGWindowName', '')
        if 'WeChat' in owner and '天天象棋' in name:
            return w['kCGWindowNumber'], w['kCGWindowBounds']
    for w in windows:
        owner = w.get('kCGWindowOwnerName', '')
        if 'WeChat' in owner:
            b = w['kCGWindowBounds']
            if int(b['Width']) > 800:
                return w['kCGWindowNumber'], b
    return None, None


def take_screenshot(win_id, path):
    """Capture window screenshot."""
    subprocess.run(['screencapture', '-x', '-o', '-l', str(win_id), path],
                   capture_output=True, check=True)
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to capture screenshot to {path}")
    return img


def setup_bot():
    """Create and configure a Bot instance from calibration."""
    bot = Bot()

    win_id, bounds = find_wechat_window()
    if win_id is None:
        print("ERROR: Could not find WeChat window")
        sys.exit(1)

    bot.win_id = win_id
    bot.win_x = int(bounds['X'])
    bot.win_y = int(bounds['Y'])
    win_w = int(bounds['Width'])
    print(f"Window: id={win_id} pos=({bot.win_x},{bot.win_y}) size={win_w}x{int(bounds['Height'])}")

    if not os.path.exists(CALIB_PATH):
        print(f"ERROR: Calibration file {CALIB_PATH} not found")
        sys.exit(1)

    with open(CALIB_PATH) as f:
        calib = json.load(f)

    x1, y1, x2, y2 = calib['x1'], calib['y1'], calib['x2'], calib['y2']
    bot.cell_w = (x2 - x1) / 8.0
    bot.cell_h = (y2 - y1) / 9.0
    bot.cols_logical = [x1 + i * bot.cell_w for i in range(9)]
    bot.rows_logical = [y1 + j * bot.cell_h for j in range(10)]

    if calib.get('win_x') != bot.win_x or calib.get('win_y') != bot.win_y:
        dx = bot.win_x - calib.get('win_x', 0)
        dy = bot.win_y - calib.get('win_y', 0)
        bot.cols_logical = [c + dx for c in bot.cols_logical]
        bot.rows_logical = [r + dy for r in bot.rows_logical]
        print(f"Window moved by ({dx},{dy}), adjusted grid")

    print(f"Cell size: {bot.cell_w:.1f}x{bot.cell_h:.1f}")

    test_path = "/tmp/xiangqi_test_screen.png"
    img = take_screenshot(win_id, test_path)
    bot.retina_scale = img.shape[1] / win_w
    print(f"Image: {img.shape[1]}x{img.shape[0]}, retina={bot.retina_scale:.2f}x")

    bot.patch_size = int(min(bot.cell_w, bot.cell_h) * bot.retina_scale * 0.7)
    print(f"Patch size: {bot.patch_size}")

    return bot, img


def print_board(board, label=""):
    """Pretty-print a board."""
    if label:
        print(f"\n{label}:")
    print("     a   b   c   d   e   f   g   h   i")
    for r in range(10):
        line = f"  {r}"
        for c in range(9):
            p = board[r][c]
            line += f"  {p:>2}" if p else "   ."
        print(line)


def test_piece_detection(bot, img):
    """Test piece detection against ground truth."""
    print("\n" + "=" * 60)
    print("TEST 1: Piece Detection (_has_piece_v2)")
    print("=" * 60)

    correct = 0
    false_pos = 0
    false_neg = 0

    print("\n  Legend: O=correct piece, .=correct empty, X=false positive, !=false negative")
    print("     a b c d e f g h i")

    for r in range(10):
        row_str = f"  {r}"
        for c in range(9):
            px, py = bot.logical_to_pixel(bot.cols_logical[c], bot.rows_logical[r])
            detected = bot._has_piece_v2(img, px, py)
            gt_has = GROUND_TRUTH[r][c] is not None

            if detected and gt_has:
                correct += 1
                row_str += " O"
            elif not detected and not gt_has:
                correct += 1
                row_str += " ."
            elif detected and not gt_has:
                false_pos += 1
                row_str += " X"
            else:
                false_neg += 1
                row_str += " !"
        print(row_str)

    total = correct + false_pos + false_neg
    gt_pieces = sum(1 for r in GROUND_TRUTH for p in r if p is not None)
    gt_empty = 90 - gt_pieces
    print(f"\n  Piece detection: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print(f"  False positives: {false_pos} (empty cells detected as piece)")
    print(f"  False negatives: {false_neg} (piece cells missed)")
    print(f"  Ground truth: {gt_pieces} pieces, {gt_empty} empty")

    return correct, false_pos, false_neg


def test_color_classification(bot, img):
    """Test red vs black classification on cells with pieces."""
    print("\n" + "=" * 60)
    print("TEST 2: Color Classification (_classify_color_v2)")
    print("=" * 60)

    correct = 0
    wrong = 0
    total = 0

    print("\n  Legend: R/B=correct, r/b=wrong classification")
    print("     a b c d e f g h i")

    for r in range(10):
        row_str = f"  {r}"
        for c in range(9):
            gt = GROUND_TRUTH[r][c]
            if gt is None:
                row_str += " ."
                continue

            px, py = bot.logical_to_pixel(bot.cols_logical[c], bot.rows_logical[r])
            color = bot._classify_color_v2(img, px, py)

            # Determine expected color
            if gt.isupper():
                expected = 'red'  # playing red, uppercase = red
            else:
                expected = 'black'

            total += 1
            if color == expected:
                correct += 1
                row_str += " R" if expected == 'red' else " B"
            else:
                wrong += 1
                row_str += " r" if expected == 'red' else " b"
        print(row_str)

    print(f"\n  Color classification: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print(f"  Wrong: {wrong}")

    return correct, wrong


def test_feature_vectors(bot, img):
    """Test feature vector self-similarity."""
    print("\n" + "=" * 60)
    print("TEST 3: Feature Vector Self-Similarity")
    print("=" * 60)

    if not hasattr(bot, 'piece_features') or not bot.piece_features:
        print("  No feature vectors captured!")
        return

    print(f"\n  Reference pieces: {list(bot.piece_features.keys())}")
    for piece, vecs in sorted(bot.piece_features.items()):
        print(f"    {piece}: {len(vecs)} vectors")

    # Check if same-type pieces are more similar to each other than to different types
    pieces = sorted(bot.piece_features.keys())
    print("\n  Self-similarity (diagonal should be highest in each row):")
    header = "      " + "  ".join(f"{p:>5}" for p in pieces)
    print(header)
    for p1 in pieces:
        row = f"  {p1:>3} "
        for p2 in pieces:
            sims = []
            for v1 in bot.piece_features[p1]:
                for v2 in bot.piece_features[p2]:
                    sims.append(bot._cosine_similarity(v1, v2))
            avg = np.mean(sims) if sims else 0
            marker = "*" if p1 == p2 else " "
            row += f"{avg:5.2f}{marker}"
        print(row)


def test_full_identification(bot, img):
    """Test full identify_v2 against ground truth."""
    print("\n" + "=" * 60)
    print("TEST 4: Full Identification (identify_v2) vs Ground Truth")
    print("=" * 60)

    board_v2 = []
    for r in range(10):
        row = []
        for c in range(9):
            px, py = bot.logical_to_pixel(bot.cols_logical[c], bot.rows_logical[r])
            piece = bot.identify_v2(img, px, py)
            row.append(piece)
        board_v2.append(row)

    print_board(board_v2, "  V2 identified board")
    print_board(GROUND_TRUTH, "  Ground truth")

    # Compare
    correct_pieces = 0
    correct_empty = 0
    wrong_type = 0
    false_pos = 0
    false_neg = 0

    print("\n  Detailed comparison:")
    for r in range(10):
        for c in range(9):
            v2 = board_v2[r][c]
            gt = GROUND_TRUTH[r][c]
            if v2 == gt:
                if gt is not None:
                    correct_pieces += 1
                else:
                    correct_empty += 1
            elif gt is None and v2 is not None:
                false_pos += 1
                print(f"    ({r},{c}): FALSE POS - v2={v2}, gt=empty")
            elif gt is not None and v2 is None:
                false_neg += 1
                print(f"    ({r},{c}): FALSE NEG - v2=empty, gt={gt}")
            else:
                wrong_type += 1
                print(f"    ({r},{c}): WRONG TYPE - v2={v2}, gt={gt}")

    total_cells = 90
    total_correct = correct_pieces + correct_empty
    gt_pieces = sum(1 for r in GROUND_TRUTH for p in r if p is not None)

    print(f"\n  Overall accuracy: {total_correct}/{total_cells} ({100*total_correct/total_cells:.1f}%)")
    print(f"  Correct pieces: {correct_pieces}/{gt_pieces}")
    print(f"  Correct empties: {correct_empty}/{90 - gt_pieces}")
    print(f"  Wrong piece type: {wrong_type}")
    print(f"  False positives: {false_pos}")
    print(f"  False negatives: {false_neg}")

    return board_v2


def test_cross_position(bot, img):
    """Test if pieces at non-initial positions can be identified.

    This is the KEY test for the v2 approach.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Cross-Position Robustness")
    print("=" * 60)

    if not hasattr(bot, 'piece_features') or not bot.piece_features:
        print("  No feature vectors available!")
        return

    # For each ground-truth piece position, try to identify it
    results = []
    for r in range(10):
        for c in range(9):
            gt = GROUND_TRUTH[r][c]
            if gt is None:
                continue

            px, py = bot.logical_to_pixel(bot.cols_logical[c], bot.rows_logical[r])
            vec = bot._compute_feature_vector(img, px, py)
            if vec is None:
                results.append({
                    'row': r, 'col': c, 'gt': gt,
                    'matched': None, 'sim': 0, 'correct': False,
                    'note': 'no feature vector'
                })
                continue

            color = bot._classify_color_v2(img, px, py)

            # Find best match with color filter
            best_piece = None
            best_sim = -1.0
            for piece, vecs in bot.piece_features.items():
                piece_color = bot.piece_color_map.get(piece, 'unknown')
                if piece_color != color and color != 'unknown':
                    continue
                for ref_vec in vecs:
                    sim = bot._cosine_similarity(vec, ref_vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_piece = piece

            # Also find best without color filter
            best_any = None
            best_any_sim = -1.0
            for piece, vecs in bot.piece_features.items():
                for ref_vec in vecs:
                    sim = bot._cosine_similarity(vec, ref_vec)
                    if sim > best_any_sim:
                        best_any_sim = sim
                        best_any = piece

            correct = (best_piece == gt)
            results.append({
                'row': r, 'col': c, 'gt': gt,
                'matched': best_piece, 'sim': best_sim,
                'matched_any': best_any, 'sim_any': best_any_sim,
                'color': color, 'correct': correct,
                'note': ''
            })

    print(f"\n  {'Pos':>6} {'GT':>3} {'Color':>6} {'Match':>6} {'Sim':>5} {'Any':>5} {'SimA':>5} {'OK':>3}")
    for r in results:
        ok = "YES" if r['correct'] else "NO"
        print(f"  ({r['row']},{r['col']})  {r['gt']:>2}  {r.get('color','?'):>5}"
              f"  {str(r['matched']):>5} {r['sim']:5.3f}"
              f"  {str(r.get('matched_any','')):>4} {r.get('sim_any',0):5.3f}  {ok}")

    correct_count = sum(1 for r in results if r['correct'])
    total = len(results)
    print(f"\n  Piece type accuracy: {correct_count}/{total} ({100*correct_count/total:.1f}%)")

    # Similarity stats
    sims = [r['sim'] for r in results if r['matched'] is not None]
    if sims:
        print(f"  Similarity stats (color-filtered):")
        print(f"    Min: {min(sims):.3f}  Max: {max(sims):.3f}")
        print(f"    Mean: {np.mean(sims):.3f}  Median: {np.median(sims):.3f}")


def save_debug_patches(bot, img):
    """Save individual cell patches for visual inspection."""
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    for r in range(10):
        for c in range(9):
            px, py = bot.logical_to_pixel(bot.cols_logical[c], bot.rows_logical[r])
            ps = bot.patch_size
            h, w = img.shape[:2]
            x1, y1 = max(0, px - ps), max(0, py - ps)
            x2, y2 = min(w, px + ps), min(h, py + ps)
            patch = img[y1:y2, x1:x2]
            if patch.size > 0:
                cv2.imwrite(f"{TEST_OUTPUT_DIR}/cell_{r}_{c}.png", patch)

                radius = int(min(bot.cell_w, bot.cell_h) * bot.retina_scale * 0.20)
                cx1 = max(0, px - radius)
                cy1 = max(0, py - radius)
                cx2 = min(w, px + radius)
                cy2 = min(h, py + radius)
                center = img[cy1:cy2, cx1:cx2]
                if center.size > 0:
                    cv2.imwrite(f"{TEST_OUTPUT_DIR}/center_{r}_{c}.png", center)


def main():
    print("=" * 60)
    print("Xiangqi identify_v2 Test Suite")
    print("=" * 60)

    bot, img = setup_bot()
    bot.detect_orientation(img)

    # Capture v1 templates for comparison
    bot.capture_templates(img)

    # Capture v2 feature vectors
    # Use the ground truth board state to build feature vectors
    # from the CURRENT screenshot (not initial position)
    print("\nCapturing feature vectors from current board (ground truth positions)...")
    bot.piece_color_map = {}
    for piece in set(p for row in GROUND_TRUTH for p in row if p):
        if piece.isupper():
            bot.piece_color_map[piece] = 'red'
        else:
            bot.piece_color_map[piece] = 'black'

    bot.piece_features = {}
    for r in range(10):
        for c in range(9):
            piece = GROUND_TRUTH[r][c]
            if piece is None:
                continue
            px, py = bot.logical_to_pixel(bot.cols_logical[c], bot.rows_logical[r])
            vec = bot._compute_feature_vector(img, px, py)
            if vec is not None:
                if piece not in bot.piece_features:
                    bot.piece_features[piece] = []
                bot.piece_features[piece].append(vec)

    total = sum(len(v) for v in bot.piece_features.values())
    print(f"  Feature vectors: {len(bot.piece_features)} types, {total} vectors")

    # Run tests
    test_piece_detection(bot, img)
    test_color_classification(bot, img)
    test_feature_vectors(bot, img)
    board_v2 = test_full_identification(bot, img)
    test_cross_position(bot, img)

    save_debug_patches(bot, img)

    # Also test: can feature vectors from THIS position match pieces at
    # OTHER positions? This simulates what happens after a move.
    print("\n" + "=" * 60)
    print("TEST 6: Leave-One-Out Cross-Validation")
    print("=" * 60)
    print("  (For each piece, remove its vector and try to match using others)")

    loo_correct = 0
    loo_total = 0

    for r in range(10):
        for c in range(9):
            piece = GROUND_TRUTH[r][c]
            if piece is None:
                continue

            px, py = bot.logical_to_pixel(bot.cols_logical[c], bot.rows_logical[r])
            vec = bot._compute_feature_vector(img, px, py)
            if vec is None:
                continue

            color = bot._classify_color_v2(img, px, py)

            # Remove this vector from the reference set
            orig_vecs = bot.piece_features.get(piece, [])
            remaining = [v for v in orig_vecs if not np.array_equal(v, vec)]

            if len(remaining) == len(orig_vecs):
                # Vector wasn't in the set (shouldn't happen), use all
                remaining = orig_vecs

            # Temporarily replace
            bot.piece_features[piece] = remaining

            # Try to match
            best_piece = None
            best_sim = -1.0
            for p, vecs in bot.piece_features.items():
                if not vecs:
                    continue
                pc = bot.piece_color_map.get(p, 'unknown')
                if pc != color and color != 'unknown':
                    continue
                for rv in vecs:
                    sim = bot._cosine_similarity(vec, rv)
                    if sim > best_sim:
                        best_sim = sim
                        best_piece = p

            # Restore
            bot.piece_features[piece] = orig_vecs

            loo_total += 1
            if best_piece == piece:
                loo_correct += 1
            else:
                print(f"    ({r},{c}) {piece}: matched {best_piece} (sim={best_sim:.3f})")

    if loo_total > 0:
        print(f"\n  LOO accuracy: {loo_correct}/{loo_total} ({100*loo_correct/loo_total:.1f}%)")

    # Final FEN
    fen_v2 = bot.board_to_fen(board_v2) if board_v2 else "N/A"
    print(f"\n{'=' * 60}")
    print(f"V2 FEN: {fen_v2}")
    print(f"{'=' * 60}")

    print(f"\nDebug patches saved to {TEST_OUTPUT_DIR}/")
    print("Done!")


if __name__ == '__main__':
    main()
