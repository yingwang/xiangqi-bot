#!/usr/bin/env python3
"""
Xiangqi Bot — macOS App
Native Cocoa GUI with graphical board, Chinese notation, and game history.
"""

import os
import sys
import re
import math

# Fix OpenCV recursion in PyInstaller
sys.path[:] = [p for p in sys.path if 'cv2' not in p]
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import threading
import io

import objc
from Foundation import (
    NSObject, NSTimer, NSRunLoop, NSDefaultRunLoopMode,
    NSMutableAttributedString, NSRange, NSMakePoint,
)
from AppKit import (
    NSApplication, NSWindow, NSButton, NSTextField, NSScrollView, NSTextView,
    NSFont, NSColor, NSMakeRect, NSApp, NSBox, NSView, NSBezierPath,
    NSWindowStyleMaskTitled, NSWindowStyleMaskClosable, NSWindowStyleMaskMiniaturizable,
    NSBackingStoreBuffered, NSBezelStyleRounded,
    NSTextAlignmentCenter, NSTextAlignmentLeft, NSTextAlignmentRight,
    NSApplicationActivationPolicyRegular,
    NSForegroundColorAttributeName, NSFontAttributeName,
    NSAttributedString,
)

if getattr(sys, '_MEIPASS', None):
    os.chdir(sys._MEIPASS)
    BUNDLE_DIR = sys._MEIPASS
else:
    BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ['XIANGQI_BOT_DIR'] = BUNDLE_DIR

# Piece display and notation
# Characters matching 天天象棋 app
# Piece names by case (used for board display)
PIECE_NAMES = {
    'R': '车', 'N': '马', 'B': '相', 'A': '仕', 'K': '帅', 'C': '炮', 'P': '兵',
    'r': '車', 'n': '馬', 'b': '象', 'a': '士', 'k': '将', 'c': '炮', 'p': '卒',
}
RED_PIECES = set('RNBAKCP')

# Notation names by color (keyed by uppercase piece type, independent of CNN case)
RED_NAMES = {'R': '车', 'N': '马', 'B': '相', 'A': '仕', 'K': '帅', 'C': '炮', 'P': '兵'}
BLACK_NAMES = {'R': '車', 'N': '馬', 'B': '象', 'A': '士', 'K': '将', 'C': '炮', 'P': '卒'}

# Column names: red uses Chinese numerals, black uses full-width digits
COL_NAMES_RED = ['九', '八', '七', '六', '五', '四', '三', '二', '一']
COL_NAMES_BLACK = ['１', '２', '３', '４', '５', '６', '７', '８', '９']
DIST_RED = ['', '一', '二', '三', '四', '五', '六', '七', '八', '九']
DIST_BLACK = ['', '１', '２', '３', '４', '５', '６', '７', '８', '９']


def uci_to_chinese(move, board, playing_red):
    """Convert UCI move to Chinese notation (四字记录法).
    This is only called for the BOT's own moves, so is_red=playing_red.
    """
    if not move or len(move) < 4:
        return move
    fc, fr = ord(move[0]) - ord('a'), int(move[1])
    tc, tr = ord(move[2]) - ord('a'), int(move[3])

    # Convert to board coords (row 0=top, col 0=left)
    if playing_red:
        from_r, from_c = 9 - fr, fc
        to_r, to_c = 9 - tr, tc
    else:
        from_r, from_c = fr, 8 - fc
        to_r, to_c = tr, 8 - tc

    piece = board[from_r][from_c] if 0 <= from_r < 10 and 0 <= from_c < 9 else None
    if not piece:
        return move

    # Bot's own move: is_red = playing_red
    return _make_notation(piece, from_r, from_c, to_r, to_c, board, playing_red, is_red=playing_red)


def _make_notation(piece, from_r, from_c, to_r, to_c, board, playing_red, is_red):
    """Build standard 四字记录法 notation for a move.

    is_red: whether the moving piece is red (determines piece name & numeral format).
            Passed explicitly because CNN may misclassify piece color (case).
    playing_red: board orientation (True = red at bottom, False = red at top).
    """
    piece_type = piece.upper()
    name = RED_NAMES[piece_type] if is_red else BLACK_NAMES[piece_type]

    # Column names: orientation-dependent
    if playing_red:
        col_red = COL_NAMES_RED
        col_black = COL_NAMES_BLACK
    else:
        col_red = COL_NAMES_RED[::-1]
        col_black = COL_NAMES_BLACK[::-1]

    if is_red:
        col_from = col_red[from_c]
        col_to = col_red[to_c]
    else:
        col_from = col_black[from_c]
        col_to = col_black[to_c]

    # 进 = toward opponent. Direction depends on board orientation + piece color.
    if is_red:
        forward_is_decreasing = playing_red   # red at bottom → advance up (decreasing row)
    else:
        forward_is_decreasing = not playing_red  # black at bottom → advance up (decreasing row)

    # 前/后 disambiguation: same-type pieces (same case) on same column
    same_col = [r for r in range(10) if board[r][from_c] is not None
                and board[r][from_c].upper() == piece_type
                and (board[r][from_c].isupper() == piece.isupper())]
    if len(same_col) >= 2:
        same_col.sort()
        if forward_is_decreasing:
            is_front = (from_r == same_col[0])
        else:
            is_front = (from_r == same_col[-1])
        char1 = ('前' if is_front else '后') + name
    else:
        char1 = name + col_from

    # Direction and target
    if from_c == to_c:
        dist = abs(to_r - from_r)
        if forward_is_decreasing:
            direction = '进' if to_r < from_r else '退'
        else:
            direction = '进' if to_r > from_r else '退'
        if is_red:
            dist_str = DIST_RED[dist] if dist <= 9 else str(dist)
        else:
            dist_str = DIST_BLACK[dist] if dist <= 9 else str(dist)
        return f"{char1}{direction}{dist_str}"
    elif from_r == to_r:
        return f"{char1}平{col_to}"
    else:
        if forward_is_decreasing:
            direction = '进' if to_r < from_r else '退'
        else:
            direction = '进' if to_r > from_r else '退'
        return f"{char1}{direction}{col_to}"


class BoardView(NSView):
    """Custom view that draws a graphical xiangqi board."""

    board_data = objc.ivar('board_data')
    last_move = objc.ivar('last_move')

    def initWithFrame_(self, frame):
        self = objc.super(BoardView, self).initWithFrame_(frame)
        if self is not None:
            self.board_data = None
            self.last_move = None
        return self

    def isFlipped(self):
        return True

    @objc.python_method
    def set_board(self, board, last_move=None):
        self.board_data = board
        self.last_move = last_move
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        bounds = self.bounds()
        w = bounds.size.width
        h = bounds.size.height

        # Background
        NSColor.colorWithRed_green_blue_alpha_(0.96, 0.92, 0.82, 1).set()
        NSBezierPath.fillRect_(bounds)

        if not self.board_data:
            # Draw placeholder
            NSColor.grayColor().set()
            return

        # Grid dimensions
        margin_x = 25
        margin_y = 20
        grid_w = w - 2 * margin_x
        grid_h = h - 2 * margin_y
        cell_w = grid_w / 8.0
        cell_h = grid_h / 9.0
        radius = min(cell_w, cell_h) * 0.42

        # Draw grid lines
        NSColor.colorWithRed_green_blue_alpha_(0.3, 0.2, 0.1, 1).set()

        # Horizontal lines
        for r in range(10):
            y = margin_y + r * cell_h
            path = NSBezierPath.bezierPath()
            path.moveToPoint_(NSMakePoint(margin_x, y))
            path.lineToPoint_(NSMakePoint(margin_x + 8 * cell_w, y))
            path.setLineWidth_(1.0)
            path.stroke()

        # Vertical lines (with river gap)
        for c in range(9):
            x = margin_x + c * cell_w
            if c == 0 or c == 8:
                # Edge lines go full height
                path = NSBezierPath.bezierPath()
                path.moveToPoint_(NSMakePoint(x, margin_y))
                path.lineToPoint_(NSMakePoint(x, margin_y + 9 * cell_h))
                path.setLineWidth_(1.0)
                path.stroke()
            else:
                # Top half
                path = NSBezierPath.bezierPath()
                path.moveToPoint_(NSMakePoint(x, margin_y))
                path.lineToPoint_(NSMakePoint(x, margin_y + 4 * cell_h))
                path.setLineWidth_(1.0)
                path.stroke()
                # Bottom half
                path = NSBezierPath.bezierPath()
                path.moveToPoint_(NSMakePoint(x, margin_y + 5 * cell_h))
                path.lineToPoint_(NSMakePoint(x, margin_y + 9 * cell_h))
                path.setLineWidth_(1.0)
                path.stroke()

        # Palace diagonals
        for (r1, c1, r2, c2) in [(0, 3, 2, 5), (0, 5, 2, 3), (7, 3, 9, 5), (7, 5, 9, 3)]:
            path = NSBezierPath.bezierPath()
            path.moveToPoint_(NSMakePoint(margin_x + c1 * cell_w, margin_y + r1 * cell_h))
            path.lineToPoint_(NSMakePoint(margin_x + c2 * cell_w, margin_y + r2 * cell_h))
            path.setLineWidth_(1.0)
            path.stroke()

        # River text
        river_y = margin_y + 4.5 * cell_h
        font = NSFont.fontWithName_size_("STKaiti", 14) or NSFont.systemFontOfSize_(14)
        attrs = {
            NSForegroundColorAttributeName: NSColor.colorWithRed_green_blue_alpha_(0.4, 0.3, 0.2, 1),
            NSFontAttributeName: font,
        }
        river_left = NSAttributedString.alloc().initWithString_attributes_("楚  河", attrs)
        river_right = NSAttributedString.alloc().initWithString_attributes_("汉  界", attrs)
        river_left.drawAtPoint_(NSMakePoint(margin_x + 0.8 * cell_w, river_y - 10))
        river_right.drawAtPoint_(NSMakePoint(margin_x + 5.2 * cell_w, river_y - 10))

        # Highlight last move
        if self.last_move:
            fr, fc, tr, tc = self.last_move
            NSColor.colorWithRed_green_blue_alpha_(1.0, 0.9, 0.3, 0.5).set()
            for (hr, hc) in [(fr, fc), (tr, tc)]:
                cx = margin_x + hc * cell_w
                cy = margin_y + hr * cell_h
                highlight = NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(cx - radius - 2, cy - radius - 2,
                               (radius + 2) * 2, (radius + 2) * 2))
                highlight.fill()

        # Draw pieces
        piece_font = NSFont.fontWithName_size_("STHeiti", radius * 1.1) or NSFont.boldSystemFontOfSize_(radius * 1.1)

        for r in range(10):
            for c in range(9):
                piece = self.board_data[r][c]
                if not piece:
                    continue

                cx = margin_x + c * cell_w
                cy = margin_y + r * cell_h
                is_red = piece in RED_PIECES

                # Piece circle background
                NSColor.colorWithRed_green_blue_alpha_(0.98, 0.96, 0.88, 1).set()
                circle = NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(cx - radius, cy - radius, radius * 2, radius * 2))
                circle.fill()

                # Circle border
                if is_red:
                    NSColor.colorWithRed_green_blue_alpha_(0.8, 0.1, 0.1, 1).set()
                else:
                    NSColor.colorWithRed_green_blue_alpha_(0.1, 0.1, 0.1, 1).set()
                circle.setLineWidth_(2.0)
                circle.stroke()

                # Piece character
                char = PIECE_NAMES.get(piece, piece)
                color = NSColor.colorWithRed_green_blue_alpha_(0.8, 0.1, 0.1, 1) if is_red \
                    else NSColor.colorWithRed_green_blue_alpha_(0.1, 0.1, 0.1, 1)
                attrs = {
                    NSForegroundColorAttributeName: color,
                    NSFontAttributeName: piece_font,
                }
                text = NSAttributedString.alloc().initWithString_attributes_(char, attrs)
                tw = text.size().width
                th = text.size().height
                text.drawAtPoint_(NSMakePoint(cx - tw / 2, cy - th / 2))


class AppDelegate(NSObject):
    def init(self):
        self = objc.super(AppDelegate, self).init()
        self.running = False
        self.bot = None
        self.bot_thread = None
        self.log_buffer = []
        self.move_num = 0
        self.board_lines = []
        self.current_board = None
        self.rounds = []  # [{'num': int, 'red': str, 'black': str, 'eval': str}]
        self.playing_red = True
        self.round_counter = 0
        return self

    def applicationDidFinishLaunching_(self, notification):
        self._build_window()
        self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.1, self, "flushLog:", None, True)
        NSRunLoop.currentRunLoop().addTimer_forMode_(self.timer, NSDefaultRunLoopMode)

    def _build_window(self):
        ww = 850
        wh = 650
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(100, 50, ww, wh),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable,
            NSBackingStoreBuffered, False)
        self.window.setTitle_("天天象棋 Bot")
        self.window.setBackgroundColor_(NSColor.colorWithRed_green_blue_alpha_(0.95, 0.95, 0.95, 1))

        content = self.window.contentView()

        # --- Title ---
        content.addSubview_(self._label(NSMakeRect(0, wh - 40, ww, 30), "天天象棋 Bot",
                                        NSFont.boldSystemFontOfSize_(20), NSColor.blackColor(),
                                        NSTextAlignmentCenter))

        self.status = self._label(NSMakeRect(20, wh - 60, ww - 40, 18),
                                  "就绪 — 请在微信中打开天天象棋并开始一局",
                                  NSFont.systemFontOfSize_(11),
                                  NSColor.colorWithRed_green_blue_alpha_(0.0, 0.45, 0.75, 1),
                                  NSTextAlignmentCenter)
        content.addSubview_(self.status)

        # --- Board (left) ---
        board_size = 380
        board_y = wh - 70 - board_size
        self.boardView = BoardView.alloc().initWithFrame_(
            NSMakeRect(15, board_y, board_size, board_size))
        content.addSubview_(self.boardView)

        # --- Right panel ---
        rx = board_size + 30
        rw = ww - rx - 15
        ry = board_y

        # Info box
        infoBox = NSBox.alloc().initWithFrame_(NSMakeRect(rx, ry + board_size - 140, rw, 140))
        infoBox.setTitle_("信息")
        infoBox.setTitleFont_(NSFont.boldSystemFontOfSize_(11))
        content.addSubview_(infoBox)
        ic = infoBox.contentView()

        y = 95
        ic.addSubview_(self._label(NSMakeRect(8, y, 60, 18), "执棋:", NSFont.boldSystemFontOfSize_(11),
                                   NSColor.darkGrayColor(), NSTextAlignmentLeft))
        self.sideLabel = self._label(NSMakeRect(68, y, 200, 18), "—",
                                     NSFont.systemFontOfSize_(11), NSColor.blackColor(), NSTextAlignmentLeft)
        ic.addSubview_(self.sideLabel)

        y -= 24
        ic.addSubview_(self._label(NSMakeRect(8, y, 60, 18), "步数:", NSFont.boldSystemFontOfSize_(11),
                                   NSColor.darkGrayColor(), NSTextAlignmentLeft))
        self.moveLabel = self._label(NSMakeRect(68, y, 200, 18), "0",
                                     NSFont.fontWithName_size_("Menlo", 13), NSColor.blackColor(), NSTextAlignmentLeft)
        ic.addSubview_(self.moveLabel)

        y -= 24
        ic.addSubview_(self._label(NSMakeRect(8, y, 60, 18), "评分:", NSFont.boldSystemFontOfSize_(11),
                                   NSColor.darkGrayColor(), NSTextAlignmentLeft))
        self.evalLabel = self._label(NSMakeRect(68, y, 200, 18), "—",
                                     NSFont.fontWithName_size_("Menlo", 13), NSColor.blackColor(), NSTextAlignmentLeft)
        ic.addSubview_(self.evalLabel)

        y -= 24
        ic.addSubview_(self._label(NSMakeRect(8, y, 60, 18), "走法:", NSFont.boldSystemFontOfSize_(11),
                                   NSColor.darkGrayColor(), NSTextAlignmentLeft))
        self.moveInfoLabel = self._label(NSMakeRect(68, y, 200, 18), "—",
                                         NSFont.systemFontOfSize_(13), NSColor.blackColor(), NSTextAlignmentLeft)
        ic.addSubview_(self.moveInfoLabel)

        # History box
        histBox = NSBox.alloc().initWithFrame_(NSMakeRect(rx, ry, rw, board_size - 150))
        histBox.setTitle_("走棋记录")
        histBox.setTitleFont_(NSFont.boldSystemFontOfSize_(11))
        content.addSubview_(histBox)

        histScroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(5, 5, rw - 14, board_size - 180))
        histScroll.setHasVerticalScroller_(True)
        self.histView = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, rw - 18, board_size - 180))
        self.histView.setFont_(NSFont.fontWithName_size_("Menlo", 11))
        self.histView.setEditable_(False)
        self.histView.setBackgroundColor_(NSColor.whiteColor())
        histScroll.setDocumentView_(self.histView)
        histBox.contentView().addSubview_(histScroll)

        # --- Log area (bottom) ---
        log_h = board_y - 55
        logBox = NSBox.alloc().initWithFrame_(NSMakeRect(15, 50, ww - 30, log_h))
        logBox.setTitle_("日志")
        logBox.setTitleFont_(NSFont.boldSystemFontOfSize_(11))
        content.addSubview_(logBox)

        logScroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(5, 5, ww - 44, log_h - 25))
        logScroll.setHasVerticalScroller_(True)
        self.logView = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, ww - 48, log_h - 25))
        self.logView.setFont_(NSFont.fontWithName_size_("Menlo", 9))
        self.logView.setTextColor_(NSColor.blackColor())
        self.logView.setBackgroundColor_(NSColor.whiteColor())
        self.logView.setEditable_(False)
        self.logView.setAutoresizingMask_(2)
        logScroll.setDocumentView_(self.logView)
        logBox.contentView().addSubview_(logScroll)

        # --- Buttons ---
        self.startBtn = NSButton.alloc().initWithFrame_(NSMakeRect(ww // 2 - 160, 8, 150, 36))
        self.startBtn.setTitle_("▶ 开始下棋")
        self.startBtn.setBezelStyle_(NSBezelStyleRounded)
        self.startBtn.setFont_(NSFont.boldSystemFontOfSize_(14))
        self.startBtn.setTarget_(self)
        self.startBtn.setAction_("toggle:")
        content.addSubview_(self.startBtn)

        quitBtn = NSButton.alloc().initWithFrame_(NSMakeRect(ww // 2 + 10, 8, 150, 36))
        quitBtn.setTitle_("退出")
        quitBtn.setBezelStyle_(NSBezelStyleRounded)
        quitBtn.setFont_(NSFont.systemFontOfSize_(14))
        quitBtn.setTarget_(self)
        quitBtn.setAction_("quit:")
        content.addSubview_(quitBtn)

        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    @objc.python_method
    def _label(self, frame, text, font, color, alignment):
        lbl = NSTextField.alloc().initWithFrame_(frame)
        lbl.setStringValue_(text)
        lbl.setFont_(font)
        lbl.setTextColor_(color)
        lbl.setBackgroundColor_(NSColor.clearColor())
        lbl.setBezeled_(False)
        lbl.setEditable_(False)
        lbl.setAlignment_(alignment)
        return lbl

    def log_msg(self, msg):
        self.log_buffer.append(msg)

    @objc.python_method
    def _parse_and_update(self, msg):
        # Move: [3] h9g7 (-0.5) or [8] b6b0 (M3)
        m = re.match(r'\[(\d+)\]\s+(\S+)\s+\(([^)]+)\)', msg)
        if m:
            self.move_num = int(m.group(1))
            uci_move = m.group(2)
            ev = m.group(3)
            self.moveLabel.setStringValue_(str(self.move_num))
            self.evalLabel.setStringValue_(ev)

            # Convert to Chinese notation
            if self.current_board:
                chinese = uci_to_chinese(uci_move, self.current_board, self.playing_red)
            else:
                chinese = uci_move
            self.moveInfoLabel.setStringValue_(chinese)

            # Apply our move to board display immediately
            if self.current_board and len(uci_move) >= 4:
                self._apply_uci_move(uci_move)

            # Color eval
            if ev.startswith('+') or ev.startswith('M'):
                self.evalLabel.setTextColor_(NSColor.colorWithRed_green_blue_alpha_(0.0, 0.6, 0.0, 1))
            elif ev.startswith('-'):
                self.evalLabel.setTextColor_(NSColor.colorWithRed_green_blue_alpha_(0.8, 0.0, 0.0, 1))
            else:
                self.evalLabel.setTextColor_(NSColor.blackColor())

            # Add to history (my move)
            my_color = "red" if self.playing_red else "black"
            self._add_move(my_color, chinese, ev)
            return

        # Side
        if 'You play: RED' in msg:
            self.playing_red = True
            self.sideLabel.setStringValue_("红方 (先手)")
            self.sideLabel.setTextColor_(NSColor.redColor())
            self.status.setStringValue_("下棋中 — 红方")
        elif 'You play: BLACK' in msg:
            self.playing_red = False
            self.sideLabel.setStringValue_("黑方 (后手)")
            self.sideLabel.setTextColor_(NSColor.blackColor())
            self.status.setStringValue_("下棋中 — 黑方")

        # Board lines
        if re.match(r'^[.RNBAKCPrnbakcp ]+$', msg) and len(msg) > 10:
            self.board_lines.append(msg)
            if len(self.board_lines) == 10:
                self._parse_board(self.board_lines)
                self.board_lines = []
            return
        elif self.board_lines and not re.match(r'^[.RNBAKCPrnbakcp ]+$', msg):
            self.board_lines = []

        # Opponent move: parse Δ line to extract move
        # Format: Δ (3,5)f6: .→c, (3,6)g6: c→.
        if msg.startswith('Δ '):
            self._parse_opponent_move(msg)

    @objc.python_method
    def _parse_board(self, lines):
        board = [[None] * 9 for _ in range(10)]
        for r, line in enumerate(lines):
            cells = line.split()
            for c, cell in enumerate(cells):
                if c < 9 and cell != '.':
                    board[r][c] = cell
        self.current_board = board
        self.boardView.set_board(board)

    @objc.python_method
    def _parse_opponent_move(self, msg):
        """Extract opponent move from Δ line and add to history."""
        # Format: Δ (row,col)name: old→new, ...
        # Normal move: piece→. at source, .→piece at dest
        # Capture: piece→. at source, old_piece→new_piece at dest
        changes = re.findall(r'\((\d+),(\d+)\)\w+:\s*(\S+)→(\S+)', msg)
        if not changes:
            return

        sources = []   # cells where a piece disappeared
        dests = []     # cells where a piece appeared or changed

        for r_str, c_str, old, new in changes:
            r, c = int(r_str), int(c_str)
            if old != '.' and new == '.':
                # Piece left this cell (source of move)
                sources.append((r, c, old))
            elif old == '.' and new != '.':
                # Piece arrived at empty cell (destination)
                dests.append((r, c, new))
            elif old != '.' and new != '.' and old != new:
                # Piece replaced another (capture destination)
                dests.append((r, c, new))

        # Determine opponent's color
        opp_is_red = not self.playing_red

        # Find the opponent's piece that moved
        source = None
        dest = None
        piece = None

        for r, c, p in sources:
            is_red = p in RED_PIECES
            if is_red == opp_is_red:
                source = (r, c)
                piece = p
                break

        if source and piece:
            # Find where this piece went
            for r, c, p in dests:
                if p == piece:
                    dest = (r, c)
                    break
            # If not found by piece match, use any dest with matching color
            if not dest:
                for r, c, p in dests:
                    is_red = p in RED_PIECES
                    if is_red == opp_is_red:
                        dest = (r, c)
                        break

        # Fallback: just use first source and first dest
        if not source and sources:
            source = (sources[0][0], sources[0][1])
            piece = sources[0][2]
        if not dest and dests:
            dest = (dests[0][0], dests[0][1])
            if not piece:
                piece = dests[0][2]

        if source and dest and piece:
            opp_is_red = not self.playing_red
            chinese = _make_notation(piece, source[0], source[1], dest[0], dest[1],
                                     self.current_board, self.playing_red, is_red=opp_is_red)
            opp_color = "black" if self.playing_red else "red"
            self._add_move(opp_color, chinese, "")

    @objc.python_method
    def _apply_uci_move(self, move):
        """Apply a UCI move to current_board and update display."""
        fc, fr = ord(move[0]) - ord('a'), int(move[1])
        tc, tr = ord(move[2]) - ord('a'), int(move[3])
        if self.playing_red:
            from_r, from_c = 9 - fr, fc
            to_r, to_c = 9 - tr, tc
        else:
            from_r, from_c = fr, 8 - fc
            to_r, to_c = tr, 8 - tc
        if 0 <= from_r < 10 and 0 <= from_c < 9 and 0 <= to_r < 10 and 0 <= to_c < 9:
            piece = self.current_board[from_r][from_c]
            self.current_board[from_r][from_c] = None
            self.current_board[to_r][to_c] = piece
            self.boardView.set_board(self.current_board, (from_r, from_c, to_r, to_c))

    @objc.python_method
    def _add_move(self, color, chinese, ev):
        """Add a move to the round-based history."""
        if color == 'red':
            # Start a new round
            self.round_counter += 1
            self.rounds.append({
                'num': self.round_counter,
                'red': chinese, 'black': '',
                'eval': ev,
            })
        else:  # black
            if self.rounds and not self.rounds[-1]['black']:
                # Fill black slot in current round
                self.rounds[-1]['black'] = chinese
                if ev:
                    self.rounds[-1]['eval'] = ev
            else:
                # Black moves first (we play black, opponent red already moved unknown)
                # Or new round where we don't know red's move
                self.round_counter += 1
                self.rounds.append({
                    'num': self.round_counter,
                    'red': '', 'black': chinese,
                    'eval': ev,
                })
        self._render_history()

    @objc.python_method
    def _render_history(self):
        """Render history with colors: red moves in red, black moves in black."""
        storage = self.histView.textStorage()
        storage.mutableString().setString_("")

        hist_font = NSFont.fontWithName_size_("STKaiti", 13) or NSFont.systemFontOfSize_(13)
        mono_font = NSFont.fontWithName_size_("Menlo", 11)
        red_attrs = {
            NSForegroundColorAttributeName: NSColor.redColor(),
            NSFontAttributeName: hist_font,
        }
        black_attrs = {
            NSForegroundColorAttributeName: NSColor.colorWithRed_green_blue_alpha_(0.1, 0.1, 0.1, 1),
            NSFontAttributeName: hist_font,
        }
        gray_attrs = {
            NSForegroundColorAttributeName: NSColor.grayColor(),
            NSFontAttributeName: mono_font,
        }

        for r in self.rounds:
            # Round number
            num_str = NSAttributedString.alloc().initWithString_attributes_(
                f" {r['num']:>2}.  ", gray_attrs)
            storage.appendAttributedString_(num_str)

            # Red move
            if r['red']:
                red_str = r['red']
                red_as = NSAttributedString.alloc().initWithString_attributes_(
                    f"{red_str:<8s}", red_attrs)
            else:
                placeholder_attrs = {
                    NSForegroundColorAttributeName: NSColor.grayColor(),
                    NSFontAttributeName: hist_font,
                }
                red_as = NSAttributedString.alloc().initWithString_attributes_(
                    "－－－－    ", placeholder_attrs)
            storage.appendAttributedString_(red_as)

            # Separator
            sep = NSAttributedString.alloc().initWithString_attributes_("  ", gray_attrs)
            storage.appendAttributedString_(sep)

            # Black move (always output fixed width to keep eval aligned)
            black_str = r['black'] if r['black'] else '        '
            black_as = NSAttributedString.alloc().initWithString_attributes_(
                f"{black_str:<8s}", black_attrs if r['black'] else gray_attrs)
            storage.appendAttributedString_(black_as)

            # Eval (always at far right)
            ev_text = f"  {r['eval']}" if r['eval'] else ""
            if ev_text:
                ev_as = NSAttributedString.alloc().initWithString_attributes_(
                    ev_text, gray_attrs)
                storage.appendAttributedString_(ev_as)

            nl = NSAttributedString.alloc().initWithString_attributes_("\n", gray_attrs)
            storage.appendAttributedString_(nl)

        rng = self.histView.string().length()
        self.histView.scrollRangeToVisible_(NSRange(rng, 0))
        rng = self.histView.string().length()
        self.histView.scrollRangeToVisible_(NSRange(rng, 0))

    @objc.python_method
    def _append_log(self, msg):
        storage = self.logView.textStorage()
        storage.mutableString().appendString_(msg + "\n")
        rng = self.logView.string().length()
        self.logView.scrollRangeToVisible_(NSRange(rng, 0))

    def flushLog_(self, timer):
        while self.log_buffer:
            msg = self.log_buffer.pop(0)
            self._parse_and_update(msg)
            self._append_log(msg)

    def toggle_(self, sender):
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self):
        self.running = True
        self.move_num = 0
        self.board_lines = []
        self.current_board = None
        self.rounds = []
        self.round_counter = 0
        self.startBtn.setTitle_("■ 停止")
        self.status.setStringValue_("正在启动...")
        self.moveLabel.setStringValue_("0")
        self.evalLabel.setStringValue_("—")
        self.moveInfoLabel.setStringValue_("—")
        self.sideLabel.setStringValue_("—")
        self.histView.setString_("")
        self.log_msg("正在初始化 Bot...")

        self.bot_thread = threading.Thread(target=self._run_bot, daemon=True)
        self.bot_thread.start()

    def stop(self):
        self.running = False
        if self.bot:
            self.bot.stop_flag = True
        self.startBtn.setTitle_("▶ 开始下棋")
        self.status.setStringValue_("已停止")
        self.log_msg("Bot 已停止。")

    def _run_bot(self):
        try:
            class LogWriter(io.TextIOBase):
                def __init__(self, callback):
                    self.callback = callback
                def write(self, s):
                    s = s.strip()
                    if s:
                        self.callback(s)
                    return len(s)

            old_stdout = sys.stdout
            sys.stdout = LogWriter(self.log_msg)

            # Fix cv2 recursion — always clean path before any imports
            sys.path[:] = [p for p in sys.path if 'cv2' not in p.split(os.sep)]
            if 'cv2' not in sys.modules:
                import cv2  # noqa

            # Clean bot module so it re-reads fresh state on restart
            for key in list(sys.modules.keys()):
                if key == 'xiangqi_bot':
                    del sys.modules[key]

            # Replace xiangqi_cnn with ONNX version
            sys.path.insert(0, BUNDLE_DIR)
            import xiangqi_cnn_onnx
            import types
            fake_cnn = types.ModuleType('xiangqi_cnn')
            fake_cnn.PieceClassifierCNN = xiangqi_cnn_onnx.PieceClassifierCNN
            fake_cnn.CLASSES = xiangqi_cnn_onnx.CLASSES
            fake_cnn.CLASS_TO_IDX = xiangqi_cnn_onnx.CLASS_TO_IDX
            fake_cnn.PIECE_TO_DIR = {
                '_': 'empty',
                'R': 'red_R', 'N': 'red_N', 'B': 'red_B', 'A': 'red_A',
                'K': 'red_K', 'C': 'red_C', 'P': 'red_P',
                'r': 'black_r', 'n': 'black_n', 'b': 'black_b', 'a': 'black_a',
                'k': 'black_k', 'c': 'black_c', 'p': 'black_p',
            }
            fake_cnn.MODEL_PATH = os.path.join(BUNDLE_DIR, 'xiangqi_cnn.onnx')
            fake_cnn.collect_from_screenshot = lambda *a, **kw: 0
            sys.modules['xiangqi_cnn'] = fake_cnn

            from xiangqi_bot import Bot, CALIB_PATH

            # Always force fresh calibration in the app
            if os.path.exists(CALIB_PATH):
                os.remove(CALIB_PATH)

            self.bot = Bot()
            try:
                self.bot.run()
            except KeyboardInterrupt:
                pass
            except Exception as e:
                self.log_msg(f"错误: {e}")

            sys.stdout = old_stdout
            if self.running:
                self.status.setStringValue_("对局结束")
                self.stop()

        except Exception as e:
            sys.stdout = sys.__stdout__
            self.log_msg(f"错误: {e}")
            self.status.setStringValue_(f"出错了: {e}")
            self.stop()

    def quit_(self, sender):
        self.running = False
        if self.bot:
            self.bot.stop_flag = True
        NSApp.terminate_(None)

    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        return True


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
