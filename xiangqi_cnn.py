#!/usr/bin/env python3
"""
CNN piece classifier for Xiangqi Bot.
Collects training data, trains model, and provides inference.

Classes (15): empty + 7 red + 7 black
  _ R N B A K C P r n b a k c p
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "cnn_data")
MODEL_PATH = os.path.join(_SCRIPT_DIR, "xiangqi_cnn.pt")
CELL_SIZE = 48  # Input size for CNN
CLASSES = ['_', 'R', 'N', 'B', 'A', 'K', 'C', 'P', 'r', 'n', 'b', 'a', 'k', 'c', 'p']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
# macOS filesystem is case-insensitive, so R and r map to the same folder!
# Use distinct folder names: red_X for uppercase, black_x for lowercase
PIECE_TO_DIR = {
    '_': 'empty',
    'R': 'red_R', 'N': 'red_N', 'B': 'red_B', 'A': 'red_A',
    'K': 'red_K', 'C': 'red_C', 'P': 'red_P',
    'r': 'black_r', 'n': 'black_n', 'b': 'black_b', 'a': 'black_a',
    'k': 'black_k', 'c': 'black_c', 'p': 'black_p',
}
DIR_TO_PIECE = {v: k for k, v in PIECE_TO_DIR.items()}


# --- CNN Model ---

class PieceNet(nn.Module):
    """Small CNN for 15-class piece classification on 48x48 patches."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(CLASSES)),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# --- Dataset ---

class PieceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []  # (path, label_idx)
        self.transform = transform
        for piece, dirname in PIECE_TO_DIR.items():
            cls_dir = os.path.join(data_dir, dirname)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith('.png'):
                    self.samples.append((os.path.join(cls_dir, fname), CLASS_TO_IDX[piece]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.resize(img, (CELL_SIZE, CELL_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        if self.transform:
            img = self.transform(img)
        return img, label


# --- Data Collection ---

def collect_from_screenshot(img, cols_logical, rows_logical, board, retina_scale,
                            win_x, win_y, cell_w, cell_h, session_id=0):
    """Extract labeled cell patches from a screenshot with known board state.

    Args:
        img: Full window screenshot (retina resolution)
        cols_logical, rows_logical: Grid coordinates in logical space
        board: 10x9 array of piece chars or None
        retina_scale: Retina scaling factor
        win_x, win_y: Window position
        cell_w, cell_h: Cell dimensions in logical pixels
        session_id: Unique ID for this collection session
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    for d in PIECE_TO_DIR.values():
        os.makedirs(os.path.join(DATA_DIR, d), exist_ok=True)

    radius = int(min(cell_w, cell_h) * retina_scale * 0.45)
    count = 0

    for r in range(10):
        for c in range(9):
            px = int((cols_logical[c] - win_x) * retina_scale)
            py = int((rows_logical[r] - win_y) * retina_scale)

            h, w = img.shape[:2]
            x1, y1 = max(0, px - radius), max(0, py - radius)
            x2, y2 = min(w, px + radius), min(h, py + radius)
            patch = img[y1:y2, x1:x2]

            if patch.shape[0] < 20 or patch.shape[1] < 20:
                continue

            piece = board[r][c]
            cls = piece if piece else '_'
            if cls not in PIECE_TO_DIR:
                continue

            dirname = PIECE_TO_DIR[cls]
            fname = f"s{session_id}_r{r}c{c}.png"
            cv2.imwrite(os.path.join(DATA_DIR, dirname, fname), patch)
            count += 1

    return count


def augment_data():
    """Apply augmentation to existing training data."""
    for dirname in PIECE_TO_DIR.values():
        cls_dir = os.path.join(DATA_DIR, dirname)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir) if f.endswith('.png') and '_aug' not in f]
        for fname in files:
            img = cv2.imread(os.path.join(cls_dir, fname))
            if img is None:
                continue
            base = fname.replace('.png', '')

            # Brightness variations
            for i, factor in enumerate([0.8, 1.2]):
                aug = np.clip(img * factor, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(cls_dir, f"{base}_aug_br{i}.png"), aug)

            # Small shift (2-pixel jitter)
            for i, (dx, dy) in enumerate([(2, 0), (-2, 0), (0, 2), (0, -2)]):
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aug = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                cv2.imwrite(os.path.join(cls_dir, f"{base}_aug_sh{i}.png"), aug)

            # Slight rotation
            for i, angle in enumerate([-3, 3]):
                M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
                aug = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                cv2.imwrite(os.path.join(cls_dir, f"{base}_aug_rot{i}.png"), aug)

            # Scale variations (simulate selected/enlarged piece)
            h, w = img.shape[:2]
            for i, scale in enumerate([1.15, 1.25, 0.85]):
                sh, sw = int(h * scale), int(w * scale)
                scaled = cv2.resize(img, (sw, sh))
                # Center crop/pad back to original size
                if scale > 1:
                    y0 = (sh - h) // 2
                    x0 = (sw - w) // 2
                    aug = scaled[y0:y0+h, x0:x0+w]
                else:
                    aug = np.zeros_like(img)
                    y0 = (h - sh) // 2
                    x0 = (w - sw) // 2
                    aug[y0:y0+sh, x0:x0+sw] = scaled
                cv2.imwrite(os.path.join(cls_dir, f"{base}_aug_sc{i}.png"), aug)


# --- Training ---

def train(epochs=30, batch_size=32, lr=0.001):
    """Train the CNN on collected data."""
    # Count samples per class
    print("Dataset:")
    total = 0
    for piece, dirname in PIECE_TO_DIR.items():
        cls_dir = os.path.join(DATA_DIR, dirname)
        n = len([f for f in os.listdir(cls_dir) if f.endswith('.png')]) if os.path.isdir(cls_dir) else 0
        total += n
        print(f"  {piece} ({dirname}): {n}")
    print(f"  Total: {total}")

    if total < 50:
        print("Not enough data! Need at least 50 samples.")
        return

    dataset = PieceDataset(DATA_DIR)
    # Split 80/20
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    model = PieceNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total_samples = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

        train_acc = correct / total_samples

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(1, val_total)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or val_acc > best_acc:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"loss={train_loss/len(train_loader):.3f} "
                  f"train={train_acc:.1%} val={val_acc:.1%}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest val accuracy: {best_acc:.1%}")
    print(f"Model saved to {MODEL_PATH}")


# --- Inference ---

class PieceClassifierCNN:
    """Inference wrapper for the trained CNN."""
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = PieceNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device,
                                               weights_only=True))
        self.model.eval()

    def classify_cell(self, patch):
        """Classify a single cell patch. Returns (piece_char_or_None, confidence)."""
        img = cv2.resize(patch, (CELL_SIZE, CELL_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(1)

        cls = CLASSES[pred.item()]
        return (None if cls == '_' else cls), conf.item()

    def parse_board(self, img, cols_logical, rows_logical, retina_scale, win_x, win_y,
                    cell_w, cell_h):
        """Parse entire board from a single screenshot. Returns 10x9 board array."""
        radius = int(min(cell_w, cell_h) * retina_scale * 0.45)
        board = [[None] * 9 for _ in range(10)]

        for r in range(10):
            for c in range(9):
                px = int((cols_logical[c] - win_x) * retina_scale)
                py = int((rows_logical[r] - win_y) * retina_scale)
                h, w = img.shape[:2]
                x1, y1 = max(0, px - radius), max(0, py - radius)
                x2, y2 = min(w, px + radius), min(h, py + radius)
                patch = img[y1:y2, x1:x2]

                if patch.shape[0] < 20 or patch.shape[1] < 20:
                    continue

                piece, conf = self.classify_cell(patch)
                board[r][c] = piece

        return board


# --- CLI ---

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['collect', 'augment', 'train', 'test'])
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    if args.action == 'collect':
        # Collect from current board position (must have bot running or calibrated)
        print("Collecting data from initial position...")
        # Import bot for screenshot and calibration
        sys.path.insert(0, '/tmp')
        from xiangqi_bot import Bot, INIT_RED, INIT_BLACK

        bot = Bot()
        bot.find_window()
        bot.load_calibration()
        img = bot.screenshot_for_processing()
        bot.detect_orientation(img)

        board = INIT_RED if bot.playing_red else INIT_BLACK
        empty_dir = os.path.join(DATA_DIR, 'empty')
        session_id = len(os.listdir(empty_dir)) if os.path.isdir(empty_dir) else 0

        n = collect_from_screenshot(
            img, bot.cols_logical, bot.rows_logical, board,
            bot.retina_scale, bot.win_x, bot.win_y,
            bot.cell_w, bot.cell_h, session_id)
        print(f"Collected {n} patches (session {session_id})")

    elif args.action == 'augment':
        print("Augmenting data...")
        augment_data()
        for cls in CLASSES:
            cls_dir = os.path.join(DATA_DIR, cls)
            n = len([f for f in os.listdir(cls_dir) if f.endswith('.png')]) if os.path.isdir(cls_dir) else 0
            print(f"  {cls}: {n}")

    elif args.action == 'train':
        train(epochs=args.epochs)

    elif args.action == 'test':
        if not os.path.exists(MODEL_PATH):
            print(f"No model at {MODEL_PATH}!")
            sys.exit(1)

        print("Testing CNN on current board...")
        sys.path.insert(0, '/tmp')
        from xiangqi_bot import Bot

        bot = Bot()
        bot.find_window()
        bot.load_calibration()
        img = bot.screenshot_for_processing()
        bot.detect_orientation(img)

        cnn = PieceClassifierCNN()
        board = cnn.parse_board(
            img, bot.cols_logical, bot.rows_logical,
            bot.retina_scale, bot.win_x, bot.win_y,
            bot.cell_w, bot.cell_h)

        print("\nCNN parsed board:")
        for r in range(10):
            line = " "
            for c in range(9):
                p = board[r][c]
                line += f" {p}" if p else " ."
            print(line)
