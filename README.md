# Xiangqi Bot

Auto-play Chinese chess (天天象棋) on macOS using Pikafish engine + CNN vision + screen automation.

**中文说明见 [下方](#天天象棋机器人)。**

## How it works

1. **Pikafish** (~3000 Elo) calculates the best move
2. **CNN classifier** (PyTorch, 15-class, 100% val accuracy) identifies pieces from screenshots
3. **FEN validation** enforces piece count rules and auto-corrects misclassifications using confidence scores
4. **Double-shot parsing** takes a second screenshot when confidence is low, averages probabilities to handle animation artifacts
5. **Quartz CGEvent** clicks to execute moves

```
Screenshot → CNN parse board → FEN validation → Compare with tracked state → Detect opponent move
                  ↓                                                                ↓
          (low confidence?)                              Pikafish best move ← Current FEN
                  ↓                                                                ↓
          2nd screenshot →                               CGEvent click → Execute move
          average probs
```

## Platform

**macOS only.** The bot uses macOS-specific APIs (Quartz CGEvent for clicking, `screencapture` for screenshots). The included Pikafish binary is compiled for macOS x86_64 (runs on Apple Silicon via Rosetta).

## Setup

### 1. Install Python dependencies

```bash
pip install opencv-python numpy pyautogui torch torchvision
```

### 2. Install Pikafish engine

Pikafish is a Chinese chess engine (forked from Stockfish, rewritten for xiangqi rules). It's a C++ compiled binary that the bot communicates with via UCI protocol over stdin/stdout.

The repo includes a pre-built macOS x86_64 binary. If it doesn't work on your machine, build from source:

```bash
git clone https://github.com/official-pikafish/Pikafish.git
cd Pikafish/src
make -j profile-build ARCH=apple-silicon   # for Apple Silicon
# or: make -j profile-build ARCH=x86-64   # for Intel Mac
cp pikafish ../../pikafish                  # copy binary to project root
```

The NNUE file (`pikafish.nnue`) is already included — it's the neural network weights that Pikafish uses to evaluate positions. Pikafish looks for it in the working directory automatically.

### 3. Run

```bash
# Open 天天象棋 in WeChat, start a game, then:
python3 xiangqi_bot.py
```

On first run, follow the calibration prompts — move your mouse to 2 corners of the board so the bot knows where the grid is. Calibration is saved to `calib.json` and reused until you delete it.

Supports both red and black sides. Auto-detects orientation from king position.

## CNN Training

Training data is auto-collected during gameplay. Debug patches are saved to `debug/` organized by piece type for easy review.

### Data augmentation

The augmentation pipeline generates variants per original image:
- **Brightness** (0.8x, 1.2x)
- **Shift** (2px jitter in 4 directions)
- **Scale** (0.85x, 1.15x, 1.25x — simulates selected/enlarged pieces)
- **Low-resolution** (downscale to 20/24/30px then upscale — simulates small board blurriness)

### Improve accuracy

1. Play a game — debug patches auto-save to `debug/`
2. Review `debug/red_C/`, `debug/red_P/` etc. for misclassifications
3. Move wrong images to the correct `cnn_data/<piece_type>/` folder
4. Retrain:

```bash
python3 xiangqi_cnn.py augment   # regenerate augmented data
python3 xiangqi_cnn.py train --epochs 30
```

### Other CNN commands

```bash
python3 xiangqi_cnn.py collect   # Collect from initial position
python3 xiangqi_cnn.py augment   # Augment data (brightness, shift, scale, low-res simulation)
python3 xiangqi_cnn.py test      # Test on current board
```

## Continuous play (multi-game auto-restart)

`continuous_play.py` is a standalone CLI supervisor that wraps `Bot` in a
loop: when a game ends, it automatically clicks through the result popups
and starts the next game. Runs headless — no GUI dependency.

```bash
python3 continuous_play.py play                # unlimited games
python3 continuous_play.py play --max-games 5  # stop after 5 games
python3 continuous_play.py test-templates      # verify templates match current screen
python3 continuous_play.py detect-end          # one-shot end-of-game check
python3 continuous_play.py diag                # dump screenshot + window info
python3 continuous_play.py crop-templates      # print manual cropping workflow
```

**How it works:**

1. **Game-end detection** — strict (0.95) `cv2.matchTemplate` on a set of
   end-screen templates (`popup_end_banner`, `popup_level_up`,
   `popup_badge_earned`, `btn_play_again`, `btn_close_x_top`, `btn_confirm`,
   `btn_switch_opponent`). Also catches Pikafish `(none)` (mate/stalemate)
   and a 150s opponent-stuck safety net.
2. **Tiered recovery** — when a game ends, cycle through template clicks
   → blind relative-coord clicks → ESC×3 → refind-window until the board
   becomes visible again.
3. **Multi-scale template matching** — each template is matched at scales
   0.85–1.15x against a 0.35x downscaled screenshot for ~7x speedup over
   naive full-res matching. Stepped thresholds (0.92 → 0.88) with
   early-exit at 0.97+.
4. **Click verification** — every click takes before/after screenshots and
   uses `bot.images_changed` to confirm the screen reacted. Failed clicks
   go on a 30s cooldown to prevent death loops.
5. **Failure snapshots** — if recovery times out, a screenshot plus state
   log is dumped to `continuous_play_debug/<timestamp>/`. End-of-game
   screenshots are auto-saved to `end_snapshots/` for future template
   tuning.
6. **Ctrl+C** — graceful stop on first press; hard force-exit after 4s via
   a daemon-thread `os._exit(130)` watchdog, so a stuck subprocess can't
   block shutdown.

**Templates** live in `templates/`. Missing templates degrade gracefully
(tier 1 just has fewer candidates; tiers 2–4 still run).

**Does not modify** `xiangqi_bot.py` or `app.py` — both are still
available as the single-game entry points:

| Entry point | Multi-game? |
|---|---|
| `python3 xiangqi_bot.py` | no (single game) |
| `python3 app.py` / `天天象棋Bot.app` | no (GUI, single game) |
| `python3 continuous_play.py play` | **yes** (auto-restart) |

## macOS App

A native macOS GUI (`天天象棋Bot.app`) with board display, move history (四字记录法), and start/stop controls.

### Build

```bash
pip install pyinstaller
python3 -m PyInstaller xiangqi_bot.spec --noconfirm
```

The app is output to `dist/天天象棋Bot.app`. It bundles Pikafish, the ONNX model, and all Python dependencies.

### Run directly (without building)

```bash
python3 app.py
```

## Files

| File | Description |
|------|-------------|
| `app.py` | macOS GUI app (AppKit/PyObjC, board view, move history) |
| `xiangqi_bot.py` | Bot engine (game loop, move execution, double-shot parsing) |
| `continuous_play.py` | Multi-game supervisor (auto-restart between games) |
| `xiangqi_cnn.py` | CNN model, training, inference, FEN validation |
| `xiangqi_cnn_onnx.py` | ONNX inference wrapper (no PyTorch needed) |
| `xiangqi_cnn.onnx` | Trained model weights (ONNX format) |
| `xiangqi_bot.spec` | PyInstaller build spec for macOS app |
| `hook-cv2.py` | PyInstaller runtime hook for OpenCV |
| `pikafish` | Pikafish engine binary (macOS x86_64) |
| `pikafish.nnue` | Neural network evaluation file for Pikafish |
| `cnn_data/` | Training data by piece type (`red_R/`, `black_r/`, `empty/`, etc.) |
| `debug/` | Debug patches from latest game session |
| `calib.json` | Grid calibration data |
| `templates/` | Popup button templates used by `continuous_play.py` |
| `end_snapshots/` | Auto-saved end-of-game screenshots (for template tuning) |
| `continuous_play_debug/` | Failure dumps from `continuous_play.py` (screenshot + state) |

---

# 天天象棋机器人

在 macOS 上自动玩天天象棋，基于 Pikafish 引擎 + CNN 视觉识别 + 屏幕自动化。

## 工作原理

1. **Pikafish**（约 3000 Elo）计算最佳走法
2. **CNN 分类器**（PyTorch，15 类，验证集 100% 准确率）从截图识别棋子
3. **FEN 校验** 强制棋子数量规则，用置信度自动修正误分类
4. **双发解析** 置信度低时再截一次图，取概率平均以应对动画瞬态
5. **Quartz CGEvent** 点击执行走棋

```
截图 → CNN 解析棋盘 → FEN 校验 → 对比跟踪状态 → 检测对方走棋
         ↓                                            ↓
    (置信度低?)                        Pikafish 最佳走法 ← 当前 FEN
         ↓                                            ↓
    再截一张 →                       CGEvent 点击 → 执行走棋
    取概率平均
```

## 运行平台

**仅 macOS。** Bot 用到 macOS 专有 API（Quartz CGEvent 点击、`screencapture` 截屏）。仓库自带的 Pikafish 二进制是 macOS x86_64 编译的（Apple Silicon 通过 Rosetta 跑）。

## 安装

### 1. 安装 Python 依赖

```bash
pip install opencv-python numpy pyautogui torch torchvision
```

### 2. 安装 Pikafish 引擎

Pikafish 是一个象棋引擎（从 Stockfish fork 而来，针对象棋规则重写），C++ 编译的二进制，bot 通过 UCI 协议经 stdin/stdout 与它通信。

仓库里自带预编译好的 macOS x86_64 二进制。如果你机器上跑不起来，从源码编译：

```bash
git clone https://github.com/official-pikafish/Pikafish.git
cd Pikafish/src
make -j profile-build ARCH=apple-silicon   # Apple Silicon
# 或 Intel Mac: make -j profile-build ARCH=x86-64
cp pikafish ../../pikafish                  # 把二进制拷到项目根目录
```

NNUE 文件 (`pikafish.nnue`) 已经包含——这是 Pikafish 用来评估局面的神经网络权重，会自动从工作目录加载。

### 3. 运行

```bash
# 先在 WeChat 里打开天天象棋开一局，然后：
python3 xiangqi_bot.py
```

首次运行会有校准提示——把鼠标移到棋盘的两个角让 bot 知道网格位置。校准结果存到 `calib.json`，删除前一直复用。

红黑两方都支持。从将/帅位置自动检测执棋方向。

## CNN 训练

训练数据在对局中自动收集。Debug 补丁按棋子类型保存到 `debug/` 方便审查。

### 数据增强

针对每张原图，增强管线生成多个变体：
- **亮度**（0.8x、1.2x）
- **位移**（四方向 2 像素抖动）
- **缩放**（0.85x、1.15x、1.25x——模拟选中/放大的棋子）
- **低分辨率**（降采样到 20/24/30 像素再升回——模拟小棋盘的模糊）

### 提升识别准确率

1. 下一盘棋——debug 补丁自动保存到 `debug/`
2. 审查 `debug/red_C/`、`debug/red_P/` 等目录找出误分类
3. 把错分的图片移到正确的 `cnn_data/<棋子类型>/` 目录
4. 重训：

```bash
python3 xiangqi_cnn.py augment   # 重新生成增强数据
python3 xiangqi_cnn.py train --epochs 30
```

### 其他 CNN 命令

```bash
python3 xiangqi_cnn.py collect   # 从初始局面收集
python3 xiangqi_cnn.py augment   # 增强数据
python3 xiangqi_cnn.py test      # 测试当前棋盘
```

## 连续对局（自动续局）

`continuous_play.py` 是一个独立 CLI 监督脚本，把 `Bot` 包在循环里：一局结束时自动点掉结算弹窗开始下一局。无 GUI 依赖，纯命令行跑。

```bash
python3 continuous_play.py play                # 无限对局
python3 continuous_play.py play --max-games 5  # 跑 5 局停
python3 continuous_play.py test-templates      # 验证模板对当前屏幕的命中情况
python3 continuous_play.py detect-end          # 一次性检测游戏是否结束
python3 continuous_play.py diag                # 保存截图和窗口信息
python3 continuous_play.py crop-templates      # 打印手动裁剪模板的说明
```

**工作原理：**

1. **游戏结束检测** — 用 0.95 的严格阈值对一组结束画面模板跑 `cv2.matchTemplate`
   （`popup_end_banner`、`popup_level_up`、`popup_badge_earned`、
   `btn_play_again`、`btn_close_x_top`、`btn_confirm`、`btn_switch_opponent`）。
   同时捕获 Pikafish 返回 `(none)`（将死/困毙）和 150 秒对方卡住的兜底信号。
2. **分层恢复** — 游戏结束后依次尝试：模板点击 → 盲点相对坐标点击 →
   ESC × 3 → 重新查找窗口，直到棋盘重新可见。
3. **多尺度模板匹配** — 每个模板在 0.85–1.15 尺度区间匹配，针对 0.35x
   降采样的截图操作，比全分辨率快约 7 倍。阶梯阈值（0.92 → 0.88），
   达到 0.97+ 时提前退出。
4. **点击校验** — 每次点击前后都截图，用 `bot.images_changed` 确认屏幕
   有变化。失败的点击进入 30 秒冷却，防止死循环。
5. **失败截图** — 恢复超时时，截图和状态日志会存到
   `continuous_play_debug/<时间戳>/`。游戏结束画面自动存到
   `end_snapshots/` 便于后续调校模板。
6. **Ctrl+C** — 首次按平缓停止；4 秒后 daemon 线程看门狗 `os._exit(130)`
   强制退出，即使有卡住的子进程也能立刻退。

**模板** 存放在 `templates/`。缺少模板会优雅降级（tier 1 的候选减少，
但 tier 2-4 依然运行）。

**不修改** `xiangqi_bot.py` 或 `app.py`——两者仍然作为单局入口可用：

| 入口 | 连续对局？ |
|---|---|
| `python3 xiangqi_bot.py` | 否（单局） |
| `python3 app.py` / `天天象棋Bot.app` | 否（GUI，单局） |
| `python3 continuous_play.py play` | **是**（自动续局） |

## macOS 应用

原生 macOS 图形界面 (`天天象棋Bot.app`)，带棋盘显示、走棋记录（四字记录法）和开始/停止控制。

### 构建

```bash
pip install pyinstaller
python3 -m PyInstaller xiangqi_bot.spec --noconfirm
```

应用输出到 `dist/天天象棋Bot.app`。包含 Pikafish、ONNX 模型和所有 Python 依赖。

### 直接运行（不构建）

```bash
python3 app.py
```

## 文件说明

| 文件 | 说明 |
|------|-------------|
| `app.py` | macOS 图形界面应用（AppKit/PyObjC，棋盘视图，走棋记录）|
| `xiangqi_bot.py` | Bot 引擎（游戏循环、走棋执行、双发解析）|
| `continuous_play.py` | 多局监督器（自动续局）|
| `xiangqi_cnn.py` | CNN 模型、训练、推理、FEN 校验 |
| `xiangqi_cnn_onnx.py` | ONNX 推理封装（无需 PyTorch）|
| `xiangqi_cnn.onnx` | 训练好的模型权重（ONNX 格式）|
| `xiangqi_bot.spec` | macOS 应用的 PyInstaller 构建规格 |
| `hook-cv2.py` | OpenCV 的 PyInstaller 运行时 hook |
| `pikafish` | Pikafish 引擎二进制（macOS x86_64）|
| `pikafish.nnue` | Pikafish 的神经网络评估文件 |
| `cnn_data/` | 按棋子类型组织的训练数据（`red_R/`、`black_r/`、`empty/` 等）|
| `debug/` | 最近对局的 debug 补丁 |
| `calib.json` | 棋盘校准数据 |
| `templates/` | `continuous_play.py` 使用的弹窗按钮模板 |
| `end_snapshots/` | 自动保存的结束画面截图（用于调校模板）|
| `continuous_play_debug/` | `continuous_play.py` 的失败转储（截图 + 状态）|
