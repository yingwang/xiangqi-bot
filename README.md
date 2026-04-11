# Xiangqi Bot / 天天象棋机器人

Auto-play Chinese chess (天天象棋) on macOS using Pikafish engine + CNN vision + screen automation.

在 macOS 上自动玩天天象棋，使用 Pikafish 引擎 + CNN 视觉识别 + 屏幕自动化。

---

## How it works / 工作原理

1. **Pikafish** (~3000 Elo) calculates the best move
2. **CNN classifier** (PyTorch, 15-class, 100% val accuracy) identifies pieces from screenshots
3. **FEN validation** enforces piece count rules and auto-corrects misclassifications using confidence scores
4. **Double-shot parsing** takes a second screenshot when confidence is low, averages probabilities to handle animation artifacts
5. **Quartz CGEvent** clicks to execute moves

1. **Pikafish**（约 3000 Elo）计算最佳走法
2. **CNN 分类器**（PyTorch，15 类，验证集 100% 准确率）从截图里识别棋子
3. **FEN 校验** 强制执行棋子数量规则，用置信度分数自动修正误分类
4. **双发解析** 置信度低时再截一次图，对概率取平均以应对动画瞬态
5. **Quartz CGEvent** 点击执行走棋

```
Screenshot → CNN parse board → FEN validation → Compare with tracked state → Detect opponent move
                  ↓                                                                ↓
          (low confidence?)                              Pikafish best move ← Current FEN
                  ↓                                                                ↓
          2nd screenshot →                               CGEvent click → Execute move
          average probs
```

## Platform / 运行平台

**macOS only.** The bot uses macOS-specific APIs (Quartz CGEvent for clicking, `screencapture` for screenshots). The included Pikafish binary is compiled for macOS x86_64 (runs on Apple Silicon via Rosetta).

**仅 macOS。** Bot 使用 macOS 专有的 API（Quartz CGEvent 点击、`screencapture` 截屏）。仓库里的 Pikafish 二进制是 macOS x86_64 编译的（Apple Silicon 通过 Rosetta 跑）。

## Setup / 安装

### 1. Install Python dependencies / 安装 Python 依赖

```bash
pip install opencv-python numpy pyautogui torch torchvision
```

### 2. Install Pikafish engine / 安装 Pikafish 引擎

Pikafish is a Chinese chess engine (forked from Stockfish, rewritten for xiangqi rules). It's a C++ compiled binary that the bot communicates with via UCI protocol over stdin/stdout.

Pikafish 是一个象棋引擎（fork 自 Stockfish，针对象棋规则重写）。它是一个 C++ 编译的二进制文件，bot 通过 UCI 协议经 stdin/stdout 与它通信。

The repo includes a pre-built macOS x86_64 binary. If it doesn't work on your machine, build from source:

仓库里自带预编译的 macOS x86_64 二进制。如果在你机器上跑不起来，从源码编译：

```bash
git clone https://github.com/official-pikafish/Pikafish.git
cd Pikafish/src
make -j profile-build ARCH=apple-silicon   # for Apple Silicon / 苹果芯片
# or: make -j profile-build ARCH=x86-64   # for Intel Mac
cp pikafish ../../pikafish                  # copy binary to project root
```

The NNUE file (`pikafish.nnue`) is already included — it's the neural network weights that Pikafish uses to evaluate positions. Pikafish looks for it in the working directory automatically.

NNUE 文件 (`pikafish.nnue`) 已经包含 — 这是 Pikafish 用于评估局面的神经网络权重，会自动从工作目录加载。

### 3. Run / 运行

```bash
# Open 天天象棋 in WeChat, start a game, then:
# 在 WeChat 里打开天天象棋开一局，然后：
python3 xiangqi_bot.py
```

On first run, follow the calibration prompts — move your mouse to 2 corners of the board so the bot knows where the grid is. Calibration is saved to `calib.json` and reused until you delete it.

首次运行按照校准提示——把鼠标移动到棋盘的两个角，bot 就知道网格在哪。校准结果会存到 `calib.json`，删除前一直复用。

Supports both red and black sides. Auto-detects orientation from king position.

红黑两方都支持。从将/帅位置自动检测执棋方向。

## CNN Training / CNN 训练

Training data is auto-collected during gameplay. Debug patches are saved to `debug/` organized by piece type for easy review.

训练数据在对局中自动收集。Debug 补丁按棋子类型保存到 `debug/` 便于审查。

### Data augmentation / 数据增强

The augmentation pipeline generates variants per original image:

针对每张原图，增强管线生成多个变体：

- **Brightness / 亮度** (0.8x, 1.2x)
- **Shift / 位移** (2px jitter in 4 directions / 四方向 2 像素抖动)
- **Scale / 缩放** (0.85x, 1.15x, 1.25x — simulates selected/enlarged pieces / 模拟选中/放大的棋子)
- **Low-resolution / 低分辨率** (downscale to 20/24/30px then upscale — simulates small board blurriness / 模拟小棋盘的模糊)

### Improve accuracy / 提升识别准确率

1. Play a game — debug patches auto-save to `debug/`
   下一盘棋——debug 补丁自动保存到 `debug/`
2. Review `debug/red_C/`, `debug/red_P/` etc. for misclassifications
   审查 `debug/red_C/`、`debug/red_P/` 等目录找出误分类
3. Move wrong images to the correct `cnn_data/<piece_type>/` folder
   把错分的图片移到正确的 `cnn_data/<棋子类型>/` 目录
4. Retrain / 重训：

```bash
python3 xiangqi_cnn.py augment   # regenerate augmented data / 重新生成增强数据
python3 xiangqi_cnn.py train --epochs 30
```

### Other CNN commands / 其他 CNN 命令

```bash
python3 xiangqi_cnn.py collect   # Collect from initial position / 从初始局面收集
python3 xiangqi_cnn.py augment   # Augment data / 增强数据
python3 xiangqi_cnn.py test      # Test on current board / 测试当前棋盘
```

## Continuous play (multi-game auto-restart) / 自动连续下棋

`continuous_play.py` is a standalone CLI supervisor that wraps `Bot` in a
loop: when a game ends, it automatically clicks through the result popups
and starts the next game. Runs headless — no GUI dependency.

`continuous_play.py` 是一个独立的 CLI 监督脚本，把 `Bot` 包装在循环里：
一局结束时自动点掉结算弹窗开始下一局。无 GUI 依赖，纯命令行运行。

```bash
python3 continuous_play.py play                # unlimited games / 无限对局
python3 continuous_play.py play --max-games 5  # stop after 5 games / 5 局后停止
python3 continuous_play.py test-templates      # verify templates match current screen / 验证模板命中
python3 continuous_play.py detect-end          # one-shot end-of-game check / 一次性检测游戏是否结束
python3 continuous_play.py diag                # dump screenshot + window info / 输出截图和窗口信息
python3 continuous_play.py crop-templates      # print manual cropping workflow / 打印手动裁剪说明
```

**How it works / 工作原理：**

1. **Game-end detection / 游戏结束检测** — strict (0.95) `cv2.matchTemplate` on a set of
   end-screen templates (`popup_end_banner`, `popup_level_up`,
   `popup_badge_earned`, `btn_play_again`, `btn_close_x_top`, `btn_confirm`,
   `btn_switch_opponent`). Also catches Pikafish `(none)` (mate/stalemate)
   and a 150s opponent-stuck safety net.

   用 0.95 的严格阈值对一组结束画面模板跑 `cv2.matchTemplate`。同时捕获 Pikafish 返回 `(none)`（将死/困毙）和 150 秒对方卡住的兜底信号。

2. **Tiered recovery / 分层恢复** — when a game ends, cycle through template clicks
   → blind relative-coord clicks → ESC×3 → refind-window until the board
   becomes visible again.

   游戏结束时依次尝试：模板点击 → 盲点相对坐标点击 → ESC × 3 → 重新查找窗口，直到棋盘重新可见。

3. **Multi-scale template matching / 多尺度模板匹配** — each template is matched at scales
   0.85–1.15x against a 0.35x downscaled screenshot for ~7x speedup over
   naive full-res matching. Stepped thresholds (0.92 → 0.88) with
   early-exit at 0.97+.

   每个模板在 0.85–1.15 尺度区间匹配，针对 0.35x 降采样的截图操作，比全分辨率快约 7 倍。阶梯阈值 (0.92 → 0.88)，0.97+ 时提前退出。

4. **Click verification / 点击校验** — every click takes before/after screenshots and
   uses `bot.images_changed` to confirm the screen reacted. Failed clicks
   go on a 30s cooldown to prevent death loops.

   每次点击前后截图，用 `bot.images_changed` 确认屏幕有变化。失败的点击进入 30 秒冷却，防止死循环。

5. **Failure snapshots / 失败截图** — if recovery times out, a screenshot plus state
   log is dumped to `continuous_play_debug/<timestamp>/`. End-of-game
   screenshots are auto-saved to `end_snapshots/` for future template
   tuning.

   恢复超时会把截图和状态日志存到 `continuous_play_debug/<时间戳>/`。结束画面自动存到 `end_snapshots/` 便于后续调校模板。

6. **Ctrl+C** — graceful stop on first press; hard force-exit after 4s via
   a daemon-thread `os._exit(130)` watchdog, so a stuck subprocess can't
   block shutdown.

   第一次按 Ctrl+C 平缓停止；4 秒后由 daemon 线程看门狗 `os._exit(130)` 强制退出，即使有卡住的子进程也能立刻退。

**Templates** live in `templates/`. Missing templates degrade gracefully
(tier 1 just has fewer candidates; tiers 2–4 still run).

模板文件存放在 `templates/`。缺少模板会优雅降级（tier 1 的候选减少，但 tier 2-4 依然运行）。

**Does not modify** `xiangqi_bot.py` or `app.py` — both are still
available as the single-game entry points:

**不修改** `xiangqi_bot.py` 或 `app.py`——两者仍然作为单局入口可用：

| Entry point / 入口 | Multi-game? / 连续对局？ |
|---|---|
| `python3 xiangqi_bot.py` | no / 否 (single game / 单局) |
| `python3 app.py` / `天天象棋Bot.app` | no / 否 (GUI, single game / 图形界面单局) |
| `python3 continuous_play.py play` | **yes / 是** (auto-restart / 自动续局) |

## macOS App / macOS 应用

A native macOS GUI (`天天象棋Bot.app`) with board display, move history (四字记录法), and start/stop controls.

原生 macOS 图形界面 (`天天象棋Bot.app`)，带棋盘显示、走棋记录（四字记录法）和开始/停止控制。

### Build / 构建

```bash
pip install pyinstaller
python3 -m PyInstaller xiangqi_bot.spec --noconfirm
```

The app is output to `dist/天天象棋Bot.app`. It bundles Pikafish, the ONNX model, and all Python dependencies.

应用输出到 `dist/天天象棋Bot.app`。包含 Pikafish、ONNX 模型和所有 Python 依赖。

### Run directly (without building) / 直接运行（不构建）

```bash
python3 app.py
```

## Files / 文件说明

| File / 文件 | Description / 说明 |
|------|-------------|
| `app.py` | macOS GUI app (AppKit/PyObjC, board view, move history) / macOS 图形界面应用（AppKit/PyObjC，棋盘视图，走棋记录）|
| `xiangqi_bot.py` | Bot engine (game loop, move execution, double-shot parsing) / Bot 引擎（游戏循环、走棋执行、双发解析）|
| `continuous_play.py` | Multi-game supervisor (auto-restart between games) / 多局监督器（自动续局）|
| `xiangqi_cnn.py` | CNN model, training, inference, FEN validation / CNN 模型、训练、推理、FEN 校验 |
| `xiangqi_cnn_onnx.py` | ONNX inference wrapper (no PyTorch needed) / ONNX 推理封装（无需 PyTorch）|
| `xiangqi_cnn.onnx` | Trained model weights (ONNX format) / 训练好的模型权重（ONNX 格式）|
| `xiangqi_bot.spec` | PyInstaller build spec for macOS app / macOS 应用的 PyInstaller 构建规格 |
| `hook-cv2.py` | PyInstaller runtime hook for OpenCV / OpenCV 的 PyInstaller 运行时 hook |
| `pikafish` | Pikafish engine binary (macOS x86_64) / Pikafish 引擎二进制（macOS x86_64）|
| `pikafish.nnue` | Neural network evaluation file for Pikafish / Pikafish 的神经网络评估文件 |
| `cnn_data/` | Training data by piece type (`red_R/`, `black_r/`, `empty/`, etc.) / 按棋子类型组织的训练数据 |
| `debug/` | Debug patches from latest game session / 最近对局的 debug 补丁 |
| `calib.json` | Grid calibration data / 棋盘校准数据 |
| `templates/` | Popup button templates used by `continuous_play.py` / `continuous_play.py` 使用的弹窗按钮模板 |
| `end_snapshots/` | Auto-saved end-of-game screenshots (for template tuning) / 自动保存的结束画面截图（用于调校模板）|
| `continuous_play_debug/` | Failure dumps from `continuous_play.py` (screenshot + state) / `continuous_play.py` 的失败转储（截图 + 状态）|
