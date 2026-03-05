# Instinct_mjlab

`Instinct_mjlab` 是基于 `mjlab` 的 InstinctLab 任务迁移项目，使用 `instinct_rl` 训练与回放策略。

## InstinctLab 原版地址

- 项目根 README: `/home/duanxin/Xyk/Project-Instinct/InstinctLab/README.md`
- 任务源码根目录: `/home/duanxin/Xyk/Project-Instinct/InstinctLab/source/instinctlab/instinctlab/tasks`

## 主要内容

- 任务族：Locomotion / Shadowing / Parkour（G1）
- 统一任务注册：`register_instinct_task(...)`
- 训练与回放入口：
  - `instinct-train`
  - `instinct-play`
  - `instinct-list-envs`

## 环境要求

- Python `>=3.10,<3.14`
- Linux + NVIDIA GPU（当前训练流程要求 CUDA）
- 与本仓库同级目录下可用的 `mjlab` 与 `instinct_rl`

## 安装

在 `Instinct_mjlab` 目录下执行：

```bash
uv sync
```

如果你使用 `pip`：

```bash
pip install -e ../mjlab
pip install -e ../instinct_rl
pip install -e .
```

## 列出可用任务

```bash
instinct-list-envs
instinct-list-envs shadowing
```

当前已注册任务 ID：

- `Instinct-Locomotion-Flat-G1-v0`
- `Instinct-Locomotion-Flat-G1-Play-v0`
- `Instinct-BeyondMimic-Plane-G1-v0`
- `Instinct-BeyondMimic-Plane-G1-Play-v0`
- `Instinct-Shadowing-WholeBody-Plane-G1-v0`
- `Instinct-Shadowing-WholeBody-Plane-G1-Play-v0`
- `Instinct-Perceptive-Shadowing-G1-v0`
- `Instinct-Perceptive-Shadowing-G1-Play-v0`
- `Instinct-Perceptive-Vae-G1-v0`
- `Instinct-Perceptive-Vae-G1-Play-v0`
- `Instinct-Parkour-Target-Amp-G1-v0`
- `Instinct-Parkour-Target-Amp-G1-Play-v0`

## 训练与回放

训练示例：

```bash
instinct-train Instinct-Locomotion-Flat-G1-v0
instinct-train Instinct-Perceptive-Shadowing-G1-v0
```

回放示例（`--load-run` 必填，建议给绝对路径）：

```bash
instinct-play Instinct-Locomotion-Flat-G1-Play-v0 --load-run <run_name>
instinct-play Instinct-Perceptive-Shadowing-G1-Play-v0 --load-run <run_name>
```

导出 ONNX（Parkour）：

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --export-onnx
```

如果环境里没有安装 console scripts，也可以用模块方式：

```bash
python -m instinct_mjlab.scripts.instinct_rl.train Instinct-Locomotion-Flat-G1-v0
python -m instinct_mjlab.scripts.instinct_rl.play Instinct-Locomotion-Flat-G1-Play-v0 --load-run <run_name>
python -m instinct_mjlab.scripts.list_envs
```

## 数据路径

- 通用数据根目录可通过 `INSTINCT_DATASETS_ROOT` 指定。
- 各任务的 motion/terrain 配置以对应任务配置文件或子 README 为准。

## 日志与输出

- 训练日志默认写入：`logs/instinct_rl/<experiment_name>/<timestamp_run>/`
- Play 视频默认写入：`.../videos/play/`

## 子模块文档

- Shadowing: `src/instinct_mjlab/tasks/shadowing/README.md`
- Parkour: `src/instinct_mjlab/tasks/parkour/README.md`
