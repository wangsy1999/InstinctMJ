# Shadowing Task

## InstinctLab 原版地址

- 项目根 README: `/home/duanxin/Xyk/Project-Instinct/InstinctLab/README.md`
- Shadowing 原版 README: `/home/duanxin/Xyk/Project-Instinct/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/README.md`
- Perceptive Shadowing 原版配置: `/home/duanxin/Xyk/Project-Instinct/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py`

## Basic Usage Guidelines

### BeyondMimic Shadowing

**Task ID:** `Instinct-BeyondMimic-Plane-G1-v0`

This is an exact replication of the BeyondMimic training configuration.

1. Go to `beyondmimic/config/g1/beyondmimic_plane_cfg.py` and set the `MOTION_NAME`, `_hacked_selected_file_`, `AmassMotionCfg.path` to the motion you want to use.

    - `MOTION_NAME`: A identifier for you to remember which motion you are using.
    - `AmassMotionCfg.path`: The folder path to where you store the motion files.
    - `_hacked_selected_file_`: The filename of the motion you want to use, relative to the `AmassMotionCfg.path` folder.

2. Train the policy:
```bash
python scripts/instinct_rl/train.py --headless --task=Instinct-BeyondMimic-Plane-G1-v0
```

3. Play trained policy (load_run must be provided, absolute path is recommended, or use `--no_resume` to visualize untrained policy):
```bash
python scripts/instinct_rl/play.py --task=Instinct-BeyondMimic-Plane-G1-v0 --load_run=<run_name>
```

### Whole Body Shadowing

**Task ID:** `Instinct-Shadowing-WholeBody-Plane-G1-v0`

1. Go to `whole_body/config/g1/plane_shadowing_cfg.py` and set the `MOTION_NAME`, `_path_`, `_hacked_selected_files_` to the motion you want to use.

    - `MOTION_NAME`: A identifier for you to remember which motion you are using.
    - `_path_`: The folder path to where you store the motion files.
    - `_hacked_selected_files_`: The filenames of the motion you want to use, relative to the `_path_` folder.

2. Train the policy:
```bash
python scripts/instinct_rl/train.py --headless --task=Instinct-Shadowing-WholeBody-Plane-G1-v0
```

3. Play trained policy (load_run must be provided, absolute path is recommended, or use `--no_resume` to visualize untrained policy):
```bash
python scripts/instinct_rl/play.py --task=Instinct-Shadowing-WholeBody-Plane-G1-v0 --load_run=<run_name>
```

### Perceptive Shadowing

**Task IDs:**
- `Instinct-Perceptive-Shadowing-G1-v0` (Deep Whole-body Parkour)

1. Configure perceptive environment variables before training or play.

    - `INSTINCT_PERCEPTIVE_MOTION_FOLDER`: motion dataset folder. The task reads `metadata.yaml` under this folder.
    - `INSTINCT_PERCEPTIVE_FORCE_PLANE`: force training terrain to plane (`true`/`false`, default: `false`).
    - `INSTINCT_PERCEPTIVE_PLAY_MOTION_PATH`: optional play-only motion folder override.
    - `INSTINCT_PERCEPTIVE_PLAY_STUB_SAMPLING_STRATEGY`: play stub sampling strategy (default: `independent`).
    - `INSTINCT_PERCEPTIVE_PLAY_MOTION_BIN_LENGTH_S`: play motion bin length (`auto`/`none`/float, default: `auto`).
    - `INSTINCT_PERCEPTIVE_PLAY_FORCE_PLANE`: force play terrain to plane (`true`/`false`, default: `false`).

2. Train the policy:
```bash
# PPO version
python scripts/instinct_rl/train.py --headless --task=Instinct-Perceptive-Shadowing-G1-v0
```

3. Play trained policy (load_run must be provided, absolute path is recommended, or use `--no_resume` to visualize untrained policy):
```bash
# PPO version
python -m instinct_mjlab.scripts.instinct_rl.play Instinct-Perceptive-Shadowing-G1-v0 --load-run=<run_name>
```

## Common Options

- `--num_envs`: Number of parallel environments (default varies by task)
- `--max_iterations`: Training iterations (default varies by task)
- `--load-run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
