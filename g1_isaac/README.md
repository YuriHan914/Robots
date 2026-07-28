# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/g1_isaac

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Training the G1 imitation-learning (AMP) policy on a retargeted motion clip:

        `scripts/skrl/train.py` trains a policy with [skrl](https://skrl.readthedocs.io). By default
        it runs the G1 AMP task (`Template-G1-Isaac-AMP-Dance-Direct-v0`), which uses skrl's
        Adversarial Motion Priors agent (`agents/skrl_amp_cfg.yaml`) to imitate the retargeted mocap
        clip `data/dance1_retarget_g1.csv` (the same clip played back by `scripts/play_motion.py`,
        loaded via `g1_isaac/tasks/direct/g1_isaac/motions/motion_loader.py`): a discriminator learns
        to tell policy rollouts apart from the reference motion, and the resulting style reward (not a
        hand-designed task reward) drives training.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/skrl/train.py

        # equivalent, spelled out explicitly
        python scripts/skrl/train.py --task Template-G1-Isaac-AMP-Dance-Direct-v0 --algorithm AMP

        # headless, fewer parallel envs, capped iterations (e.g. for a quick smoke test)
        python scripts/skrl/train.py --headless --num_envs 32 --max_iterations 10
        
        # train resume
        python scripts/skrl/train.py --headless --num_envs 32 --max_iterations 1000 --checkpoint best_agent.pt
        ```

        - `--task`: gym task id to train (defaults to `Template-G1-Isaac-AMP-Dance-Direct-v0`); see
          `python scripts/list_envs.py` for all registered tasks.
        - `--algorithm`: RL algorithm to use when a task registers more than one skrl agent config
          (defaults to `AMP`); selects the `skrl_<algorithm>_cfg_entry_point` agent config, e.g.
          `agents/skrl_amp_cfg.yaml` for `AMP`.
        - `--agent`: explicit agent config entry point name, overriding the one derived from
          `--algorithm` (defaults to `None`).
        - `--num_envs`: number of parallel environments to simulate (defaults to the task's config,
          `4096` for the G1 AMP task).
        - `--seed`: random seed for the environment and agent; pass `-1` to sample one randomly
          (defaults to the agent config's seed, `42`).
        - `--max_iterations`: number of policy training iterations; overrides the agent config's
          `trainer.timesteps` (defaults to `None`, i.e. use the agent config as-is).
        - `--checkpoint`: path to a model checkpoint to resume training from (defaults to `None`).
        - `--ml_framework`: `torch` or `jax` backend for skrl (defaults to `torch`).
        - `--distributed`: run multi-GPU/multi-node training (requires a GPU device; incompatible
          with `--device cpu`).
        - `--video`, `--video_length`, `--video_interval`: record training videos (off by default);
          length/interval are in simulation steps (`200`/`2000` by default).
        - `--export_io_descriptors`: export IO descriptors (manager-based environments only; not
          applicable to this direct-workflow task).
        - `--headless`: run without the Isaac Sim UI (from Isaac Lab's `AppLauncher`; recommended for
          long training runs).
        - `--device`: simulation device, e.g. `cuda:0` or `cpu` (from `AppLauncher`; defaults to the
          task's config).

    - Playing back a trained G1 AMP policy:

        `scripts/skrl/play.py` loads a checkpoint produced by `scripts/skrl/train.py` above and runs it
        in inference mode (deterministic actions, no exploration/learning). By default it plays the
        same G1 AMP task (`Template-G1-Isaac-AMP-Dance-Direct-v0`) and automatically picks the latest
        checkpoint under `logs/skrl/g1_amp_dance/`.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/skrl/play.py --num_envs 32

        # play a specific checkpoint instead of the latest one
        python scripts/skrl/play.py --num_envs 32 --checkpoint logs/skrl/g1_amp_dance/<timestamp>_amp_torch/checkpoints/best_agent.pt

        # record a video of the playback instead of opening the viewer
        python scripts/skrl/play.py --num_envs 32 --headless --video --video_length 200
        ```

        - `--task`: gym task id to play (defaults to `Template-G1-Isaac-AMP-Dance-Direct-v0`).
        - `--algorithm`: RL algorithm/agent config used to build the agent, must match how the
          checkpoint was trained (defaults to `AMP`).
        - `--agent`: explicit agent config entry point name, overriding the one derived from
          `--algorithm` (defaults to `None`).
        - `--checkpoint`: path to a specific model checkpoint to load (defaults to `None`, i.e.
          auto-discover the latest checkpoint under `logs/skrl/<experiment.directory>/`).
        - `--use_pretrained_checkpoint`: load a published pre-trained checkpoint from Nucleus instead
          of a local one (not applicable to this task, which has no published checkpoint).
        - `--num_envs`: number of parallel environments to simulate (defaults to the task's config,
          `4096` for the G1 AMP task - pass a small number for viewing).
        - `--seed`: random seed for the environment and agent (defaults to the agent config's seed).
        - `--real-time`: throttle stepping to match the environment's real-time rate instead of
          running as fast as possible.
        - `--ml_framework`: `torch` or `jax` backend for skrl (defaults to `torch`).
        - `--video`, `--video_length`: record a playback video instead of/alongside the live viewer
          (`--video_interval` doesn't apply here; one video of `--video_length` steps is recorded).
        - `--disable_fabric`: use USD I/O instead of fabric for reading/writing simulation data.
        - `--headless`: run without the Isaac Sim UI (from Isaac Lab's `AppLauncher`).
        - `--device`: simulation device, e.g. `cuda:0` or `cpu` (from `AppLauncher`; defaults to the
          task's config).

    - Training the G1 ASAP-style motion-tracking policy:

        `scripts/asap/train.py` trains a policy with a self-contained PPO implementation
        (`scripts/asap/algo`, ported from ASAP - https://github.com/LeCAR-Lab/ASAP - not skrl).
        By default it runs `Template-G1-Isaac-ASAP-Motion-Direct-v0`
        (`g1_isaac/tasks/direct/g1_isaac/g1_isaac_asap_env.py`), which uses hand-crafted DeepMimic-style
        tracking rewards (body/feet/keypoint position, rotation, velocity, joint position/velocity, see
        `docs/asap_motion_tracking.txt` for the full breakdown) computed directly against the reference
        motion, instead of the AMP task's discriminator-driven style reward. Configuration lives in
        `agents/asap_motion_cfg.yaml` (network/PPO hyperparameters) and `G1AsapEnvCfg` in
        `g1_isaac_env_cfg.py` (reward weights, curriculum, termination, motion file).

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/asap/train.py

        # headless, fewer parallel envs, capped iterations (e.g. for a quick smoke test)
        python scripts/asap/train.py --headless --num_envs 4 --max_iterations 5

        # full training run (adjust --num_envs to fit your GPU memory - each env spawns two
        # articulations, the policy robot and a reference "ghost" robot used for FK, roughly doubling
        # per-env cost vs. the AMP task)
        python scripts/asap/train.py --headless --num_envs 2048 --max_iterations 30000

        # resume from a checkpoint
        python scripts/asap/train.py --headless --num_envs 2048 --max_iterations 30000 \
            --checkpoint logs/asap/g1_asap_motion/<timestamp>/checkpoints/model_100.pt

        # record training videos too
        python scripts/asap/train.py --headless --num_envs 512 --video --video_length 200 --video_interval 2000
        ```

        - `--task`: gym task id to train (defaults to `Template-G1-Isaac-ASAP-Motion-Direct-v0`).
        - `--num_envs`: number of parallel environments to simulate (defaults to the task's config,
          `2048`).
        - `--seed`: random seed for the environment and PPO; pass `-1` to sample one randomly
          (defaults to the agent config's seed, `42`).
        - `--max_iterations`: number of PPO training iterations; overrides
          `agents/asap_motion_cfg.yaml`'s `runner.num_iterations` (defaults to `None`, i.e. use the
          agent config as-is).
        - `--checkpoint`: path to a model checkpoint (`.pt`) to resume training from (defaults to `None`).
        - `--video`, `--video_length`, `--video_interval`: record training videos (off by default);
          length/interval are in simulation steps (`200`/`2000` by default).
        - `--headless`: run without the Isaac Sim UI (from Isaac Lab's `AppLauncher`; recommended for
          long training runs).
        - `--device`: simulation device, e.g. `cuda:0` or `cpu` (from `AppLauncher`; defaults to the
          task's config).

        Logs/checkpoints/TensorBoard event files are written to
        `logs/asap/<experiment_name>/<timestamp>/` (config snapshots under `params/`, checkpoints under
        `checkpoints/model_<iteration>.pt`). View training curves with:

        ```bash
        tensorboard --logdir logs/asap
        ```

    - Playing back a trained G1 ASAP-style motion-tracking policy:

        `scripts/asap/play.py` loads a checkpoint produced by `scripts/asap/train.py` above and runs it
        deterministically (the Gaussian policy's mean action, no exploration). By default it plays the
        same task (`Template-G1-Isaac-ASAP-Motion-Direct-v0`) and automatically picks the latest
        checkpoint under `logs/asap/g1_asap_motion/`. It also switches to deterministic (round-robin)
        clip assignment and always starts playback from the beginning of the clip, and re-enables
        self-collisions (same conventions as `scripts/skrl/play.py`).

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/asap/play.py --num_envs 8 --real-time

        # play a specific checkpoint instead of the latest one
        python scripts/asap/play.py --num_envs 8 --checkpoint logs/asap/g1_asap_motion/<timestamp>/checkpoints/model_30000.pt

        # record a video of the playback instead of opening the viewer
        python scripts/asap/play.py --num_envs 8 --headless --video --video_length 300
        ```

        - `--task`: gym task id to play (defaults to `Template-G1-Isaac-ASAP-Motion-Direct-v0`).
        - `--checkpoint`: path to a specific model checkpoint to load (defaults to `None`, i.e.
          auto-discover the latest checkpoint under `logs/asap/<experiment_name>/`).
        - `--num_envs`: number of parallel environments to simulate (pass a small number for viewing).
        - `--seed`: random seed for the environment.
        - `--real-time`: throttle stepping to match the environment's real-time rate instead of
          running as fast as possible.
        - `--video`, `--video_length`: record a playback video instead of/alongside the live viewer.
        - `--headless`: run without the Isaac Sim UI (from Isaac Lab's `AppLauncher`).
        - `--device`: simulation device, e.g. `cuda:0` or `cpu` (from `AppLauncher`; defaults to the
          task's config).

        See `docs/asap_motion_tracking.txt` (Korean) for a full explanation of the algorithm
        (PPO implementation) and policy/reward/observation design this task uses.

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

    - Playing back a retargeted motion clip on the G1 robot:

        This replays a retargeted motion capture CSV (root pose + 29 joint angles) on the Unitree
        G1 robot in the simulator. By default it is played back kinematically (the exact recorded
        pose is written into the sim every frame); pass `--dynamic` to instead let physics
        (gravity, contacts, PD actuators) actually drive the robot from the clip's starting pose,
        which may cause it to fall over if the motion isn't dynamically feasible for G1.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/play_motion.py --motion_file data/dance1_retarget_g1.csv --loop

        # physics-driven playback instead of exact kinematic replay
        python scripts/play_motion.py --motion_file data/dance1_retarget_g1.csv --dynamic
        ```

        - `--motion_file`: path to the retargeted motion CSV (defaults to `data/dance1_retarget_g1.csv`).
        - `--fps`: playback frame rate of the motion clip (defaults to `30`, the source capture rate).
        - `--loop`: repeat the clip instead of holding the last frame once it finishes.
        - `--dynamic`: drive the robot with PD position targets and let physics determine the outcome,
          instead of teleporting the exact recorded pose every frame.
        - `--physics_hz`: physics simulation rate in Hz, only used in `--dynamic` mode (defaults to `200`).

    - Adding full-body collision to the G1 robot for `--dynamic` playback:

        `scripts/play_motion.py --dynamic`'s G1 asset (`assets/g1_full_collision.usd`) only has
        collision geometry generated for it once - the upstream newton-assets G1 model it's derived
        from has real collision meshes on just 8 of 44 links (feet, ankles, knees, pelvis, waist,
        torso), so other links (hips, shoulders, elbows, wrists, fingers) would otherwise pass
        straight through the ground/other bodies with no contact response in `--dynamic` mode.

        `scripts/tools/add_g1_full_body_collision.py` fixes this: it opens the upstream G1 USD,
        adds a box-collider proxy (sized from each link's visual mesh bounding box) to every rigid
        body link that has no collision geometry yet, and writes the result to
        `assets/g1_full_collision.usd`. It is plain USD authoring - no Isaac Sim/Kit app needs to be
        running, just a Python with `pxr` (usd-core) importable (this project's own `isaac` conda
        env already has it). The output file is not committed to git (excluded by `.gitignore`), so
        re-run this script whenever `assets/g1_full_collision.usd` is missing or the upstream
        newton-assets G1 model is updated.

        ```bash
        python scripts/tools/add_g1_full_body_collision.py
        ```

        This script takes no arguments (source/output paths are constants at the top of the file).
        Example output:

        ```text
        already had collision (8): ['pelvis', 'left_knee_link', 'left_ankle_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link']
        added box collider (36): ['left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_ankle_pitch_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_ankle_pitch_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'left_hand_index_0_link', 'left_hand_index_1_link', 'left_hand_middle_0_link', 'left_hand_middle_1_link', 'left_hand_thumb_0_link', 'left_hand_thumb_1_link', 'left_hand_thumb_2_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_hand_index_0_link', 'right_hand_index_1_link', 'right_hand_middle_0_link', 'right_hand_middle_1_link', 'right_hand_thumb_0_link', 'right_hand_thumb_1_link', 'right_hand_thumb_2_link']
        skipped, no visual geometry found (7): ['right_ankle_roll_front_link', 'right_ankle_roll_mid_link', 'right_ankle_roll_back_link', 'left_ankle_roll_front_link', 'left_ankle_roll_mid_link', 'left_ankle_roll_back_link', 'imu_link']

        wrote /mnt/data/github/Robots/g1_isaac/assets/g1_full_collision.usd
        ```

        The 7 skipped links have no visual mesh (an IMU sensor mount and the feet's 3-point contact
        sub-links), so there is nothing to derive a collider from - this is expected.

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/g1_isaac/g1_isaac/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/g1_isaac"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```



