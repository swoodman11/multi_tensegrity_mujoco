# Multi-Tensegrity MuJoCo Simulator

Lightweight research code for tensegrity simulation and reinforcement learning.

## Environment Setup

```bash
conda create --name tensegrity_gnn python=3.12
conda activate tensegrity_gnn
pip install -r requirements.txt
```

## Dual-Tensegrity (Existing) Demo

```bash
python run.py
```

## New Single Tensegrity Runner (`run_single.py`)

This script runs a single 3‑bar tensegrity defined in `mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml` with **6 active cables** (stiffness=1000) and **3 passive cables** (stiffness=20000).

### Quick Start

Write an example action JSON and run it:

```bash
python run_single.py --write-example actions_example.json
python run_single.py --sequence actions_example.json
```

Headless with video + plots:

```bash
python run_single.py --sequence actions_example.json --no-vis --video-save --plots
```

### JSON Action File Format

Each action = 6 normalized target cable lengths in `[0,1]` applied for **one MuJoCo timestep** (from the XML `timestep`). Total simulated time = `(num_actions * timestep)`.

```json
{
	"actions": [
		[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
		[0.2, 0.8, 0.2, 0.8, 0.2, 0.8],
		[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
	]
}
```

You can also pass `--total-steps N` to truncate or repeat the last action.

### CLI Flags

| Flag | Description |
|------|-------------|
| `--sequence FILE` | Path to JSON file (required unless writing example). |
| `--write-example FILE` | Generate an example JSON and exit. |
| `--no-vis` | Run without on-screen viewer (headless). |
| `--video-save` | Save MP4 to `sim_output/single/single_run.mp4`. |
| `--kp/--ki/--kd` | PID gains (defaults 2.0 / 0.0 / 1.0). |
| `--total-steps N` | Override number of steps (pad or truncate). |
| `--plots` | Save action, observation, and reward plots. |

### Output Artifacts

Saved to `sim_output/single/`:

* `single_run.mp4` (if `--video-save`)
* `actions.png` (if `--plots`)
* `observations.png` (if `--plots`)
* `rewards.png` (if `--plots`)

### Reward & Observation Modularity

The single simulator (`single_tensegrity_mjc_simulation.py`):

* Observations currently include: normalized cable lengths, length rates, previous action, IMU gravity vectors.
* Reward starts with an "activity" term (mean absolute length change) and is easy to extend (edit `compute_reward`).
* Add/remove observation components by editing `ObservationConfig` or the `get_observation` assembly.

### Action Range Handling

Actions are clipped to `[0,1]`. The **first** out-of-range occurrence raises a `ValueError`. Subsequent violations are counted and reported at the end.

### Extending for RL

`SingleTensegrityMuJoCoSimulator` exposes:

```python
reset() -> obs
step(action) -> (obs, reward, done, info)
```

You can wrap it in a Gymnasium environment easily by forwarding those calls and defining action/observation spaces.

### Adding New Reward Terms

Edit `compute_reward()` and call `terms.add("name", value)`. Plots will automatically pick up new terms when using `--plots`.

---

## Troubleshooting

* Missing video file: ensure `--video-save` and MuJoCo install is valid.
* Action dimension errors: verify each JSON row has exactly 6 floats.
* No plots: add `--plots` flag.

---

## License / Notice
Experimental research code—APIs may change without notice.
