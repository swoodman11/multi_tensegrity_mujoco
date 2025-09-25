# Multi-Tensegrity MuJoCo Simulator - AI Coding Guide

## ⚠️ Experimental Research Codebase Disclaimer

This is an **exploratory learning project** for reinforcement learning and tensegrity robotics. The codebase is under active development and may contain:

- **Inconsistent implementations** across files
- **Outdated documentation** that doesn't match current code
- **Experimental approaches** that may not represent best practices
- **Parameter mismatches** between components (e.g., observation dimensions)

**AI Agent Guidelines:**
- ✅ Use this guide as a starting point, but **always verify against actual code**
- ✅ **Flag inconsistencies** between documentation and implementation
- ✅ **Question assumptions** - if something seems off, investigate further
- ✅ **Cross-reference** parameter values across multiple files before using them
- ✅ **Alert the user** to any discrepancies or potential errors you discover

When in doubt, inspect the actual source code rather than relying solely on this documentation.

## Project Architecture

This is a MuJoCo-based physics simulation for tensegrity robots with reinforcement learning capabilities. The codebase has two main execution paths:

### Core Components

- **`TensegrityMuJoCoSimulator`** (`mujoco_physics_engine/tensegrity_mjc_simulation.py`): Main simulator class that wraps MuJoCo for tensegrity-specific operations
- **`TensegrityEnv`** (`tensegrity_env.py`): Gymnasium environment wrapper for RL training with Stable-Baselines3
- **`AbstractMuJoCoSimulator`** (`mujoco_physics_engine/mujoco_simulation.py`): Base MuJoCo interface providing rendering and simulation primitives

### Cable Control System

The project uses a sophisticated cable-driven tensegrity control system:
- **12 actuated cables** per robot (6 per tensegrity structure in dual-robot setup)
- **PID controllers** (`mujoco_physics_engine/pid.py`) for cable length control with normalized target lengths (0.0 to 1.0)
- **DC motor models** (`mujoco_physics_engine/cable_motor.py`) with winch radius 0.035m and max speed 220 RPM
- **Cable naming convention**: `t1_s_X_bY` format where t1/t2 = robot, X/Y = attachment points

## Development Workflows

### Environment Setup
```bash
conda create --name tensegrity_gnn python=3.12
conda activate tensegrity_gnn
pip install -r requirements.txt
```

### Key Commands

- **Run simulation**: `python run.py` (generates visualization frames in `sim_output/`)
- **Train RL model**: `python train.py` (creates PPO model, logs to `ppo_tensegrity_tensorboard/`)
- **Test trained model**: `python test_trained_model.py` (loads saved model for evaluation)
- **Test MuJoCo setup**: `python test_mujoco_simulator.py` (basic MuJoCo viewer test)

## XML Model Configuration

- Models located in `mujoco_physics_engine/xml_models/`
- Two configurations: `two_3bar_new_platform_config_1.xml` and `config_2.xml`
- Each represents dual 3-bar tensegrity robots connected by platform

## RL Integration Patterns

### Observation Space
- **Default dimension**: 78 (configurable via `obs_dim` parameter in TensegrityMuJoCoSimulator)
- **CRITICAL**: Verify obs_dim consistency between TensegrityEnv (tensegrity_env.py) and TensegrityMuJoCoSimulator 
- **Type**: Box space with infinite bounds
- Includes robot state vectors from MuJoCo simulation
- **Debug tip**: Print `env.observation_space.shape` and `sim.obs_dim` to verify alignment

### Action Space  
- **12 actuators** with continuous control [-1.0, 1.0]
- Actions map to normalized target cable lengths via PID control
- Example target patterns: `[1.0]*12` (all extended), `[0]*6 + [1]*6` (half contracted)

### Reward Function
Implemented in `sim_step()` method of `TensegrityMuJoCoSimulator` - examine this for task-specific objectives.

## Code Patterns

### Simulator Initialization
```python
# Always specify xml_path as Path object
xml = Path('mujoco_physics_engine/xml_models/two_3bar_new_platform_config_1.xml')
# Default obs_dim=78, verify this matches your environment setup
sim = TensegrityMuJoCoSimulator(xml, visualize=True, obs_dim=78)
```

### Cable Control
```python
# Use normalized lengths 0.0-1.0, not absolute measurements
target_lengths = [0.5 for _ in range(sim.n_actuators)]  # 50% extension
frames = sim.run_target_lengths(target_lengths, vis_save_dir=output_dir)
```

### Environment Usage
```python
# Always pass visualize parameter explicitly
env = TensegrityEnv(visualize=True)  # For testing
env = TensegrityEnv(visualize=False)  # For training
```

## Debugging & Testing

- Use `test_mujoco_simulator.py` to verify MuJoCo installation and model loading
- `bring_to_grnd()` method automatically positions robots above ground plane
- Visualization outputs saved as sequential PNG frames in `sim_output/`
- TensorBoard logs available at `./ppo_tensegrity_tensorboard/` for training analysis

### Common Issues & Validation

**Observation Space Mismatch**: If getting shape errors during training:
```python
# Verify dimensions match between environment and simulator
env = TensegrityEnv()
print(f"Env obs space: {env.observation_space.shape}")
print(f"Sim obs_dim: {env.sim.obs_dim}")
```

**Cable Control Issues**: Verify actuator count matches expected 12 cables:
```python
print(f"Number of actuators: {sim.n_actuators}")
print(f"Cable sites defined: {len(sim.cable_sites)}")
```

### Verification Workflow for AI Agents

Before making changes based on this documentation:

1. **Cross-check parameters** across files (e.g., `obs_dim` in simulator vs environment)
2. **Validate examples** by running code snippets to ensure they work as documented
3. **Compare defaults** between different classes that should be consistent
4. **Test integration points** where multiple components interact
5. **Report discrepancies** to the user with specific file locations and line numbers

Example validation pattern:
```python
# Always verify documented values against actual implementation
sim = TensegrityMuJoCoSimulator(xml_path)
env = TensegrityEnv()
if sim.obs_dim != env.observation_space.shape[0]:
    print(f"MISMATCH FOUND: Simulator obs_dim={sim.obs_dim} but Environment expects {env.observation_space.shape[0]}")
```

## Key Dependencies

- **MuJoCo 3.3.6**: Physics simulation engine
- **Stable-Baselines3**: PPO algorithm implementation  
- **OpenCV**: Video/image processing for visualization
- **Gymnasium**: RL environment standard interface