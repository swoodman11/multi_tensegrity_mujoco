# Single Tensegrity Training Pipeline

Complete training and testing pipeline for single 3-bar tensegrity robot with 6 actuated cables.

## 🎯 Overview

This pipeline provides a complete workflow for training reinforcement learning agents on a single tensegrity robot using SAC (Soft Actor-Critic) with GPU acceleration and behavioral cloning pretraining from gait patterns.

### Key Differences from Dual Robot Pipeline
- **6 actuators** (vs 12 for dual robot)
- **27-dimensional observations** (vs 96 for dual robot)
- **Single XML model**: `3bar_new_platform_all_cables.xml`
- **Flexible observation dimensions** via `obs_dim` parameter
- **JSON-based gait patterns** for easy experimentation

---

## 📁 Files Created

### Core Simulator and Environment
1. **`mujoco_physics_engine/single_tensegrity_mjc_simulation.py`**
   - Single tensegrity MuJoCo simulator
   - 6 actuated cables (td_0 to td_5)
   - Hardcoded cable site pairs
   - Reward function copied from dual robot (adapted for dimensions)
   - ⚠️ **NEEDS UPDATE**: Verify `actuated_ids = [0, 1, 2, 3, 4, 5]` are correct

2. **`single_tensegrity_env.py`**
   - Gymnasium environment wrapper
   - Action space: `Box(0, 1, (6,))` - normalized cable lengths
   - Observation space: `Box(-inf, inf, (obs_dim,))` - default 27D

### Gait Pattern Files
3. **`dual_robot_first_6.json`**
   - First 6 actuators extracted from dual robot roll sequence
   - 31 steps, proven gait pattern

4. **`optimal_rolling_gait.json`**
   - Custom-designed rolling gait for single robot
   - 20 steps, alternating tripod lift pattern

5. **`single_tensegrity_gait_generation.py`**
   - Utility to generate various gait patterns
   - Patterns: single cable test, alternating tripod, wave, sequential, pulsing, random walk
   - Saves gaits to JSON files

### Training and Testing
6. **`gpu_pretraining_SAC_single.py`**
   - Main GPU training script
   - Loads gaits from JSON files
   - Behavioral cloning + RL fine-tuning
   - Auto-detects GPU and selects optimal config

7. **`test_trained_model_SAC_single.py`**
   - Test trained models with visualization
   - Statistics and reward component analysis

8. **`run_single.py`**
   - Simple demonstration script
   - Can load gait JSON or use built-in test pattern

---

## 🚀 Quick Start

### 1. Generate Gait Patterns (Optional)
```bash
python single_tensegrity_gait_generation.py
```
This creates various test gait JSON files.

### 2. Test Simulator
```bash
# Run with built-in test pattern
python run_single.py

# Run with specific gait file
python run_single.py dual_robot_first_6.json 5

# Run with optimal rolling gait
python run_single.py optimal_rolling_gait.json 3
```

### 3. Train Model
```bash
python gpu_pretraining_SAC_single.py
```
This will:
- Auto-detect your GPU
- Select optimal configuration
- Load gait from `dual_robot_first_6.json` (default)
- Run behavioral cloning pretraining
- Fine-tune with SAC
- Save model to `models/gpu/sac_single_gpu_pretraining_*`

### 4. Test Trained Model
```bash
# Test latest model
python test_trained_model_SAC_single.py

# Test specific model
python test_trained_model_SAC_single.py --model models/gpu/sac_single_gpu_pretraining_rtx4090_large_20251020_123456

# Run 10 episodes without visualization
python test_trained_model_SAC_single.py --episodes 10 --no-viz
```

---

## ⚙️ Configuration

### Changing Gait Pattern

Edit `gpu_pretraining_SAC_single.py` line ~560:
```python
gait_json = "dual_robot_first_6.json"  # Change this
# Options: "optimal_rolling_gait.json", "alternating_tripod.json", etc.
```

### Adjusting Observation Dimension

By default, observations are 27D:
- 6 cable lengths (normalized)
- 6 cable length rates
- 6 previous actions
- 9 IMU values (3 rods × 3D gravity vectors)

To modify, edit `single_tensegrity_mjc_simulation.py` line ~85:
```python
self.obs_components = {
    'cable_lengths': 6,
    'cable_rates': 6,
    'prev_action': 6,
    'imu': 9,  # Can change to 0 to remove IMU
}
```

### Controller Parameters

Default PID values (matching dual robot):
- `controller_kp=10.0`
- `controller_ki=0.2`
- `controller_kd=2.0`

Modify in `single_tensegrity_env.py` line ~30 or when creating simulator.

---

## 🔍 Important Notes

### ⚠️ Verification Needed

1. **Actuated Cable IDs**: In `single_tensegrity_mjc_simulation.py` line ~63:
   ```python
   # NEEDS UPDATE: Verify these are correct
   self.actuated_ids = [0, 1, 2, 3, 4, 5]
   ```
   Verify these correspond to tendons td_0 through td_5 in the XML.

2. **Cable Site Pairs**: Hardcoded in lines ~67-80. Verify against XML:
   ```xml
   <!-- From 3bar_new_platform_all_cables.xml -->
   <tendon>
     <spatial name="td_0" ...>
       <site site="s_3_b5"/>
       <site site="s_b5_3"/>
     </spatial>
     ...
   ```

### Observation Space Flexibility

The observation dimension is flexible. If you modify `obs_components`, the `obs_dim` will automatically adjust unless explicitly set:

```python
# Option 1: Auto-calculate (recommended)
sim = SingleTensegrityMuJoCoSimulator(xml_path, obs_dim=None)

# Option 2: Explicit dimension
sim = SingleTensegrityMuJoCoSimulator(xml_path, obs_dim=27)
```

### Reward Function

The reward function is copied exactly from the dual robot with dimension adjustments:
- Contact count band: [1, 4] instead of [2, 6]
- Hover cap: 3 instead of 4
- All other terms and weights preserved

Located in `single_tensegrity_mjc_simulation.py` `sim_step()` method (lines ~200-450).

---

## 📊 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./sac_single_tensegrity_tensorboard_rtx4090_large/
```

### Files Created During Training
- `models/gpu/sac_single_gpu_pretraining_*` - Trained models
- `logs/best_model_single_*` - Best model checkpoints
- `logs/evals_single_*` - Evaluation logs
- `sac_single_tensegrity_tensorboard_*` - TensorBoard logs

---

## 🎮 Custom Gait Patterns

Create your own gait JSON files:

```json
{
  "description": "My custom gait",
  "robot_type": "single_3bar_tensegrity",
  "num_actuators": 6,
  "sequence_length": 10,
  "actions": [
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [0.8, 0.2, 0.8, 0.2, 0.8, 0.2],
    ...
  ]
}
```

Each action is an array of 6 values in [0, 1] representing normalized cable target lengths.

---

## 🐛 Troubleshooting

### Issue: Observation dimension mismatch
**Solution**: Check that `obs_dim` in environment matches simulator output:
```python
env = SingleTensegrityEnv()
print(f"Env obs space: {env.observation_space.shape}")
print(f"Sim obs_dim: {env.sim.obs_dim}")
```

### Issue: GPU Out of Memory
**Solution**: Reduce batch size in `gpu_optimized_configs()`:
```python
"batch_size": 512,  # Reduce from 2048
```

### Issue: Robot not moving
**Solution**: 
1. Verify actuator IDs are correct
2. Check cable site pairs match XML
3. Increase PID Kp value for more responsive control

---

## 📝 Next Steps

1. **Verify actuated_ids** match your XML configuration
2. **Test simulator** with `run_single.py`
3. **Generate gaits** with `single_tensegrity_gait_generation.py`
4. **Train model** with `gpu_pretraining_SAC_single.py`
5. **Evaluate** with `test_trained_model_SAC_single.py`
6. **Iterate** on gait patterns and reward weights

---

## 📚 Related Files

- Original dual robot: `gpu_pretraining_SAC.py`, `tensegrity_env.py`
- XML model: `mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml`
- Base classes: `mujoco_physics_engine/mujoco_simulation.py`
- PID/Motor: `mujoco_physics_engine/pid.py`, `mujoco_physics_engine/cable_motor.py`
