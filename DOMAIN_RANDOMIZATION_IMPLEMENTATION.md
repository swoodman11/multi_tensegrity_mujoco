# Domain Randomization Implementation Summary

## Overview
This document summarizes the implementation of domain randomization with directional locomotion rewards for the tensegrity robot reinforcement learning system.

## Key Changes

### 1. Observation Space Update
- **Changed from 96D to 98D** for Tier-2 observations
- **Added 2D goal direction vector** at the end of observation
- Goal direction is a unit vector representing the desired movement direction

### 2. Domain Randomization (in `gpu_pretraining_SAC.py`)

#### Random Yaw Rotation (0-360°)
```python
yaw_angle = np.random.uniform(0, 2 * np.pi)
goal_direction = np.array([np.cos(yaw_angle), np.sin(yaw_angle)])
```
- Applied at each environment reset
- Goal direction changes with each episode
- Forces policy to be rotation-invariant

#### Friction Randomization (±20%)
```python
friction_multiplier = np.random.uniform(0.8, 1.2)
```
- Applied to all geom friction coefficients
- Makes policy robust to surface variation

#### Action Noise (σ=0.05)
```python
noise = np.random.normal(0, 0.05, size=action.shape)
noisy_action = np.clip(action + noise, 0.0, 1.0)
```
- Gaussian noise added to actions during pretraining
- Improves policy robustness and exploration

### 3. Reward Function Changes (in `tensegrity_mjc_simulation.py`)

#### Removed Fixed Goal Position Reward
- Old system rewarded reaching specific XY coordinates (e.g., [5.0, 0.0])
- New system rewards velocity in arbitrary directions

#### New Directional Reward Components

**Directional Velocity Reward:**
```python
directional_velocity = np.dot(velocity_xy, goal_direction)
directional_velocity_reward = 15.0 * np.tanh(5.0 * directional_velocity)
```
- Rewards movement along goal direction
- Uses tanh for bounded rewards
- Weight: 15.0

**Perpendicular Penalty:**
```python
perpendicular_direction = np.array([-goal_direction[1], goal_direction[0]])
perpendicular_velocity = np.dot(velocity_xy, perpendicular_direction)
perpendicular_penalty = -2.0 * abs(perpendicular_velocity)
```
- Penalizes sideways drift
- Encourages straight-line motion
- Weight: -2.0

**Cumulative Distance Bonus:**
```python
cumulative_distance_bonus = 2.0 * np.tanh(0.2 * cumulative_distance)
```
- Small bonus for net progress in goal direction
- Asymptotic to prevent unbounded growth
- Weight: 2.0

### 4. Simulator Updates (in `tensegrity_mjc_simulation.py`)

#### New Attributes
```python
self.goal_direction = np.array([1.0, 0.0])  # Default to +X direction
self.initial_yaw = 0.0
self.cumulative_distance = 0.0
```

#### Updated `reset()` Method
- Samples random yaw angle (0 to 2π)
- Calculates goal_direction as rotated X-axis
- Applies yaw rotation to robot base
- Randomizes friction coefficients
- Resets tracking variables

#### Helper Function Added
```python
def _euler_to_quaternion(self, roll, pitch, yaw):
    """Convert Euler angles to quaternion [w, x, y, z]"""
```

#### Updated `get_observation_tier2()`
- Now returns 98D vector (was 96D)
- Appends `self.goal_direction` (2D) at the end

### 5. Testing Function (in `test_trained_model_SAC.py`)

#### New `test_directional_control()` Function
- Tests model on 8 cardinal/intercardinal directions:
  - North (0°), Northeast (45°), East (90°), Southeast (135°)
  - South (180°), Southwest (225°), West (270°), Northwest (315°)
- Metrics tracked per direction:
  - Total reward
  - Directional distance traveled
  - Perpendicular deviation
  - Distance per step
  - Directional efficiency

#### Usage
```bash
python test_trained_model_SAC.py --test-directions
python test_trained_model_SAC.py --test-directions --no-vis  # Without visualization
```

## Training Workflow

### 1. Pretraining with Domain Randomization
```bash
python gpu_pretraining_SAC.py
```
- Uses hand-designed gait sequences
- Applies domain randomization each cycle
- Adds action noise to improve robustness
- Creates diverse training experiences

### 2. Pure RL Training (Optional)
```bash
python gpu_train_SAC.py
```
- No behavioral cloning pretraining
- Starts from random policy
- Uses same domain randomization

### 3. Testing Directional Control
```bash
python test_trained_model_SAC.py --test-directions
```
- Evaluates policy on 8 different directions
- Reports performance metrics

## Expected Outcomes

### Policy Characteristics
- **Rotation-invariant**: Works equally well in any direction
- **Robust**: Handles friction variations
- **Generalizable**: Adapts to new directions not seen in training
- **Efficient**: Minimizes perpendicular drift

### Training Benefits
- More diverse training data
- Better generalization
- Faster learning (with pretraining)
- More robust to real-world variations

## Configuration Summary

### Domain Randomization Parameters
- **Yaw rotation**: 0-360° (0-2π radians)
- **Friction variation**: ±20% (0.8-1.2x multiplier)
- **Action noise**: σ=0.05 (Gaussian)

### Observation Dimensions
- **Tier-2 mode**: 98D (was 96D)
- **Legacy mode**: 106D (was 104D)

### Reward Weights
- Directional velocity: 15.0
- Perpendicular penalty: -2.0
- Cumulative distance: 2.0
- (Plus existing locomotion rewards)

## Files Modified

1. **`gpu_pretraining_SAC.py`**: Added `apply_domain_randomization()` function
2. **`tensegrity_mjc_simulation.py`**: Updated observation space, reset method, reward function
3. **`test_trained_model_SAC.py`**: Added `test_directional_control()` function
4. **`tensegrity_env.py`**: Already uses dynamic `obs_dim` from simulator (no changes needed)

## Next Steps

1. Train model with domain randomization: `python gpu_pretraining_SAC.py`
2. Monitor training progress in TensorBoard
3. Test directional control: `python test_trained_model_SAC.py --test-directions`
4. Analyze performance metrics across different directions
5. Fine-tune reward weights if needed based on results

## Notes

- The observation space now includes goal direction, making the policy **goal-conditioned**
- Domain randomization is applied **at every reset** during pretraining
- The policy learns to **interpret the goal direction** from observations
- Testing can override goal direction to evaluate specific directions
- All changes maintain backward compatibility with existing code structure
