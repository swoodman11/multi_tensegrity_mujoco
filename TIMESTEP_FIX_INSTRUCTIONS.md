# Timestep and Simulation Fix Instructions

## 🎯 Mission Statement

Fix critical timing mismatch between action frequency, physics simulation, and rendering across the **entire repository**. This document guides you to implement consistent timing behavior in all training, testing, and demonstration scripts.

---

## 📋 Problem Description

### The Original Issue

**Critical Timing Mismatch Discovered**: The simulation had a severe inconsistency between behavioral cloning demonstrations and reinforcement learning training.

#### What Was Wrong:

1. **Behavioral Cloning (BC) Demonstrations**:
   - External scripts (e.g., `run_single.py`) executed **100 physics timesteps per action**
   - Each action lasted **1.0 second** of simulated time
   - This was the intended behavior for realistic gait execution

2. **Reinforcement Learning Training**:
   - The `sim_step()` method in simulators executed **only 1 physics timestep per action**
   - Each action lasted **0.01 seconds** of simulated time
   - This created a **100× time scale mismatch** with BC demonstrations

3. **Consequence**:
   - RL agents were being trained on a completely different timescale than demonstrations
   - Gait patterns that worked in demonstrations would not transfer to RL policies
   - A 5-second episode in RL was actually only 0.05 seconds of physics time

#### Root Cause:

The `sim_step()` method in both simulator classes only called `mujoco.mj_step()` once, instead of looping for the intended action duration.

---

## ✅ Solution Implemented

### Core Principle:

**"One action = 100 physics timesteps = 1.0 second of simulated time"**

This matches the intended design where:
- MuJoCo physics timestep (`dt`) = 0.01 seconds
- Control frequency = 1 Hz (one action per second)
- Physics frequency = 100 Hz (100 steps per second)

---

## 🔧 Files Modified (Reference Implementation)

### 1. Single Robot Simulator
**File**: `mujoco_physics_engine/single_tensegrity_mjc_simulation.py`

**Location**: Inside the `sim_step()` method (around line 400-450)

**Changes Made**:

```python
def sim_step(self, target_lengths: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
    """Execute one control step (1 second = 100 physics timesteps)."""
    
    # Calculate number of physics steps per action
    steps_per_action = int(1.0 / self.dt)  # 100 steps for dt=0.01
    
    # Execute action for full duration (100 physics timesteps)
    for step_i in range(steps_per_action):
        # Update cable motors with PID control
        for cable_id in range(self.n_actuators):
            self.cable_motors[cable_id].set_target_length(target_lengths[cable_id])
            current_length = self._get_cable_length(cable_id)
            force = self.cable_motors[cable_id].compute_force(current_length, self.dt)
            self.mjc_data.ctrl[cable_id] = force
        
        # Step physics once
        mujoco.mj_step(self.mjc_model, self.mjc_data)
        
        # Render every 5th physics step when visualization enabled (20 fps)
        if self.visualize and step_i % 5 == 0:
            self.render()
            time.sleep(self.render_pause)  # Default 0.01s for smooth playback
    
    # After all physics steps, get observation and compute reward
    obs = self.get_observation()
    reward, done, info = self._compute_reward()
    
    return obs, reward, done, info
```

**Key Changes**:
- ✅ Added `steps_per_action = int(1.0 / self.dt)` calculation
- ✅ Wrapped physics step in `for step_i in range(steps_per_action):` loop
- ✅ Added internal rendering every 5th physics step (when `visualize=True`)
- ✅ Added configurable `render_pause` for visualization speed control
- ✅ Reward computation happens **after** all 100 physics steps complete

**Added Constructor Parameter**:
```python
def __init__(self, ..., render_pause: float = 0.01):
    # ...
    self.render_pause = render_pause
```

---

### 2. Dual Robot Simulator
**File**: `mujoco_physics_engine/tensegrity_mjc_simulation.py`

**Location**: Inside the `sim_step()` method (around line 230-280)

**Changes Made**: 
- **Identical structure** to single robot simulator
- Same 100-step loop with internal rendering
- Same `render_pause` parameter and logic

---

### 3. Single Robot Environment
**File**: `single_tensegrity_env.py`

**Changes Made**:

```python
class SingleTensegrityEnv(gym.Env):
    def __init__(self, ..., max_episode_steps=50, render_pause=0.01):
        # ...
        self.max_episode_steps = max_episode_steps  # 50 steps = 50 seconds
        
        self.sim = SingleTensegrityMuJoCoSimulator(
            xml_path=xml_path,
            # ... other params ...
            render_pause=render_pause  # Pass to simulator
        )
```

**Key Changes**:
- ✅ Reduced `max_episode_steps` from **500 → 50** (still 50 seconds of sim time)
- ✅ Added `render_pause` parameter and passed to simulator
- ✅ Added episode truncation logic with `_elapsed_steps` counter

---

### 4. Dual Robot Environment  
**File**: `tensegrity_env.py`

**Changes Made**:
- **Identical structure** to single robot environment
- `max_episode_steps=50`
- `render_pause` parameter passed to simulator

---

### 5. Training Scripts (All 4 Updated)

**Files**:
- `cpu_pretraining_SAC_single.py`
- `gpu_pretraining_SAC_single.py`
- `cpu_pretraining_SAC.py` (dual robot)
- `gpu_pretraining_SAC.py` (dual robot)

**Changes Made**:

```python
# Before (incorrect):
total_timesteps = 100000  # Would be 100k seconds = 27.8 hours simulated

# After (correct):
total_timesteps = 10000  # 10k seconds = 2.78 hours simulated (reasonable)

# Evaluation frequency adjusted proportionally
eval_freq = 1000  # Was 10000, reduced by 10x
```

**Key Changes**:
- ✅ Reduced training timesteps by 10× (e.g., 100k → 10k)
- ✅ Reduced eval frequency by 10× (e.g., 10k → 1k)
- ✅ Episode length already reduced in environments

---

### 6. Demonstration Scripts

**File**: `run_single.py`

**Changes Made**:

```python
# Before (double-loop, redundant):
for cycle in range(num_cycles):
    for action in gait_sequence:
        for _ in range(100):  # ❌ External 100-step loop
            sim.sim_step(action)
            sim.render()  # ❌ External render calls

# After (clean, single call):
for cycle in range(num_cycles):
    for action in gait_sequence:
        sim.sim_step(action)  # ✅ Internally handles 100 steps + rendering
```

**Key Changes**:
- ✅ Removed external 100-step loop (now handled internally)
- ✅ Removed external `render()` calls (now handled internally)
- ✅ Removed `time.sleep()` (now handled by `render_pause`)

---

### 7. Test Model Scripts (All 4 Updated)

**Files**:
- `test_trained_model_SAC_single.py`
- `test_trained_model_SAC.py`
- `test_trained_model.py`
- `test_trained_model_TD3.py`

**Changes Made**:

```python
# Before (external rendering):
for step in range(max_timesteps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # ❌ Redundant external render
    time.sleep(0.01)  # ❌ External sleep

# After (clean):
for step in range(max_timesteps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    # ✅ Rendering handled internally by simulator
```

**Key Changes**:
- ✅ Removed all `env.render()` calls
- ✅ Removed all `time.sleep()` calls
- ✅ Default timesteps changed from 500 → 50 (still 50 seconds)

---

## 🎯 Your Task: Apply These Fixes Repository-Wide

### Step 1: Verify Reference Implementation

First, confirm the fixes are correctly implemented in the files listed above:

```bash
# Check simulator files have 100-step loop
grep -n "steps_per_action" mujoco_physics_engine/single_tensegrity_mjc_simulation.py
grep -n "steps_per_action" mujoco_physics_engine/tensegrity_mjc_simulation.py

# Check environments have max_episode_steps=50
grep -n "max_episode_steps=50" single_tensegrity_env.py
grep -n "max_episode_steps=50" tensegrity_env.py
```

---

### Step 2: Find All Similar Files

Use these commands to discover **all** training, testing, and execution scripts:

```bash
# Find all Python files that might need fixes
find . -name "*.py" -type f | grep -E "(train|test|run|pretrain|demo)" | sort

# Search for files that call sim_step or env.step
grep -r "sim_step\|env\.step" --include="*.py" | cut -d: -f1 | sort -u

# Search for files that have episode/timestep parameters
grep -r "total_timesteps\|max_episode_steps\|num_cycles" --include="*.py" | cut -d: -f1 | sort -u

# Search for files with render() calls
grep -r "\.render()" --include="*.py" | cut -d: -f1 | sort -u
```

---

### Step 3: Categorize Files Found

Organize discovered files into categories:

#### A. **Simulator Files** (Core Physics)
- Should have 100-step loop in `sim_step()`
- Should have internal rendering logic
- Should have `render_pause` parameter

#### B. **Environment Files** (Gym Wrappers)
- Should have `max_episode_steps=50`
- Should pass `render_pause` to simulator
- Should have truncation logic

#### C. **Training Scripts**
- Should have reduced timesteps (typically 10-20k, not 100k+)
- Should have proportional eval frequency (e.g., 500-1000)
- Should create environments with `visualize=False`

#### D. **Testing/Evaluation Scripts**
- Should NOT have external `env.render()` or `sim.render()` calls
- Should NOT have external `time.sleep()` calls
- Should use reasonable timesteps (50-100 steps = 50-100 seconds)

#### E. **Demonstration/Run Scripts**
- Should NOT have external 100-step loops around `sim_step()`
- Should NOT have external rendering calls
- Should call `sim_step()` once per action

---

### Step 4: Apply Fixes to Each Category

#### For Simulator Files:

**Check these files exist and need fixes**:
- Any files inheriting from `AbstractMuJoCoSimulator`
- Any custom simulator implementations
- Look for: `mujoco_simulation.py`, `*_simulation.py`, `*_simulator.py`

**Required changes**:
1. Add `render_pause: float = 0.01` parameter to `__init__`
2. Store `self.render_pause = render_pause`
3. Import `time` module at top
4. Modify `sim_step()` to have 100-step loop:
   ```python
   steps_per_action = int(1.0 / self.dt)
   for step_i in range(steps_per_action):
       # Motor updates
       # Physics step
       # Conditional rendering every 5th step
   ```

#### For Environment Files:

**Check these files exist and need fixes**:
- Any files defining `gym.Env` or `gymnasium.Env` classes
- Look for: `*_env.py`, `*environment.py`

**Required changes**:
1. Add `max_episode_steps=50` parameter to `__init__`
2. Add `render_pause=0.01` parameter to `__init__`
3. Pass `render_pause=render_pause` to simulator constructor
4. Add episode truncation logic:
   ```python
   self._elapsed_steps = 0
   
   def step(self, action):
       self._elapsed_steps += 1
       # ... existing step logic ...
       truncated = self._elapsed_steps >= self.max_episode_steps
       return obs, reward, terminated, truncated, info
   ```

#### For Training Scripts:

**Check these files exist and need fixes**:
- Files with names like: `train*.py`, `*training*.py`, `pretrain*.py`
- Files that call `model.learn()`

**Required changes**:
1. Review `total_timesteps` parameter:
   - If > 50k, consider reducing by 10× (or justify why not)
   - Document expected training time
2. Review `eval_freq` parameter:
   - Should be proportional to `total_timesteps`
   - Recommend 500-1000 for frequent evaluations
3. Ensure environments created with `visualize=False`
4. Check `max_episode_steps` is passed correctly

#### For Testing Scripts:

**Check these files exist and need fixes**:
- Files with names like: `test*.py`, `eval*.py`, `*test*.py`
- Files that load trained models and run episodes

**Required changes**:
1. **Remove all external render calls**:
   ```python
   # DELETE these lines:
   env.render()
   sim.render()
   time.sleep(0.01)
   ```
2. Update default timesteps:
   - If script has `--timesteps` arg, set default to 50-100 (not 500)
   - Update help text to mention 1 step = 1 second
3. Ensure environment created with `visualize=True`

#### For Demonstration Scripts:

**Check these files exist and need fixes**:
- Files with names like: `run*.py`, `demo*.py`, `*demo.py`
- Files that execute gaits or action sequences

**Required changes**:
1. **Remove external 100-step loops**:
   ```python
   # DELETE this pattern:
   for _ in range(100):
       sim.sim_step(action)
   
   # REPLACE with:
   sim.sim_step(action)
   ```
2. **Remove external rendering**:
   ```python
   # DELETE these:
   sim.render()
   time.sleep(0.01)
   ```

---

### Step 5: Verification Checklist

After applying fixes, verify each file:

#### Simulator Verification:
- [ ] `sim_step()` has 100-step loop
- [ ] `render_pause` parameter exists
- [ ] Internal rendering every 5th step
- [ ] `time` module imported
- [ ] Reward computed after all physics steps

#### Environment Verification:
- [ ] `max_episode_steps=50` (or appropriate value)
- [ ] `render_pause` parameter passed to simulator
- [ ] Truncation logic with `_elapsed_steps`
- [ ] `reset()` resets `_elapsed_steps = 0`

#### Training Script Verification:
- [ ] Reasonable `total_timesteps` (typically 10-20k)
- [ ] Proportional `eval_freq` (typically 500-1000)
- [ ] `visualize=False` for training environment
- [ ] Comments explain timestep choices

#### Testing Script Verification:
- [ ] No external `render()` calls
- [ ] No external `time.sleep()` calls
- [ ] Reasonable default timesteps (50-100)
- [ ] `visualize=True` for test environment
- [ ] Help text accurate

#### Demo Script Verification:
- [ ] No external 100-step loops
- [ ] No external `render()` calls
- [ ] No external `time.sleep()` calls
- [ ] Clean, single `sim_step()` calls

---

### Step 6: Testing Protocol

After fixes, test each category:

#### 1. Test Simulator Timing:
```python
# Create test script: test_simulator_timing.py
from single_tensegrity_env import SingleTensegrityEnv
import time

env = SingleTensegrityEnv(visualize=False)
obs, _ = env.reset()

# Time 100 steps (should be ~100 seconds of sim time)
start = time.time()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        break

elapsed = time.time() - start
print(f"100 RL steps took {elapsed:.2f}s real time")
print(f"Expected: 100 seconds simulated time")
```

#### 2. Test Visualization:
```bash
# Should show smooth rendering at reasonable speed
python run_single.py handmade_gait_best_rolling.json 5
```

#### 3. Test Training:
```bash
# Should complete in expected time without errors
python cpu_pretraining_SAC_single.py
```

#### 4. Test Model:
```bash
# Should show smooth visualization
python test_trained_model_SAC_single.py --model <path>
```

---

## 📊 Summary of Timing Parameters

After all fixes, the repository should have consistent timing:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `dt` | 0.01s | MuJoCo physics timestep |
| `steps_per_action` | 100 | Physics steps per RL action |
| Action duration | 1.0s | Real-time per RL step |
| Control frequency | 1 Hz | RL actions per second |
| Physics frequency | 100 Hz | Physics steps per second |
| `max_episode_steps` | 50 | Episode length (50 seconds) |
| Render frequency | 20 Hz | Frames per second (every 5th step) |
| `render_pause` | 0.01s | Delay between renders |

---

## 🚨 Common Pitfalls to Avoid

### ❌ Don't Do This:
```python
# External 100-step loop (WRONG)
for _ in range(100):
    sim.sim_step(action)

# External rendering (WRONG)
env.step(action)
env.render()
time.sleep(0.01)

# Inconsistent episode lengths (WRONG)
env1 = Env(max_episode_steps=500)  # 500 seconds
env2 = Env(max_episode_steps=50)   # 50 seconds
```

### ✅ Do This Instead:
```python
# Single sim_step call (CORRECT)
sim.sim_step(action)  # Internally does 100 physics steps

# No external rendering (CORRECT)
env.step(action)  # Internally renders if visualize=True

# Consistent episode lengths (CORRECT)
env1 = Env(max_episode_steps=50)  # 50 seconds
env2 = Env(max_episode_steps=50)  # 50 seconds
```

---

## 📝 Reporting Back

After completing all fixes, create a summary report:

```markdown
# Timestep Fix Implementation Report

## Files Modified:
- [ ] List all files changed
- [ ] Categorize by type (simulator/env/train/test/demo)

## Tests Passed:
- [ ] Simulator timing test
- [ ] Visualization test  
- [ ] Training script test
- [ ] Model testing script

## Issues Encountered:
- Document any problems or uncertainties
- List any files that need special attention

## Verification:
- Confirm all timestep parameters consistent
- Confirm no external rendering loops remain
- Confirm episode lengths standardized
```

---

## 🎓 Understanding the Fix

**The Key Insight**: 

In reinforcement learning, "one timestep" should represent one **control decision**, not one **physics step**. Our robots operate with 1 Hz control frequency (one decision per second), which requires 100 physics iterations at 100 Hz to simulate properly.

**Before**: RL agent made 100 decisions per second (too fast, unrealistic)
**After**: RL agent makes 1 decision per second (correct, matches hardware)

This fix ensures that:
1. ✅ BC demonstrations and RL training use the same timescale
2. ✅ Episode lengths are manageable (50 seconds vs 5 seconds)
3. ✅ Gait patterns transfer correctly from demos to learned policies
4. ✅ Visualization runs at smooth, watchable speeds
5. ✅ Training times are reasonable and predictable

---

## 📚 Reference Implementation Branch

The fixes described here are implemented in the **`Zac_Test_2`** branch.

You can diff against this branch to see exact changes:
```bash
git diff origin/main origin/Zac_Test_2 -- mujoco_physics_engine/
git diff origin/main origin/Zac_Test_2 -- *_env.py
git diff origin/main origin/Zac_Test_2 -- *training*.py
git diff origin/main origin/Zac_Test_2 -- test_*.py
git diff origin/main origin/Zac_Test_2 -- run*.py
```

---

## ✅ Success Criteria

The repository is fully fixed when:

1. ✅ All simulators execute 100 physics steps per `sim_step()`
2. ✅ All environments have `max_episode_steps=50` (or justified alternative)
3. ✅ All training scripts use appropriate timesteps (10-20k typical)
4. ✅ All test scripts have no external render/sleep calls
5. ✅ All demo scripts have no external 100-step loops
6. ✅ Visualization is smooth and controllable
7. ✅ Training completes in expected time
8. ✅ Timing parameters are consistent across all files

---

## 🆘 Need Help?

If you encounter issues:

1. **Check the reference files** in the list above
2. **Compare your changes** with the patterns shown in this document
3. **Test incrementally** - don't change everything at once
4. **Document uncertainties** - note any files you're unsure about
5. **Ask questions** - better to clarify than to introduce new bugs

Good luck! 🚀
