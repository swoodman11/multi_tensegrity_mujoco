# Critical Updates: Glide Detection & Directional Evaluation

**Date**: October 21, 2025  
**Status**: ✅ Critical Fixes Applied  
**Branch**: Zac_Test_3

---

## 🚨 Issue 1: Incorrect Glide Detection - FIXED

### Problem Identified

**Your observation was spot-on!** The original glide detection was fundamentally wrong:

**OLD Logic** (INCORRECT):
```python
# Checked for 8+ endpoints below 4cm
low_endpoints = int(np.sum(end_pts_z < 0.04))
if velocity_mag > 0.05 and low_endpoints >= 8:
    glide_penalty = -6.0 * velocity_mag
```

**Why This Failed**:
- Real gliding behavior: **3-4 endpoints on ground** (not 8+)
- Robot glides by **vibrating/oscillating controls** at high frequency
- Old detection would never trigger for actual gliding!

---

### Fix Applied

**NEW Logic** (CORRECT):
```python
# Detects vibration-based gliding: 3-6 endpoints grounded + high velocity + control oscillations
glide_penalty = 0.0
if velocity_mag > 0.04 and 3 <= num_ground <= 6:
    if hasattr(self, 'prev_controls') and self.prev_controls is not None:
        control_change = float(np.mean(np.abs(controls - self.prev_controls)))
        
        # High-frequency oscillations (>0.05 mean change = vibration)
        if control_change > 0.05:
            # STRONG penalty for vibration-based gliding
            glide_penalty = -12.0 * velocity_mag  # Doubled from -6.0
            glide_penalty = max(glide_penalty, -20.0)  # Increased cap
        else:
            # Medium penalty for fast movement with few contacts
            glide_penalty = -6.0 * velocity_mag
            glide_penalty = max(glide_penalty, -10.0)
```

**Key Improvements**:
1. ✅ **Correct ground contact range**: 3-6 endpoints (matches actual gliding)
2. ✅ **Vibration detection**: Measures control oscillation frequency
3. ✅ **Stronger penalty**: -12.0 per m/s for vibration gliding (vs -6.0 before)
4. ✅ **Higher cap**: -20.0 max penalty (vs -10.0 before)

---

## 🎯 Issue 2: Evaluation Only Tests Random Directions - FIXED

### Problem Identified

**Original evaluation** (from `EvalCallback`):
- Only tested random goal directions
- No systematic testing of challenging angles (especially 180-315°)
- Couldn't diagnose direction-specific failures

---

### Fix Applied: Custom DirectionalEvalCallback

**NEW Evaluation Strategy**:
```python
class DirectionalEvalCallback(BaseCallback):
    """
    Tests both random AND cardinal directions:
    - 8 random directions (generalization test)
    - 8 cardinal directions: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
    """
```

**What It Tests**:

1. **8 Random Episodes**: 
   - Tests generalization to arbitrary directions
   - Same as before, but now explicit

2. **8 Cardinal Directions** (NEW):
   - 0° (East, forward-right)
   - 45° (Northeast)
   - 90° (North, right)
   - 135° (Northwest)
   - **180° (West, behind)** ⬅️ Critical test
   - **225° (Southwest)** ⬅️ Critical test
   - **270° (South, left)** ⬅️ Critical test
   - **315° (Southeast)** ⬅️ Critical test

**TensorBoard Logging**:
```
eval/mean_random_reward       - Average of 8 random episodes
eval/mean_cardinal_reward     - Average of 8 cardinal directions
eval/mean_reward             - Overall average (used for best model)
eval/cardinal_0deg           - Reward for 0° direction
eval/cardinal_45deg          - Reward for 45° direction
eval/cardinal_90deg          - Reward for 90° direction
eval/cardinal_135deg         - Reward for 135° direction
eval/cardinal_180deg         - Reward for 180° direction ⭐ KEY METRIC
eval/cardinal_225deg         - Reward for 225° direction ⭐ KEY METRIC
eval/cardinal_270deg         - Reward for 270° direction ⭐ KEY METRIC
eval/cardinal_315deg         - Reward for 315° direction ⭐ KEY METRIC
```

**Console Output Example**:
```
Evaluation at 10000 steps:
  Random directions (n=8): 5234.12
  Cardinal directions:
      0°:  6123.45 ✓
     45°:  5987.23 ✓
     90°:  6234.56 ✓
    135°:  5456.78 ✓
    180°:  2345.67 ✗  ← LOW! Rear-facing issue detected
    225°:  2678.90 ✗  ← LOW! Rear-facing issue detected
    270°:  5890.12 ✓
    315°:  4123.45 ✓
  Overall mean: 4854.27
  Best mean: 4854.27
```

---

## Files Modified

### 1. `mujoco_physics_engine/tensegrity_mjc_simulation.py`

**Lines 746-769** (Glide Detection):

```python
# BEFORE (WRONG):
low_endpoints = int(np.sum(end_pts_z < 0.04))
if velocity_mag > 0.05 and low_endpoints >= 8:
    glide_penalty = -6.0 * velocity_mag
    glide_penalty = max(glide_penalty, -10.0)

# AFTER (CORRECT):
if velocity_mag > 0.04 and 3 <= num_ground <= 6:
    control_change = float(np.mean(np.abs(controls - self.prev_controls)))
    if control_change > 0.05:
        glide_penalty = -12.0 * velocity_mag  # Strong penalty for vibration
        glide_penalty = max(glide_penalty, -20.0)
    else:
        glide_penalty = -6.0 * velocity_mag
        glide_penalty = max(glide_penalty, -10.0)
```

---

### 2. `gpu_pretraining_SAC.py`

**Lines 35-142** (NEW DirectionalEvalCallback class):
- Added custom callback class with cardinal direction testing
- Logs 10 metrics per evaluation (vs 1 before)
- Provides detailed console output with per-direction results

**Lines 519-531** (Callback Usage):
```python
# BEFORE:
eval_callback = EvalCallback(
    eval_env,
    ...
    n_eval_episodes=10,
    ...
)

# AFTER:
eval_callback = DirectionalEvalCallback(
    eval_env,
    ...
    n_random_episodes=8,  # 8 random + 8 cardinal = 16 episodes per eval
    ...
)
```

---

## Expected Impact

### Glide Detection Fix

**Immediate (Next 10k steps)**:
- ✅ Robot should immediately experience penalties when gliding
- ✅ Vibration-based locomotion becomes costly
- ⚠️ May see temporary reward dip as policy adjusts
- ✅ Should force robot to develop proper walking/rolling gaits

**Medium-term (50k steps)**:
- ✅ Gliding episodes should drop to near-zero
- ✅ Control smoothness may improve (vibration is penalized)
- ✅ Reward should recover and exceed previous levels

---

### Directional Evaluation Fix

**Immediate**:
- ✅ Can now diagnose direction-specific failures in TensorBoard
- ✅ 180-315° performance explicitly tracked
- ✅ Best model selection based on overall performance (not just lucky random samples)

**TensorBoard Analysis**:
```python
# You can now check:
# 1. Overall trends
tensorboard --logdir=./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient

# 2. Look for these new metrics:
#    - eval/cardinal_180deg  (should improve with abs() fix)
#    - eval/cardinal_225deg  (should improve with abs() fix)
#    - eval/cardinal_270deg
#    - eval/cardinal_315deg

# 3. Success = all cardinal directions within 20% of each other
```

---

## Testing Protocol

### Monitor These Metrics in TensorBoard

**Primary Metrics**:
1. **`eval/mean_reward`**: Overall performance (random + cardinal average)
2. **`eval/mean_cardinal_reward`**: Average across 8 cardinal directions
3. **`eval/cardinal_180deg`**: Critical test for rear-facing (was problematic)

**Success Criteria** (by 300k steps):
- ✅ All cardinal directions achieve >3,000 reward
- ✅ No direction <70% of best direction
- ✅ `eval/cardinal_180deg` within 80% of `eval/cardinal_0deg`
- ✅ Standard deviation across cardinal directions <25%

---

### Verification Commands

**1. Check if glide penalty is triggering:**
```python
# Add debug print in tensegrity_mjc_simulation.py temporarily:
if glide_penalty < -5.0:
    print(f"GLIDE DETECTED: vel={velocity_mag:.3f}, ground={num_ground}, penalty={glide_penalty:.2f}")
```

**2. Monitor new evaluation metrics:**
```bash
# Run TensorBoard
tensorboard --logdir=./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient

# Look for new scalar metrics under "eval" category:
# - cardinal_0deg through cardinal_315deg
# - mean_cardinal_reward
# - mean_random_reward
```

**3. Extract direction-specific data:**
```python
# Use analyze_tensorboard.py to extract
python analyze_tensorboard.py ./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient/SAC_41

# Look for eval/cardinal_* metrics in output
```

---

## Rollback Plan (If Needed)

### If Glide Penalty Too Harsh

**Symptoms**:
- Reward drops >30% and doesn't recover by +20k steps
- Robot becomes overly conservative (barely moving)

**Fix**:
```python
# Reduce penalty strength
glide_penalty = -8.0 * velocity_mag  # Instead of -12.0
glide_penalty = max(glide_penalty, -15.0)  # Instead of -20.0
```

---

### If Evaluation Takes Too Long

**Symptoms**:
- Eval frequency slows training significantly
- Each evaluation takes >5 minutes

**Fix Option 1** - Reduce cardinal tests:
```python
# Test 4 cardinal directions instead of 8
self.cardinal_angles = [0, 90, 180, 270]  # Main 4 directions
```

**Fix Option 2** - Reduce eval frequency:
```python
eval_freq=20000,  # Instead of 10000 (half as often)
```

---

## Migration from SAC_41 to SAC_42

**If you're starting a new run with these fixes**:

1. The new evaluation metrics won't show in SAC_41 logs (old callback)
2. New run (SAC_42) will have complete directional metrics from start
3. Can compare:
   - SAC_41: `eval/mean_reward` (overall, random only)
   - SAC_42: `eval/mean_reward` (overall, random + cardinal)
   - SAC_42: `eval/cardinal_180deg` (specific rear-facing test)

**Recommendation**: Let SAC_41 continue to completion (500k), then start SAC_42 with these fixes for clean comparison.

---

## Key Insights from Your Observations

1. **Gliding Mechanism**: You correctly identified the robot uses **vibration** (high-frequency control oscillations) with only 3-4 endpoints on ground - not the 8+ endpoint "hover" I initially assumed.

2. **Evaluation Blind Spot**: Random direction testing wasn't catching the systematic failure at 180-315° angles. Cardinal direction testing makes failures obvious.

3. **Reward Function Design**: The `abs()` fix for bidirectional turning was correct, but we couldn't verify it was working without testing those specific angles.

---

## Summary of All Active Fixes

| Fix | File | Status | Purpose |
|-----|------|--------|---------|
| Bidirectional turning (`abs()`) | tensegrity_mjc_simulation.py | ✅ Active | Allow turning either direction to goal |
| Hover weight increase (3.0→8.0) | tensegrity_mjc_simulation.py | ✅ Active | General anti-glide |
| **Vibration glide detection** | **tensegrity_mjc_simulation.py** | **✅ NEW** | **Detect 3-4 point oscillation gliding** |
| **Cardinal direction eval** | **gpu_pretraining_SAC.py** | **✅ NEW** | **Test 8 specific angles including 180-315°** |

---

## Next Steps

1. **✅ DONE**: Applied vibration glide detection fix
2. **✅ DONE**: Added DirectionalEvalCallback with 8 cardinal direction tests
3. **⏳ WAITING**: Let training continue, monitor for glide penalty activation
4. **📊 MONITOR**: Watch TensorBoard for new `eval/cardinal_*` metrics (will appear at next eval interval)

**Next Checkpoint**: 160k-170k steps (next evaluation with new callback)
- Should see 10 new eval metrics appear
- Check if 180°, 225°, 270°, 315° improve with abs() fix
- Verify glide penalty reduces gliding behavior

---

**Status**: Critical fixes applied. Training can continue with proper glide detection and comprehensive directional evaluation.
