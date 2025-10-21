# Quick Summary: SAC_41 Analysis & Fixes

**Date**: October 21, 2025  
**Status**: ✅ Fixes Applied, Training Continuing

---

## 🎉 Great News

**Training is working!** Reward improved from -2,535 to +8,321 (eval) over 150k steps.

---

## ⚠️ Issues Found & Fixed

### Issue 1: Robot Struggles Turning to 180-315° Directions

**Problem**: Reward function penalized robot for moving when goal was behind it.

**Root Cause**: 
```python
# OLD - penalized backward movement
rotation_alignment_reward = 5.0 * alignment  
```

**Fix Applied**:
```python
# NEW - rewards movement toward goal from ANY direction
rotation_alignment_reward = 5.0 * abs(alignment)
```

**Why This Works**: Tensegrity robots are bidirectional - they can roll/walk with either "front" or "back" toward the goal. Using `abs()` rewards movement toward the goal regardless of robot orientation.

---

### Issue 2: Robot Occasionally Glides/Skates

**Problem**: Robot sometimes glides at low altitude instead of proper walking/rolling.

**Fixes Applied**:

1. **Increased hover penalty**: 3.0 → 8.0 (line 746)
2. **Added direct glide penalty** (NEW, lines 748-757):
```python
glide_penalty = 0.0
if velocity_mag > 0.05 and low_endpoints >= 8:  # Fast movement with most endpoints very low
    glide_penalty = -6.0 * velocity_mag  # Strong penalty proportional to speed
    glide_penalty = max(glide_penalty, -10.0)  # Cap at -10
```

**Why This Works**: 
- Direct detection of gliding (not gated by stall conditions)
- Proportional to velocity (faster glides penalized more)
- Immediate feedback (no need to stall for 5 steps first)

---

## 📊 Expected Outcomes

### Short-Term (Next 50k steps):
- ✅ Robot learns to turn toward rear-facing goals
- ✅ Gliding reduces significantly
- ⚠️ Reward may dip slightly (200-210k) as policy adjusts, then recover by 250k

### Long-Term (300k+ steps):
- ✅ Consistent performance across all 360° goal directions
- ✅ Clean walking/rolling gaits with minimal gliding
- ⚠️ Monitor critic loss - if exceeds 3,500, may need architectural changes

---

## 🎯 Next Steps

### Immediate (Now):
1. ✅ **DONE**: Applied reward fixes
2. ⏳ **Continue training** to 500k steps
3. 📊 **Monitor TensorBoard** for reward trends and critic loss

### Decision Points:

**At 200k steps** (~2 hours):
- Check if reward dips due to glide penalty
- If drops >20%, reduce glide penalty strength

**At 300k steps** (~6 hours):
- Test turning behavior on 180-315° goals
- If critic loss >3,500, consider stopping and increasing network size

**At 500k steps** (completion):
- Full evaluation across all directions (0-360°)
- Decide: continue to 1M steps OR implement architectural changes

---

## 📁 Files Modified

1. **`mujoco_physics_engine/tensegrity_mjc_simulation.py`**
   - Line 746: Increased hover_weight (3.0 → 8.0)
   - Lines 748-757: Added direct glide_penalty
   - Line 786: Changed rotation_alignment to use abs() for bidirectional turning
   - Line 831: Added glide_penalty to reward composition

---

## 🔍 Testing Protocol

After training completes, test specific directions:
```python
test_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 directions

# Success criteria:
# - All directions achieve >3,000 average reward
# - No direction shows <50% of best direction
# - Coefficient of variation <30% across all directions
```

---

## ⚠️ Known Issue: Critic Loss

**Observation**: Critic loss still increasing (844 → 2,278)

**Why Training Still Works**: SAC is robust to critic errors - actor can still learn from noisy Q-value gradients.

**Monitoring**: Watch for explosion >5,000 which could indicate instability.

**Contingency**: If critic loss exceeds 3,500 at 300k steps, consider increasing critic network size:
```python
"policy_kwargs": dict(
    net_arch=dict(
        pi=[512, 512, 256],      # Actor unchanged
        qf=[1024, 1024, 512]     # Critic LARGER
    )
)
```

---

## 📈 Current Training Status

| Metric | Value | Status |
|--------|-------|--------|
| Training Reward | +5,509 | ✅ Improving |
| Eval Reward | +8,321 | ✅ Excellent |
| Episode Length | 500 | ✅ Consistent |
| Critic Loss | 2,278 | ⚠️ Increasing |
| FPS | 17 | ⚠️ Slow but acceptable |

---

## 📖 Full Details

See `SAC_41_ANALYSIS_AND_FIXES.md` for comprehensive analysis including:
- Detailed root cause analysis with code examples
- Full reward component breakdown
- Comparison to previous training runs
- Technical notes on reward scaling
- Complete testing protocol

---

**Status**: Ready to continue training. Fixes should improve turning behavior and reduce gliding. Monitor TensorBoard at 200k and 300k step checkpoints.
