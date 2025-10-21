# SAC_41 Implementation - Walking/Rolling Gait Optimization

## 🎯 Objective
Transform reward structure to eliminate sliding/gliding and encourage:
1. **Walking gaits** - lifting feet off ground
2. **Rolling gaits** - heavy reward for x-axis rotation
3. **Directional control** - strong alignment with goal direction
4. **Rotation alignment** - turning to face goal before/during movement

---

## ✅ Changes Implemented

### **1. Disabled Sliding Reward** ❌ → 0.0
```python
grounded_centroid_xy_reward_weight = 0.0  # Was 2.0
# This was REWARDING sliding - now disabled completely
```

### **2. Boosted Directional Velocity** 25.0 → 35.0 (+40%)
```python
directional_velocity_reward = 35.0 * float(np.tanh(5.0 * directional_velocity))
# Primary movement signal - strongly rewards goal-directed motion
```

### **3. Increased Perpendicular Penalty** -3.0 → -5.0 (+67%)
```python
perpendicular_penalty = -5.0 * abs(perpendicular_velocity)
# Punishes sideways drift much harder
```

### **4. Reduced Generic Movement** 25.0 → 10.0 (-60%)
```python
+ 10.0 * com_step_progress  # Was 25.0
# Less reward for direction-agnostic movement
```

---

## 🆕 New Reward Terms

### **5. Rotation Alignment Reward** (Weight: 5.0)
```python
rotation_alignment_reward = 0.0
if hasattr(self, 'prev_pos') and self.prev_pos is not None:
    velocity_xy = (robot_pos[:2] - self.prev_pos[:2]) / self.dt
    velocity_mag = float(np.linalg.norm(velocity_xy))
    if velocity_mag > 0.01:  # Only when moving
        velocity_unit = velocity_xy / velocity_mag
        alignment = float(np.dot(velocity_unit, self.goal_direction))
        rotation_alignment_reward = 5.0 * alignment  # Weight: 5.0
```
**Purpose:** Rewards moving in the direction you're facing (encourages rotation toward goal)

### **6. Foot Lift Reward** (Weight: +3.0 / -2.0)
```python
foot_lift_reward = 0.0
num_lifted = int(np.sum(lifted_mask))
num_grounded = int(np.sum(ground_mask))
total_points = len(lifted_mask)
if total_points > 0:
    lift_ratio = num_lifted / total_points
    if 0.3 <= lift_ratio <= 0.7:
        foot_lift_reward = 3.0  # Balanced stance (walking)
    elif lift_ratio < 0.1:
        foot_lift_reward = -2.0  # All grounded (sliding) - PENALIZE
    elif lift_ratio > 0.9:
        foot_lift_reward = -1.0  # All lifted (falling)
```
**Purpose:** 
- Rewards 30-70% feet lifted (walking gait)
- **Penalizes <10% lifted (sliding/gliding)**
- Penalizes >90% lifted (falling)

### **7. Rolling Reward - HEAVY** (Weight: 15.0)
```python
rolling_reward = 0.0
try:
    imu_ang = self._get_IMU_angular_velocities()
    if len(imu_ang) >= 3:
        # Extract x-axis angular velocities (every 3rd element)
        x_angular_vels = imu_ang[::3]
        # Mean absolute x-axis rotation across all IMUs
        x_rotation_magnitude = float(np.mean(np.abs(x_angular_vels)))
        # HEAVY reward for rolling motion
        rolling_reward = 15.0 * float(np.tanh(2.0 * x_rotation_magnitude))
except Exception:
    rolling_reward = 0.0
```
**Purpose:** **STRONGLY rewards x-axis rotation (rolling gait)** - highest single reward term!

---

## 📊 New Reward Balance

### **Movement Rewards (Directional):**
| Term | Weight | Type |
|------|--------|------|
| Directional velocity | **35.0** max | PRIMARY signal |
| Rolling (x-axis rotation) | **15.0** max | HEAVY incentive |
| COM progress | **10.0** | Backup signal |
| Rotation alignment | **5.0** | Directional aid |
| Foot lift (walking) | **3.0** | Gait shaping |
| Lifted centroid | **1.0** | Minor |
| **Total Directional** | **~69** | |

### **Penalties:**
| Term | Weight | Type |
|------|--------|------|
| Perpendicular drift | **-5.0** | Anti-slide |
| Action smoothing | **-3.0** max | Anti-jitter |
| Foot lift (all grounded) | **-2.0** | Anti-slide |
| Stall | **-0.5** max | Anti-stop |
| **Total Penalties** | **~-10.5** | |

### **Ratio: 6.6:1** (Directional focus)

---

## 🎭 Key Behavioral Changes Expected

### **Before (SAC_40):**
- ❌ Slides along ground (rewarded by `grounded_centroid_xy_reward`)
- ❌ Moves in random directions (direction-agnostic rewards dominated)
- ❌ No rotation toward goal
- ❌ Keeps all feet on ground

### **After (SAC_41):**
- ✅ **Rolling motion heavily rewarded** (15.0 weight on x-axis rotation)
- ✅ **Lifts feet to walk** (3.0 reward for balanced lift, -2.0 penalty for all-grounded)
- ✅ **Follows goal direction** (35.0 directional + 5.0 alignment)
- ✅ **Rotates to face goal** (alignment reward encourages turning)
- ✅ **Minimal sliding** (0.0 grounded reward, -2.0 penalty, -5.0 drift penalty)

---

## 🔍 Debugging Info Added

New fields in `info` dict:
- `rotation_alignment_reward`: How well velocity aligns with goal
- `foot_lift_reward`: Reward/penalty for lift ratio
- `rolling_reward`: X-axis rotation reward
- `num_lifted`: Number of endpoints off ground
- `num_grounded`: Number of endpoints on ground

**Usage:** Check TensorBoard or print info during episodes to see which rewards dominate.

---

## 📈 Expected Training Results (150k steps)

| Metric | SAC_40 (75k) | SAC_41 Target (150k) | Reasoning |
|--------|--------------|----------------------|-----------|
| eval/mean_reward | +547 | **+2,500 to +3,500** | Better reward alignment + more training |
| rollout/ep_rew_mean | -629 | **+1,500 to +2,500** | Should be strongly positive by 150k |
| Critic loss | 904 | **<500** | Simpler, more consistent rewards stabilize |
| Actor loss | -2,495 | **-1,500 to -2,000** | Similar aggressive learning |
| Visual: Direction | Poor | **Good** | Strong directional signals (35.0 weight) |
| Visual: Gait | Sliding | **Heavy Rolling** | 15.0 rolling reward dominates |
| Visual: Rotation | None | **Active turning** | Alignment reward (5.0) |

### Checkpoint Expectations:
- **50k**: Break positive, see early rolling behavior
- **100k**: Strong rolling, good directional control, eval +1,500
- **150k**: Refined rolling/walking, eval +2,500-3,500

---

## 🚀 Training Configuration

```python
learning_rate: 1.5e-4
timesteps: 150_000  # Extended from 75k to 150k
eval_freq: 10000    # Changed from 5k to 10k (15 total evaluations)
n_eval_episodes: 10
batch_size: 2048
gamma: 0.99
ent_coef: 0.05
net_arch: [512, 512, 256]
```

---

## ⏱️ Timeline

- **Now**: Changes implemented and ready
- **+3 hours**: Training complete (150k steps)
- **Visual test**: Check for walking/rolling gaits and directional control at 100k and 150k
- **If successful**: Launch 500k training (~10 hours)
- **Total to production**: ~13 hours

---

## 🎯 Success Criteria

### At 150k steps, robot should:
1. ✅ **Roll frequently** (high `rolling_reward` in logs)
2. ✅ **OR walk with lifted feet** (30-70% lift ratio)
3. ✅ **Follow goal direction consistently** (high `directional_velocity_reward`)
4. ✅ **Rotate to align** before moving (positive `rotation_alignment_reward`)
5. ✅ **Minimal sliding** (low `grounded_centroid_xy_reward`, `foot_lift_reward` not -2.0)
6. ✅ **Eval reward > +2,500** (at 150k)

### If any criteria fail:
- Rolling too dominant → Reduce `rolling_reward` from 15.0 to 10.0
- Not enough rotation → Increase `rotation_alignment_reward` from 5.0 to 8.0
- Still sliding → Further increase `perpendicular_penalty` to -7.0
- Poor direction → Further boost `directional_velocity_reward` to 40.0

---

## 📝 Files Modified

1. `mujoco_physics_engine/tensegrity_mjc_simulation.py`:
   - Lines ~741-742: Disabled grounded reward (0.0)
   - Lines ~722-725: Boosted directional (35.0), perpendicular (-5.0)
   - Lines ~757-802: Added 3 new reward terms
   - Lines ~804-819: Updated reward_raw calculation
   - Lines ~855-860: Added debug info fields

---

## 🔄 Quick Rollback (If Needed)

If results are worse, revert to SAC_40 settings:
```python
grounded_centroid_xy_reward_weight = 2.0  # Re-enable
directional_velocity_reward = 25.0  # Reduce
perpendicular_penalty = -3.0  # Reduce
com_step_progress weight = 25.0  # Restore
# Comment out 3 new reward terms
```

---

## 🎓 Key Insight

**The fundamental fix:** Removing `grounded_centroid_xy_reward` (which rewarded sliding) and adding heavy `rolling_reward` creates a clear incentive structure:

**Old:** "Move ground points = good" → Sliding optimal ❌  
**New:** "Rotate x-axis = best, lift feet = good, slide = bad" → Walking/Rolling optimal ✅

The 15.0 weight on `rolling_reward` makes it the **second highest reward term** after directional velocity (35.0 max), ensuring the robot learns rolling as a primary gait option.
