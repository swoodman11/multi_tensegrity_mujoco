# SAC_40 Analysis - 75k Steps with LR 1.5e-4

## 📊 TensorBoard Results Summary

### Key Metrics at 75k Steps:

| Metric | SAC_40 (Purple) | SAC_34 Baseline (Green) | Assessment |
|--------|-----------------|-------------------------|------------|
| **eval/mean_reward** | +547 | +4,819 (500k) | ⚠️ Severely underperforming |
| **rollout/ep_rew_mean** | -629 | +5,238 (500k) | ❌ Still negative |
| **train/actor_loss** | -2,495 | -1,121 | ⚠️ More aggressive but not helping |
| **train/critic_loss** | 904 | 284 | ❌ 3.2x higher - unstable value estimation |
| **Episode length** | 500 (capped) | 500 | ✅ Consistent |
| **Learning rate** | 0.0002 (1.5e-4) | 0.0002 | ✅ Fixed |

---

## 🔴 Critical Problems Identified

### 1. **Poor Directional Control**
**Your observation:** "Robot doesn't follow desired direction at all"

**Root cause:** Direction-agnostic rewards dominate:
- `grounded_centroid_xy_reward`: Weight 2.0 - **rewards ANY ground movement**
- `lifted_centroid_xy_reward`: Weight 1.0 - **rewards ANY lifted movement**  
- `com_step_progress`: Weight 25.0 - **rewards ANY COM movement**
- **Total direction-agnostic: ~28 weight**

vs.

- `directional_velocity_reward`: 25.0 max - **only reward for goal direction**
- `perpendicular_penalty`: -3.0 - **weak punishment for drift**
- **Total directional: ~22 net weight**

**Result:** Robot gets more reward moving sideways/backwards than struggling to go forward.

---

### 2. **Excessive Sliding/Gliding**
**Your observation:** "Robot slides its feet more than it shuffles or rolls"

**Root cause:** `grounded_centroid_xy_reward` **directly rewards ground sliding**
```python
# Current code (line ~605-617)
if np.any(ground_mask):
    grounded_xy = end_pts[ground_mask][:, :2]
    g_centroid = grounded_xy.mean(axis=0)
    if hasattr(self, 'prev_grounded_centroid_xy'):
        g_disp = float(np.linalg.norm(g_centroid - self.prev_grounded_centroid_xy))
        grounded_centroid_xy_reward = g_disp  # ← REWARDS SLIDING
```

**This is the opposite of what you want!** It rewards:
- ✅ Controlled slip (dragging feet)
- ✅ Uncontrolled slip (passive sliding)
- ❌ Lifting feet (no ground contact = no reward)

---

### 3. **No Rotation Incentive**
**Your observation:** "Robot struggles to rotate to face desired direction"

**Problem:** Zero rewards for:
- Aligning robot orientation with goal direction
- Angular velocity toward goal heading
- Reducing angular misalignment

**Current state:** Robot has no reason to rotate before moving.

---

### 4. **High Critic Loss = Value Function Can't Learn**
- Critic loss **904** vs SAC_34's **284** (3.2x worse)
- Actor loss more negative (-2,495) but not translating to better policy
- **Diagnosis:** Reward structure too complex/contradictory for value estimation

---

## 🎯 Recommended Solution

### **Major Reward Restructuring for Walking/Rolling Gaits**

Based on your goals:
1. ✅ Lift feet to walk (not slide)
2. ✅ Roll as alternative gait
3. ✅ Follow goal direction
4. ✅ Rotate to align with goal
5. ❌ No controlled/uncontrolled sliding

---

## 🔧 Implementation Plan

### **Step 1: Remove Sliding Rewards**

```python
# REMOVE/DISABLE grounded_centroid_xy_reward completely
grounded_centroid_xy_reward_weight = 0.0  # Was 2.0 - ELIMINATES sliding incentive
```

**Rationale:** This reward directly contradicts your walking/rolling goal.

---

### **Step 2: Add Rotation Alignment Reward**

```python
# NEW: Reward rotating to face goal direction
def calculate_rotation_alignment_reward(self):
    """
    Reward for robot orientation aligning with goal direction.
    Encourages rotation before/during movement.
    """
    # Get robot's forward direction (you'll need to extract from qpos/IMU)
    # Simplified: use velocity direction as proxy for orientation
    if hasattr(self, 'prev_pos') and self.prev_pos is not None:
        velocity_xy = (robot_pos[:2] - self.prev_pos[:2]) / self.dt
        velocity_mag = np.linalg.norm(velocity_xy)
        
        if velocity_mag > 0.01:  # Only when moving
            velocity_unit = velocity_xy / velocity_mag
            # Cosine similarity: 1 = aligned, -1 = opposite, 0 = perpendicular
            alignment = float(np.dot(velocity_unit, self.goal_direction))
            # Reward alignment, allow exploration when stationary
            return 5.0 * alignment  # Weight: 5.0
    return 0.0

rotation_alignment_reward = calculate_rotation_alignment_reward(self)
```

---

### **Step 3: Increase Directional Rewards**

```python
# Boost directional velocity (PRIMARY movement signal)
directional_velocity_reward = 35.0 * float(np.tanh(5.0 * directional_velocity))  # Was 25.0 → +40%

# Increase perpendicular penalty (PUNISH drift harder)
perpendicular_penalty = -5.0 * abs(perpendicular_velocity)  # Was -3.0 → +67%
```

---

### **Step 4: Add Foot Lift Reward**

```python
# NEW: Reward lifting feet off ground (walking gait)
def calculate_foot_lift_reward(self, lifted_mask, ground_mask):
    """
    Reward for having some endpoints lifted (not all grounded = sliding).
    Encourages walking/stepping behavior.
    """
    num_lifted = np.sum(lifted_mask)
    num_grounded = np.sum(ground_mask)
    total_points = len(lifted_mask)
    
    # Ideal: 30-70% of points lifted (walking), not 0% (sliding) or 100% (falling)
    lift_ratio = num_lifted / total_points
    
    if 0.3 <= lift_ratio <= 0.7:
        # Reward balanced stance
        return 3.0
    elif lift_ratio < 0.1:
        # Penalize all-grounded (sliding)
        return -2.0
    elif lift_ratio > 0.9:
        # Penalize all-lifted (falling)
        return -1.0
    return 0.0

foot_lift_reward = calculate_foot_lift_reward(self, lifted_mask, ground_mask)
```

---

### **Step 5: Add Rolling Gait Reward (Optional)**

```python
# NEW: Reward x-axis rotation (rolling motion)
def calculate_rolling_reward(self):
    """
    Reward rotation about x-axis (forward roll) if moving forward.
    Alternative to walking gait.
    """
    imu_ang = self._get_IMU_angular_velocities()
    x_angular_vels = imu_ang[::3]  # X-axis components
    
    # Reward x-rotation if also moving forward
    if hasattr(self, 'prev_pos') and self.prev_pos is not None:
        velocity_xy = (robot_pos[:2] - self.prev_pos[:2]) / self.dt
        forward_velocity = float(np.dot(velocity_xy, self.goal_direction))
        
        if forward_velocity > 0.01:  # Only reward rolling when moving forward
            x_rotation_magnitude = np.mean(np.abs(x_angular_vels))
            return 4.0 * float(np.tanh(2.0 * x_rotation_magnitude))  # Weight: 4.0
    return 0.0

rolling_reward = calculate_rolling_reward(self)
```

---

### **Step 6: Reduce COM Progress Weight**

```python
# Reduce direction-agnostic COM reward
com_step_progress_weight = 10.0  # Was 25.0 → reduce by 60%
# Still rewards movement but less dominant
```

---

### **Step 7: Updated Reward Calculation**

```python
reward_raw = (
    endpoint_height_reward                                    # ~5-10
    + 0.0 * grounded_centroid_xy_reward                      # DISABLED (was 2.0)
    + lifted_centroid_xy_reward_weight * lifted_centroid_xy_reward  # Keep 1.0
    + 10.0 * com_step_progress                                # Reduced from 25.0
    + 6.0 * lifted_swing_reward                               # Keep
    + 35.0 * directional_velocity_reward                      # Boosted from 25.0
    + perpendicular_penalty                                   # -5.0 (was -3.0)
    + rotation_alignment_reward                               # NEW: 5.0 weight
    + foot_lift_reward                                        # NEW: 3.0 weight
    + rolling_reward                                          # NEW: 4.0 weight
    + action_smooth_penalty                                   # Keep -0.2
    + stall_penalty                                           # Keep
    + resume_bonus                                            # Keep
    + lift_dwell_penalty                                      # Keep
    + hover_weight * hover_term                               # Keep
    + cumulative_distance_bonus                               # Keep
)
```

---

## 📈 New Reward Balance

### **Movement Rewards (Directional):**
- Directional velocity: **35.0** (max, primary signal)
- Rotation alignment: **5.0** (new, encourages facing goal)
- Foot lift: **3.0** (new, encourages walking)
- Rolling: **4.0** (new, alternative gait)
- COM progress: **10.0** (reduced, backup signal)
- Lifted centroid: **1.0** (keep, general movement)
- **Total directional: ~58**

### **Penalties (Anti-drift/Anti-slide):**
- Perpendicular: **-5.0** (increased from -3.0)
- Action smooth: **-3.0** max (keep)
- Stall: **-0.5** max (keep)
- Foot lift (all grounded): **-2.0** (new, anti-sliding)
- **Total penalties: ~-10.5**

### **Ratio: 5.5:1** (directional focus with gait shaping)

---

## 🚀 Training Configuration Changes

### Keep Current Settings:
```python
learning_rate: 1.5e-4  # Good choice given high critic loss
timesteps: 75_000      # Sufficient for validation
eval_freq: 5000        # Good granularity
n_eval_episodes: 10    # Good statistics
```

### Expected Results After Changes:
- **Eval reward at 75k:** Target +1,000 to +2,000 (vs current +547)
- **Rollout reward:** Break positive by 50k
- **Critic loss:** Stabilize under 600 (simpler reward structure)
- **Visual behavior:** 
  - Robot rotates to face goal first
  - Lifts feet to walk OR rolls forward
  - Minimal sliding/gliding
  - Follows goal direction consistently

---

## 📋 Implementation Checklist

### Quick Changes (No New Code):
- [ ] Set `grounded_centroid_xy_reward_weight = 0.0`
- [ ] Increase `directional_velocity_reward` weight to 35.0
- [ ] Increase `perpendicular_penalty` to -5.0
- [ ] Reduce `com_step_progress` weight to 10.0

### New Reward Terms (Requires Code):
- [ ] Add `rotation_alignment_reward` function
- [ ] Add `foot_lift_reward` function  
- [ ] Add `rolling_reward` function (optional, can add later)
- [ ] Update `reward_raw` calculation to include new terms
- [ ] Update `info` dict for debugging

### Testing:
- [ ] Run 75k training as SAC_41
- [ ] Visual test at 50k and 75k steps
- [ ] Compare TensorBoard metrics to SAC_40
- [ ] If successful, train to 500k as SAC_42

---

## 🎯 Expected Timeline

- **Now**: Implement changes (30 min)
- **+1.5 hrs**: Complete 75k training (SAC_41)
- **Visual test**: Check directional control and gait quality
- **If good**: Launch 500k training (~6 hours)
- **Total to production model**: ~8 hours

---

## 🔍 Key Insight

**The core problem:** `grounded_centroid_xy_reward` was **rewarding the exact behavior you wanted to eliminate** (sliding). This created a fundamental conflict:

- Sliding = high `grounded_centroid_xy_reward` ✅
- Walking = low/zero `grounded_centroid_xy_reward` ❌

By removing this and adding explicit walking/rolling rewards, you align the reward structure with your actual goals.

---

## ⚠️ Important Notes

1. **Rotation alignment** needs robot orientation data - may need to extract from quaternions in `qpos` or IMU
2. **Foot lift reward** percentages (30-70%) are initial guesses - tune based on visual testing
3. **Rolling reward** is optional - can disable if you prefer pure walking
4. If critic loss still high after these changes, consider:
   - Further reducing reward scale (clip to ±30 instead of ±40)
   - Simplifying by removing cumulative distance bonus
   - Increasing batch size to 4096

Would you like me to implement these changes now?
