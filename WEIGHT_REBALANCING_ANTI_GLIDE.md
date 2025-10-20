# Weight Rebalancing Strategy - Anti-Glide

## Current Problem
Robot glides/slides instead of actively using cables for locomotion.

## Current Reward Weights Analysis

### Movement Incentives (encourage displacement)
- `grounded_centroid_xy_reward_weight`: **14.0** ← HIGH
- `com_step_progress`: **24.0** ← VERY HIGH
- `lifted_centroid_xy_reward`: **5.0**
- `directional_velocity_reward`: **15.0** (max from tanh scaling)
- `lifted_swing_reward`: **7.0**
- **Total potential movement reward: ~65**

### Anti-Glide Mechanisms (discourage passive movement)
- `hover_weight`: **3.0** (conditionally applied)
- `action_smooth_penalty`: **-0.05** weight, capped at **-1.0** ← VERY WEAK
- `stall_penalty`: **-4.0** max (but only for low speed)
- **Total anti-glide penalty: ~8**

### Problem
**Ratio: 65:8 ≈ 8:1 in favor of movement** → Robot learns to maximize displacement with minimal effort (gliding)

---

## Proposed Weight Adjustments

### Strategy: Reduce movement incentives, strengthen anti-glide penalties

```python
# CURRENT → PROPOSED

# Reduce excessive movement rewards
lifted_centroid_xy_reward_weight = 5.0 → 3.0       # -40% reduction
grounded_centroid_xy_reward_weight = 14.0 → 8.0    # -43% reduction  
com_step_progress weight: 24.0 → 15.0              # -38% reduction
lifted_swing_reward weight: 7.0 → 5.0              # -29% reduction

# Strengthen anti-glide penalties
action_smooth_penalty weight: -0.05 → -0.3         # 6x stronger
action_smooth_penalty cap: -1.0 → -5.0             # 5x larger penalty
hover_weight: 3.0 → 6.0                            # 2x stronger
stall_penalty multiplier: -4.0 → -6.0              # 1.5x stronger
```

### New Balance
- Movement rewards: ~38 max (down from ~65)
- Anti-glide penalties: ~17 max (up from ~8)
- **New ratio: 38:17 ≈ 2.2:1** ← Much more balanced

---

## Implementation

### Change 1: Reduce Movement Reward Weights
```python
# Line ~741-742
lifted_centroid_xy_reward_weight = 3.0    # was 5.0
grounded_centroid_xy_reward_weight = 8.0  # was 14.0
```

### Change 2: Reduce COM and Swing Weights
```python
# Line ~758-763
reward_raw = (
    endpoint_height_reward
    + lifted_centroid_xy_reward_weight * lifted_centroid_xy_reward
    + grounded_centroid_xy_reward_weight * grounded_centroid_xy_reward
    + 15.0 * com_step_progress  # was 24.0
    + 5.0 * lifted_swing_reward  # was 7.0
    + stall_penalty
    # ... rest unchanged
)
```

### Change 3: Strengthen Action Smoothing Penalty
```python
# Line ~748-756
action_smooth_penalty = 0.0
if controls is not None:
    if hasattr(self, 'prev_controls') and self.prev_controls is not None:
        try:
            action_smooth_penalty = -0.3 * float(np.sum(np.abs(controls - self.prev_controls)))  # was -0.05
        except Exception:
            action_smooth_penalty = 0.0
    self.prev_controls = controls.copy()
action_smooth_penalty = max(action_smooth_penalty, -5.0)  # was -1.0
```

### Change 4: Increase Hover Penalty
```python
# Line ~745
hover_weight = 0.0 if (self.stall_streak < 5 or lifted_count >= 2) else 6.0  # was 3.0
```

### Change 5: Strengthen Stall Penalty (Optional)
```python
# Line ~650 (in stall_penalty calculation)
stall_penalty = -6.0 * stall_ratio * (0.5 + 0.5 * ramp)  # was -4.0
```

---

## Testing Protocol

### Phase 1: Conservative (Start Here)
Apply Changes 1, 2, 3 only. Train 25k steps.

**Expected:**
- Some reduction in gliding
- Policy should remain stable
- Reward may decrease initially (acceptable)

### Phase 2: Moderate
If still gliding, add Change 4 (hover penalty). Train 25k steps.

**Expected:**
- Further reduction in gliding  
- May see more endpoint lifting behavior
- Episode length should remain stable

### Phase 3: Aggressive  
If still gliding, add Change 5 (stall penalty). Train 25k steps.

**Expected:**
- Strong anti-gliding pressure
- Watch for policy collapse (ep_len < 200)
- If collapse occurs, revert Change 5

---

## Success Metrics

### TensorBoard (after 25k steps)
- `rollout/ep_rew_mean`: 1000-3000 (may decrease from 5000, acceptable)
- `rollout/ep_len_mean`: 400-800 (should remain stable)
- `action_smooth_penalty` in info logs: -2 to -4 (activating)

### Visual Test
- Cable length changes visible during locomotion
- Less "ice skating" smooth motion
- More variable velocity (gait-like patterns)

### Warning Signs
- `rollout/ep_len_mean` < 200 → penalties too harsh
- `rollout/ep_rew_mean` < -500 → excessive penalties
- Robot stops moving entirely → reduce penalties

---

## Tuning Ladder

If results after Phase 1 are:

**Still heavy gliding:**
→ Proceed to Phase 2 (add hover penalty increase)

**Partial improvement (50% less gliding):**
→ Train to 100k steps with current weights, reassess

**Minimal gliding but slower movement:**
→ SUCCESS! Continue to 500k training

**Policy collapsed:**
→ Revert to smaller changes:
```python
grounded_centroid_xy_reward_weight = 11.0  # gentler reduction
com_step_progress weight: 20.0  # gentler reduction
action_smooth_penalty weight: -0.15  # gentler increase
```

---

## Implementation Order

1. ✅ Apply Phase 1 changes (Changes 1, 2, 3)
2. ✅ Train 25k steps (~20 min)
3. ✅ Visual test + check TensorBoard
4. ⏸️ If needed, apply Phase 2 (Change 4)
5. ⏸️ Train another 25k steps
6. ⏸️ If needed, apply Phase 3 (Change 5)
7. ✅ Once working, full 500k training

---

## Quick Reference: Line Numbers

| Change | File Location | Line # |
|--------|---------------|--------|
| Centroid weights | tensegrity_mjc_simulation.py | 741-742 |
| COM/swing weights | tensegrity_mjc_simulation.py | 761-762 |
| Action smooth weight | tensegrity_mjc_simulation.py | 752 |
| Action smooth cap | tensegrity_mjc_simulation.py | 756 |
| Hover weight | tensegrity_mjc_simulation.py | 745 |
| Stall penalty | tensegrity_mjc_simulation.py | ~650 |

---

## Rationale

### Why Reduce Movement Rewards?
- Robot currently gets huge reward for ANY displacement
- Gliding is the easiest way to achieve displacement
- Reducing makes movement less dominant in total reward

### Why Strengthen Action Smoothing Penalty?
- Gliding involves minimal control changes (static posture)
- Stronger penalty punishes "do nothing" strategy
- Cap increase allows meaningful penalty accumulation

### Why Increase Hover Penalty?
- Gliding often keeps robot in hover zone (just above ground)
- Stronger penalty pushes toward ground contact changes
- Conditional application prevents over-penalization

### Why NOT Add New Terms?
- Current reward structure already has anti-glide mechanisms
- They're just too weak relative to movement rewards
- Rebalancing is cleaner than adding complexity

---

## Expected Behavior Change

### Before (Gliding)
```
Step reward breakdown:
  grounded_centroid: +8.2
  com_step_progress: +18.5
  directional_velocity: +12.1
  action_smooth_penalty: -0.4
  hover_term: 0.0
  → Total: ~38 (high reward, minimal effort)
```

### After (Active Locomotion)
```
Step reward breakdown:
  grounded_centroid: +4.0  (reduced)
  com_step_progress: +9.0  (reduced)
  directional_velocity: +10.5
  action_smooth_penalty: -2.8  (stronger)
  hover_term: -3.0  (activating)
  → Total: ~18 (moderate reward, requires effort)
```

The robot must now work harder (cable actuation) to achieve similar total reward.

---

## Rollback Plan

If after all phases, policy is broken:

```python
# Minimal intervention (safest)
grounded_centroid_xy_reward_weight = 12.0  # small reduction from 14.0
com_step_progress weight: 22.0             # small reduction from 24.0
action_smooth_penalty weight: -0.1         # small increase from -0.05
action_smooth_penalty cap: -2.0            # small increase from -1.0

# This gives ~3:1 ratio instead of 8:1, likely enough
```

---

**Recommendation: Start with Phase 1 (conservative changes), train 25k steps, then reassess.**
