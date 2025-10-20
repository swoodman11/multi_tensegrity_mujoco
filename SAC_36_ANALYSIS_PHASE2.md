# SAC_36 Analysis - Phase 1 Results

## TensorBoard Analysis Summary

### Key Findings

**Reward Progression:**
- Step 5k: -4584
- Step 15k: -3540
- Step 25k: -3363
- **Trend**: ↑ Improving (+1221 over 25k steps)

**Episode Length:**
- Stable at **500 steps** (max episode length cap)
- Robot surviving but not moving much

**Actor/Critic:**
- Actor loss: -268 → -1041 (increasing magnitude, typical for SAC)
- Critic loss: 50 → 201 (increasing, learning value function)

**Visual Observation:**
- ✅ **Gliding STOPPED** - Success!
- ⚠️ **Low movement** - Robot too conservative

## Problem Diagnosis

### Success: Anti-Gliding Works
The negative rewards (-3363 avg) indicate penalties are working:
- Robot no longer exploiting friction
- Action smoothing penalty likely activating
- Height/contact dynamics engaging properly

### Issue: Over-Correction
We reduced movement rewards **too much**:

**Current weights:**
```python
lifted_centroid_xy: 3.0        # Was 5.0 (originally)
grounded_centroid_xy: 8.0      # Was 14.0 (originally)  
com_step_progress: 15.0        # Was 24.0 (originally)
lifted_swing_reward: 5.0       # Was 7.0 (originally)
directional_velocity: 15.0     # Unchanged (max from tanh)
```

**Result**: Total movement incentive ~46, but with strong penalties (~13), net is too conservative.

## Recommended Adjustment Strategy

### Option A: Moderate Increase (RECOMMENDED)
Increase movement rewards while keeping anti-glide penalties:

```python
lifted_centroid_xy_reward_weight = 4.0     # Was 3.0, increase 33%
grounded_centroid_xy_reward_weight = 11.0  # Was 8.0, increase 37%
com_step_progress weight: 18.0             # Was 15.0, increase 20%
lifted_swing_reward weight: 6.0            # Was 5.0, increase 20%

# Keep penalties at current strength:
action_smooth_penalty: -0.3                # Keep as is
action_smooth_penalty cap: -5.0            # Keep as is
```

**Rationale:**
- Movement rewards: 15.0 → 18.0 (COM) + 4.0 + 11.0 + 6.0 = ~54 total
- Penalties: ~13
- New ratio: **54:13 ≈ 4:1** (between original 8:1 and Phase 1's 3:1)
- Should encourage movement without re-enabling gliding

### Option B: Small Increase (Conservative)
Smaller bump if worried about gliding returning:

```python
lifted_centroid_xy_reward_weight = 3.5     # Was 3.0, increase 17%
grounded_centroid_xy_reward_weight = 10.0  # Was 8.0, increase 25%
com_step_progress weight: 17.0             # Was 15.0, increase 13%
lifted_swing_reward weight: 5.5            # Was 5.0, increase 10%
```

**Rationale:**
- More conservative adjustment
- Ratio: ~50:13 ≈ 3.8:1
- Less risk of gliding return

### Option C: Train Longer First
Continue with current weights for 100k-200k steps:

**Pros:**
- May naturally improve as exploration continues
- SAC's entropy bonus could help discover better policies
- Avoids hyperparameter thrashing

**Cons:**
- -3363 reward is quite negative (far from SAC_34's +5000)
- May be stuck in local minimum (too conservative policy)
- Wastes compute time if weights truly need adjustment

## Recommendation: Option A + Extended Training

**Phase 2 Plan:**

1. **Apply Option A weight increases** (moderate boost)
2. **Train 100k steps** (not just 25k)
3. **Evaluate at checkpoints**: 25k, 50k, 75k, 100k
4. **Visual test at 100k** to see quality of movement

**Expected outcomes:**
- Reward should improve: -3363 → 0 to +2000 range
- Movement should increase while staying non-gliding
- Episode length stays at 500 (healthy)

## Alternative: Reduce Action Smoothing Penalty

If increasing movement rewards brings back gliding, try:

```python
action_smooth_penalty weight: -0.3 → -0.2  # Reduce penalty slightly
action_smooth_penalty cap: -5.0 → -3.0     # Lower cap

# This allows more control variation without harsh penalty
```

## Decision Matrix

| Scenario | Action | Expected Result |
|----------|--------|-----------------|
| **Want safe, gradual improvement** | Option B (small increase) + 100k training | Slow but steady progress |
| **Want balanced approach** | Option A (moderate increase) + 100k training | ⭐ **RECOMMENDED** |
| **Very risk-averse** | Option C (no change) + 100k training | May stay stuck at -3k reward |
| **Aggressive** | Option A + reduce action penalty + 100k | Fast improvement, risk of gliding |

## Implementation: Option A (Recommended)

### Change 1: Increase Centroid Weights
```python
# Line ~741-742
lifted_centroid_xy_reward_weight = 4.0   # was 3.0
grounded_centroid_xy_reward_weight = 11.0  # was 8.0
```

### Change 2: Increase COM and Swing Weights  
```python
# Line ~758-759
+ 18.0 * com_step_progress  # was 15.0
+ 6.0 * lifted_swing_reward  # was 5.0
```

### Training Command
```powershell
# Run for 100k steps to see learning curve
python gpu_pretraining_SAC.py --total-timesteps 100000
```

## Monitoring Plan

### Every 25k steps, check:

**Good progress indicators:**
- Reward increasing: -3363 → -2000 → -500 → +1000
- Episode length stable at 500
- Actor loss magnitude stabilizing
- Critic loss converging (< 300)

**Warning signs:**
- Reward plateaus at negative values
- Episode length drops suddenly
- FPS drops significantly (computational issue)

**Gliding return indicators:**
- Reward jumps too quickly (> +3000 in 25k steps)
- Visual test shows sliding
- action_smooth_penalty stops activating (stays near 0)

## Success Criteria (100k steps)

- [ ] Reward > 0 (breaking even)
- [ ] Visual test shows active locomotion (not gliding)
- [ ] Directional control maintained
- [ ] Episode length 400-500
- [ ] No computational issues

If all met → **Full 500k training**

## Fallback Plan

If gliding returns at 50k-75k steps:

1. Stop training
2. Revert to Phase 1 weights
3. Instead reduce action_smooth_penalty to -0.2
4. Resume training

---

**Next Action**: Apply Option A changes and train 100k steps
