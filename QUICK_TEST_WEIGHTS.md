# Quick Start - Weight Rebalancing Test

## Changes Applied (Phase 1 - Conservative)

### Movement Rewards REDUCED:
- `lifted_centroid_xy_reward_weight`: 5.0 → **3.0** (-40%)
- `grounded_centroid_xy_reward_weight`: 14.0 → **8.0** (-43%)
- `com_step_progress` weight: 24.0 → **15.0** (-38%)
- `lifted_swing_reward` weight: 7.0 → **5.0** (-29%)

### Anti-Glide Penalties INCREASED:
- `action_smooth_penalty` weight: -0.05 → **-0.3** (6x stronger)
- `action_smooth_penalty` cap: -1.0 → **-5.0** (5x larger)

### Domain Randomization:
- 🚫 **Friction randomization DISABLED** (fixed at XML defaults for cleaner testing)
- ✅ **Goal direction randomization ACTIVE** (0-360° each reset)

### Net Effect:
- **Before**: Movement rewards ~65, penalties ~8 (ratio 8:1)
- **After**: Movement rewards ~38, penalties ~13 (ratio 3:1)
- **Friction**: Consistent across all episodes (easier to evaluate gliding)

---

## Test Now (20 minutes)

```powershell
# Start validation training
python gpu_pretraining_SAC.py --total-timesteps 25000

# In separate terminal, monitor TensorBoard
tensorboard --logdir=./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient
```

---

## What to Watch

### TensorBoard Metrics (every 5k steps)

**Policy Health:**
- `rollout/ep_len_mean`: Should stay **400-800** (healthy)
  - ⚠️ If < 200 → policy collapsed, revert changes
- `rollout/ep_rew_mean`: May decrease to **1000-3000** (acceptable)
  - ⚠️ If < -500 → penalties too harsh

**Anti-Glide Activation:**
- `action_smooth_penalty` (in custom scalars): Should be **-2 to -4**
  - If 0 → robot not changing actions (good for anti-glide!)
  - If -5 → hitting cap (very active, could be good or jittery)

### Visual Test (after 25k)

```powershell
python test_trained_model_SAC.py --test-directions
```

**Look for:**
- ✅ Less smooth/constant velocity (more variable)
- ✅ Visible cable length changes
- ✅ Still moves in correct directions
- ❌ Robot frozen/not moving → penalties too strong

---

## Decision Tree After 25k Steps

```
Visual test shows:
│
├─ Heavy gliding still present?
│  → Apply Phase 2 (see WEIGHT_REBALANCING_ANTI_GLIDE.md)
│  → Increase hover_weight: 3.0 → 6.0
│
├─ Some gliding, some improvement?
│  → Continue to 100k steps with current weights
│  → Reassess at 100k
│
├─ Minimal/no gliding?
│  → SUCCESS! Run full 500k training
│
└─ Policy collapsed (ep_len < 200)?
   → Revert to gentler changes:
      grounded_centroid: 14.0 → 11.0
      com_step_progress: 24.0 → 20.0
      action_smooth: -0.05 → -0.1
      action_smooth cap: -1.0 → -2.0
```

---

## Expected Results

### Optimistic (Best Case)
- Episode length: 600-800
- Reward: 2000-3000 (down from 5000, but with better quality movement)
- Visual: Clear reduction in gliding, more active locomotion
- Action: Continue to 500k training

### Realistic (Most Likely)
- Episode length: 500-700
- Reward: 1500-2500
- Visual: Moderate improvement, some gliding remains
- Action: Apply Phase 2 (increase hover penalty)

### Pessimistic (Worst Case)
- Episode length: < 300
- Reward: < 500
- Visual: Robot barely moves or collapses
- Action: Revert to gentler weight changes

---

## Quick Commands

```powershell
# Start training
python gpu_pretraining_SAC.py --total-timesteps 25000

# Monitor TensorBoard
tensorboard --logdir=./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient

# Test after training
python test_trained_model_SAC.py --test-directions

# Find latest model
ls models/gpu/ | sort -Property LastWriteTime -Descending | select -First 1
```

---

## Rollback if Needed

If policy collapses, revert changes:

```powershell
# Use git to undo changes
git checkout HEAD -- mujoco_physics_engine/tensegrity_mjc_simulation.py

# Then apply gentler changes manually (see Rollback Plan in main doc)
```

---

## Next Steps Options

### Option A: Working Well → Full Training
```powershell
python gpu_pretraining_SAC.py --total-timesteps 500000
# Wait ~6.5 hours
```

### Option B: Needs More → Phase 2
Edit `tensegrity_mjc_simulation.py` line 745:
```python
hover_weight = 0.0 if (self.stall_streak < 5 or lifted_count >= 2) else 6.0  # was 3.0
```
Then train another 25k steps.

### Option C: Too Harsh → Gentler Settings
Edit weights to intermediate values:
```python
grounded_centroid_xy_reward_weight = 11.0  # between 8.0 and 14.0
com_step_progress weight: 20.0             # between 15.0 and 24.0
action_smooth_penalty weight: -0.15        # between -0.3 and -0.05
action_smooth_penalty cap: -2.0            # between -5.0 and -1.0
```

---

## Success Indicators

After 25k steps, you've succeeded if:

1. ✅ `rollout/ep_len_mean` > 400 (policy stable)
2. ✅ Visual test shows less gliding than before
3. ✅ Robot still moves directionally
4. ✅ `action_smooth_penalty` shows activation (-1 to -4)

Even if some gliding remains, this indicates the direction is correct.

---

**START TRAINING NOW** and report results after 25k steps! 🚀

Timeline: ~20 minutes for validation run.
