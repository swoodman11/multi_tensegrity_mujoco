# Phase 2 Weight Adjustment - Encouraging Movement

## SAC_36 Results (25k steps)

✅ **Success**: Gliding stopped!  
⚠️ **Issue**: Robot too conservative, not moving much  
📊 **Reward**: -4584 → -3363 (improving but still very negative)  
🎯 **Goal**: Increase movement while maintaining anti-gliding

---

## Changes Applied (Phase 2)

### Movement Rewards INCREASED:
- `lifted_centroid_xy`: 3.0 → **4.0** (+33%)
- `grounded_centroid_xy`: 8.0 → **11.0** (+37%)
- `com_step_progress`: 15.0 → **18.0** (+20%)
- `lifted_swing_reward`: 5.0 → **6.0** (+20%)

### Anti-Glide Penalties UNCHANGED:
- `action_smooth_penalty`: **-0.3** (keeping strong)
- `action_smooth_penalty cap`: **-5.0** (keeping strong)

### New Balance:
- **Movement rewards**: ~54 total
- **Anti-glide penalties**: ~13 total
- **Ratio**: 4:1 (sweet spot between original 8:1 and Phase 1's 3:1)

---

## Weight Evolution Summary

| Component | Original | Phase 1 | Phase 2 | Change Rationale |
|-----------|----------|---------|---------|------------------|
| lifted_centroid | 5.0 | 3.0 | **4.0** | Moderate recovery |
| grounded_centroid | 14.0 | 8.0 | **11.0** | Moderate recovery |
| com_step_progress | 24.0 | 15.0 | **18.0** | Moderate recovery |
| lifted_swing | 7.0 | 5.0 | **6.0** | Moderate recovery |
| action_smooth | -0.05 | -0.3 | **-0.3** | Keep strong penalty |
| action_smooth cap | -1.0 | -5.0 | **-5.0** | Keep strong penalty |

---

## Training Plan

### Run Extended Training (100k steps)

```powershell
python gpu_pretraining_SAC.py --total-timesteps 100000
```

**Duration**: ~1.5 hours (based on SAC_36's 23 FPS)

### Monitor Every 25k Steps

**Key metrics to watch:**

1. **`rollout/ep_rew_mean`**:
   - 25k (baseline): -3363
   - 50k target: -2000 to -1000
   - 75k target: -500 to +500
   - 100k target: +1000 to +2500
   
2. **`rollout/ep_len_mean`**:
   - Should stay: 400-500 (healthy)
   - Warning if: < 300 (policy struggling)
   
3. **`action_smooth_penalty`** (custom scalars):
   - Should remain: -1 to -3 (penalties activating)
   - Warning if: near 0 (no control changes → potential gliding return)

### Visual Test Checkpoints

```powershell
# At 50k steps
python test_trained_model_SAC.py --test-directions

# At 100k steps  
python test_trained_model_SAC.py --test-directions
```

**What to look for:**
- ✅ Active cable actuation visible
- ✅ Endpoint lifting and planting cycles
- ✅ Directional movement in goal direction
- ❌ Smooth sliding (gliding returned)
- ❌ No movement (still too conservative)

---

## Expected Trajectory

### Optimistic (Best Case)
```
Step    Reward   Status
25k:    -3363    Starting point (SAC_36)
50k:    -1000    Rapid improvement
75k:    +500     Positive territory!
100k:   +2000    Good performance
→ Continue to 500k
```

### Realistic (Most Likely)
```
Step    Reward   Status
25k:    -3363    Starting point
50k:    -1800    Gradual improvement
75k:    -800     Still learning
100k:   +500     Breaking positive
→ Continue to 500k, expect +2500-3500 final
```

### Pessimistic (Needs Adjustment)
```
Step    Reward   Status
25k:    -3363    Starting point
50k:    -3000    Minimal improvement
75k:    -2800    Stuck in local minimum
100k:   -2500    Not working
→ Consider Option B (smaller weights) or reduce action penalty
```

---

## Success Criteria (100k steps)

For proceeding to full 500k training:

- [ ] `rollout/ep_rew_mean` > 0 (positive rewards)
- [ ] `rollout/ep_len_mean` > 400 (stable episodes)
- [ ] Visual test shows active locomotion (no gliding)
- [ ] Directional control maintained (moves toward goal)
- [ ] Training stable (no crashes, reasonable FPS)

---

## Warning Signs & Actions

### Gliding Returns (50k-75k steps)

**Indicators:**
- Reward jumps too fast (> +3000 in 25k steps)
- Visual test shows sliding
- action_smooth_penalty near 0

**Action:**
1. Stop training
2. Reduce weights back to Phase 1 levels
3. Instead reduce action_smooth_penalty: -0.3 → -0.2
4. Resume training

### Stuck at Negative Rewards (75k-100k)

**Indicators:**
- Reward still < -1500 at 100k
- No improvement trend
- Visual test shows minimal movement

**Action:**
1. Apply Option B+ (larger weight increase):
   ```python
   grounded_centroid: 11.0 → 13.0
   com_step_progress: 18.0 → 21.0
   ```
2. OR reduce action_smooth_penalty: -0.3 → -0.15
3. Continue training

### Policy Collapse

**Indicators:**
- Episode length drops < 200
- Reward crashes to < -5000
- Robot falls or stops functioning

**Action:**
1. Stop immediately
2. Revert all weight changes
3. Investigate cause (may be unrelated to weights)

---

## Comparison to Baseline

### SAC_34 (Original, with gliding)
- 500k steps → +5000 reward
- Heavy gliding behavior
- Good directional control

### Phase 2 Goal (No gliding)
- 500k steps → +2500-4000 reward (estimate)
- Active cable-driven locomotion
- Good directional control maintained

**Expected trade-off**: Slightly lower max reward for better quality locomotion.

---

## Quick Commands

```powershell
# Start training
python gpu_pretraining_SAC.py --total-timesteps 100000

# Monitor TensorBoard
tensorboard --logdir=./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient

# Check progress (every 25k)
python analyze_tensorboard.py ./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient/SAC_37

# Visual test (at 50k and 100k)
python test_trained_model_SAC.py --test-directions
```

---

## Timeline

- **Now**: Start 100k training
- **+40 min**: Check 50k results, visual test
- **+1.5 hrs**: Check 100k results, visual test, decide on 500k
- **+6.5 hrs** (if proceeding): Complete 500k training
- **Total**: ~8 hours to fully trained model (if all goes well)

---

**Status**: ✅ Phase 2 weights applied, ready for 100k training run! 🚀

The balance should now encourage movement while keeping anti-glide penalties strong enough to prevent sliding.
