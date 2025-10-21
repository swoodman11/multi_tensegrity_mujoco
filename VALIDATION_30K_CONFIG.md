# Validation Run Configuration - 30k Steps

## Changes Applied

Updated training script for quick validation to verify SAC_34 config + new rewards match baseline performance.

---

## Training Parameters

```python
# In gpu_pretraining_SAC.py

# Total training steps
timesteps = 30_000  # Was 1_000_000

# Evaluation frequency  
eval_freq = 5000    # Was 10_000

# Number of evaluations
# 30k / 5k = 6 evaluation points
```

---

## Evaluation Schedule

```
Step 5k:   First checkpoint
Step 10k:  Second checkpoint
Step 15k:  Third checkpoint
Step 20k:  Fourth checkpoint
Step 25k:  Fifth checkpoint
Step 30k:  Final checkpoint
```

**Total**: 6 evaluation points over ~25 minutes

---

## Comparison to SAC_34 (First 30k)

### SAC_34 Performance (Baseline):
Based on the TensorBoard image, SAC_34 at ~25k-30k steps showed:
- `eval/mean_reward`: Approximately -1000 to 0 (still learning)
- `rollout/ep_rew_mean`: Similar range
- Smooth upward trajectory

### Expected with New Config:
- Similar or better reward progression
- Stable episode lengths (500)
- No critic loss explosion
- Smooth learning curve

---

## Success Criteria (30k steps)

### Must Have:
- [ ] `rollout/ep_rew_mean` > -1000 at 30k
- [ ] `train/critic_loss` < 300 (stable)
- [ ] No training crashes or OOM errors
- [ ] Episode length stable ~500

### Good Indicators:
- [ ] `rollout/ep_rew_mean` > -500 at 30k
- [ ] Reward trending upward consistently
- [ ] Actor loss stable magnitude (-500 to -1500)

### Excellent Performance:
- [ ] `rollout/ep_rew_mean` > 0 at 30k
- [ ] Clear improvement every 5k steps
- [ ] Visual test shows some movement

---

## Timeline

```
Start:    0 min     Launch training
Eval 1:   5 min     5k steps
Eval 2:   10 min    10k steps
Eval 3:   15 min    15k steps
Eval 4:   20 min    20k steps
Eval 5:   25 min    25k steps
Eval 6:   30 min    30k steps COMPLETE
```

**Total duration**: ~25-30 minutes

---

## What to Monitor

### TensorBoard (real-time):
```powershell
tensorboard --logdir=./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient
```

Watch:
- `rollout/ep_rew_mean` - should climb steadily
- `train/critic_loss` - should be < 300 and stable
- `train/actor_loss` - should stabilize in magnitude
- `rollout/ep_len_mean` - should stay ~500

### Console Output:
Look for:
- GPU memory stable (~8-10GB for RTX 2080 Ti)
- No warnings or errors
- FPS ~20-25 steps/second
- "Saving best model" messages

---

## Decision Tree After 30k

### If Excellent (reward > 0):
```
✅ Config validated!
→ Change timesteps to 500_000
→ Change eval_freq to 10_000
→ Run full training
```

### If Good (reward -500 to 0):
```
✅ On track, similar to SAC_34
→ Continue to 100k for more data
→ If still good, proceed to 500k
```

### If Acceptable (reward -1000 to -500):
```
⚠️ Slower than expected
→ Check if learning curve is rising
→ If yes: Continue to 100k
→ If flat: Increase movement rewards more
```

### If Poor (reward < -1000):
```
❌ Issue detected
→ Compare to SAC_33 (plateaued at -3363 at 25k)
→ Check critic loss (should be < 300)
→ May need to further boost rewards
```

---

## Quick Commands

### Start Validation:
```powershell
python gpu_pretraining_SAC.py
```

### Monitor Progress:
```powershell
# Terminal 1: Training
python gpu_pretraining_SAC.py

# Terminal 2: TensorBoard
tensorboard --logdir=./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient --reload_interval=30

# Terminal 3: Check latest run
python analyze_tensorboard.py ./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient/SAC_<latest_number>
```

### After Completion:
```powershell
# Find the model
ls models/gpu/ | sort -Property LastWriteTime -Descending | select -First 1

# Visual test (optional for 30k, recommended)
python test_trained_model_SAC.py --model models/gpu/<latest_model> --test-directions
```

---

## Scaling to Full Training

If validation is successful, update these lines:

```python
# Line ~627 in gpu_pretraining_SAC.py
timesteps = 500_000  # Change from 30_000

# Line ~415 in gpu_pretraining_SAC.py  
eval_freq = 10_000   # Change from 5_000
```

This gives:
- 500k total steps (~6.5 hours)
- 50 evaluation points
- Same proven config, just longer

---

## Files Modified

1. ✅ `gpu_pretraining_SAC.py`
   - Line 415: `eval_freq = 5000` (was 10000)
   - Line 627: `timesteps = 30_000` (was 1_000_000)
   - Line 628-631: Updated print messages

---

## Current Configuration Summary

**SAC Hyperparameters**: ✅ SAC_34 matched
- Learning rate: 2e-4
- Network: [512, 512, 256]
- Batch size: 2048
- Gamma: 0.99
- Entropy: 0.05

**Reward Weights**: ✅ Optimized for non-gliding
- COM progress: 25.0
- Directional velocity: 25.0
- Centroid: 1.0 + 2.0 (low)
- Action smoothing: -0.2 (relaxed)

**Training**: ✅ Quick validation
- Steps: 30k
- Evals: Every 5k (6 total)
- Duration: ~25-30 min

---

**Status**: ✅ Ready for validation run!

**Command**:
```powershell
python gpu_pretraining_SAC.py
```

This will quickly verify if the new configuration performs as well as SAC_34 baseline. 🚀
