# SAC_34 Configuration Restored

## Changes Applied

Updated `gpu_pretraining_SAC.py` config `rtx2080ti_32gb_efficient` to **exactly match SAC_34** hyperparameters.

---

## SAC_34 Configuration (Proven Baseline)

### Hyperparameters
```python
learning_rate: 0.0002        # 2e-4 (was 3e-4)
batch_size: 2048             # unchanged
gamma: 0.99                  # (was 0.999)
ent_coef: 0.05               # fixed (was 0.1)
buffer_size: 500,000         # unchanged
tau: 0.005                   # target network update rate
target_entropy: -12.0        # entropy target
```

### Network Architecture
```python
Actor Network:
  Input: 98 (observation space)
  Hidden: [512, 512, 256] with ReLU
  Output: 12 (action space)

Critic Network:
  Input: 110 (obs + action: 98 + 12)
  Hidden: [512, 512, 256] with ReLU  
  Output: 1 (Q-value)
```

### Training Settings
```python
learning_starts: 5000
train_freq: (4, "step")
gradient_steps: 4
```

---

## Key Differences from Previous Config

| Parameter | Previous (SAC_33) | SAC_34 (Restored) | Impact |
|-----------|-------------------|-------------------|--------|
| **Learning Rate** | 3e-4 | **2e-4** | More stable, slower learning |
| **Gamma** | 0.999 | **0.99** | Less emphasis on future rewards |
| **Entropy Coef** | 0.1 | **0.05** | Less exploration, more exploitation |
| **Network** | [2048, 1024, 512, 256] | **[512, 512, 256]** | Smaller, faster, proven architecture |
| **Optimizer eps** | 1e-5 | *default* | Standard Adam settings |

---

## Why These Settings Matter

### Learning Rate (2e-4 vs 3e-4)
- **Lower = More stable critic learning**
- SAC_34 didn't have critic loss explosion
- SAC_33 may have learned too fast and overfit

### Gamma (0.99 vs 0.999)
- **0.99 = Less long-term planning**
- For locomotion, immediate rewards (step-by-step movement) matter more
- 0.999 can cause instability in continuous tasks

### Entropy Coefficient (0.05 vs 0.1)
- **0.05 = Balanced exploration/exploitation**
- 0.1 may have kept policy too random
- SAC_34 found good balance for this task

### Network Size ([512,512,256] vs [2048,1024,512,256])
- **Smaller = Faster training, less overfitting**
- 512-512-256 is proven to work for this obs/action space
- Larger networks can overfit on limited replay buffer

---

## Expected Impact on Training

### SAC_33 Behavior (with old config):
- Plateaued at +719 reward after 130k steps
- Large network may have overfit
- High entropy (0.1) kept policy too exploratory
- High gamma (0.999) may have destabilized learning

### Expected with SAC_34 Config:
- ✅ More stable learning curve (like SAC_34's smooth ascent)
- ✅ Better convergence (proven to reach +5000)
- ✅ Faster training (smaller network)
- ⚠️ May still need reward weight adjustments for non-gliding locomotion

---

## Current Reward Weights (Latest)

These are **independent** of SAC hyperparameters and still active:

```python
# Movement rewards
lifted_centroid_xy: 1.0              # Reduced for directional control
grounded_centroid_xy: 2.0            # Reduced for directional control
com_step_progress: 18.0              # Moderate
lifted_swing_reward: 6.0             # Moderate
directional_velocity: 20.0 max       # Primary directional signal
cumulative_distance_bonus: 2.0 max   # Small bonus

# Penalties
action_smooth_penalty: -0.3          # Anti-gliding
perpendicular_penalty: -3.0          # Anti-drift
hover_penalty: -3.0 (conditional)    # Anti-gliding
```

---

## Training Recommendation

### Option 1: Test SAC_34 Config with Current Rewards
```powershell
# Quick validation (100k steps)
python gpu_pretraining_SAC.py --total-timesteps 100000
```

**Expected**: Better learning stability, but may still plateau due to conservative reward weights.

### Option 2: SAC_34 Config + Boosted Movement Rewards
Apply the previously recommended reward increases:

```python
# In tensegrity_mjc_simulation.py
com_step_progress: 18.0 → 25.0
directional_velocity: 20.0 → 25.0
action_smooth_penalty: -0.3 → -0.2
```

Then train:
```powershell
python gpu_pretraining_SAC.py --total-timesteps 500000
```

**Expected**: Similar learning to SAC_34 (+4000-5000 reward) but without gliding.

---

## Verification

To confirm config is correct, after training starts check the log output:

```
SAC MODEL CONFIGURATION:
----------------------------------------
   Config: rtx2080ti_32gb_efficient
   Learning rate: 0.0002        ← Should match
   Batch size: 2048             ← Should match
   Network architecture: [512, 512, 256]  ← Should match
```

---

## Summary

✅ **Config restored to SAC_34 proven settings**  
✅ **Network architecture matches exactly**  
✅ **All hyperparameters verified from saved model**  
🎯 **Ready for training with stable, proven configuration**

**Next step**: Decide on Option 1 (test current rewards) or Option 2 (boost movement rewards) and start training.
