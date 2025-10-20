# Test Mode Configuration

## Current Settings (For Testing)

**Test run: 25k timesteps, 50 demo cycles**

This configuration will complete in ~5-10 minutes and verify:
- ✅ TensorBoard logging works
- ✅ EvalCallback executes without hanging
- ✅ Episode truncation works correctly
- ✅ Model saving works

### Active Test Parameters

| Parameter | Test Value | Production Value | File Location |
|-----------|------------|------------------|---------------|
| `timesteps` | 25,000 | 2,000,000 | `gpu_pretraining_SAC.py` line ~603 |
| `cycles` | 50 | 250-500 | `gpu_pretraining_SAC.py` line ~604 |
| `eval_freq` | 5,000 | 20,000 | `gpu_pretraining_SAC.py` line ~394 |
| `n_eval_episodes` | 2 | 3 | `gpu_pretraining_SAC.py` line ~395 |

## Test Run Expected Behavior

### Timeline (25k timesteps)
- **0-5 min**: Demo cycle generation (50 cycles × 32 steps)
- **5-8 min**: Behavioral cloning (250 epochs)
- **8-13 min**: RL training (25k steps with 5 evaluations)

### What to Check During Test

1. **TensorBoard (http://localhost:6006):**
   - Should show data appearing in `rollout/ep_rew_mean`
   - Check `train/actor_loss` and `train/critic_loss` are not NaN
   - Verify `time/fps` is reasonable (100-500 fps)

2. **Terminal Output:**
   - Watch for evaluation messages every 5k steps
   - Should see: "Eval num_timesteps=5000, episode_reward=..."
   - No hanging or timeouts

3. **Files Created:**
   - `./logs/best_model_rtx2080ti_32gb_efficient/best_model.zip`
   - `./logs/evals_rtx2080ti_32gb_efficient/evaluations.npz`
   - `./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient/SAC_XX/events.out.tfevents.*`
   - `./models/gpu/sac_gpu_pretraining_rtx2080ti_32gb_efficient_TIMESTAMP.zip`

## After Successful Test

### 1. Change Parameters Back to Production Values

Open `gpu_pretraining_SAC.py` and search for "TEST:" comments:

**Line ~603-607:**
```python
timesteps = 2_000_000  # Was: 25_000
cycles = 500           # Was: 50 (or use 250 for faster)
```

**Line ~394-398:**
```python
eval_freq=20000,       # Was: 5000
n_eval_episodes=3,     # Was: 2
```

### 2. Start Production Training

```powershell
# Start training (will take ~4-8 hours for 2M steps)
python gpu_pretraining_SAC.py

# In separate terminal - start TensorBoard
tensorboard --logdir=./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient --reload_interval=30
```

### 3. Monitor Progress

Check TensorBoard every hour at these milestones:

| Timesteps | Expected Avg Reward | Action |
|-----------|---------------------|--------|
| 100k | -800 to -400 | Continue if decreasing |
| 500k | -400 to -100 | Test directional performance |
| 1M | -100 to +50 | Should see multi-directional movement |
| 1.5M | +50 to +100 | Directional bias should reduce |
| 2M | +100 to +200 | Final evaluation |

### 4. Test Checkpoints

```powershell
# Test best model at any time during training
python test_trained_model_SAC.py --model logs/best_model_rtx2080ti_32gb_efficient/best_model --test-directions --no-vis

# Test final saved model
python test_trained_model_SAC.py --model models/gpu/sac_gpu_pretraining_rtx2080ti_32gb_efficient_TIMESTAMP --test-directions --no-vis
```

## Quick Command Reference

```powershell
# Run test (current settings)
python gpu_pretraining_SAC.py

# Start TensorBoard
tensorboard --logdir=./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient

# Test a model
python test_trained_model_SAC.py --model PATH_TO_MODEL --test-directions --no-vis

# Clean up old test runs (optional)
Remove-Item -Recurse ./logs/best_model_rtx2080ti_32gb_efficient
Remove-Item -Recurse ./logs/evals_rtx2080ti_32gb_efficient
Remove-Item -Recurse ./sac_tensegrity_tensorboard_rtx2080ti_32gb_efficient
```

## Troubleshooting Test Run

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| No TensorBoard data | Event files not being written | Check terminal for errors |
| Evaluation hangs | Episode not truncating | Verify `max_episode_steps=500` in env |
| GPU OOM | Batch size too large | Reduce to 512 in config |
| Low FPS (<50) | Visualization enabled | Ensure `visualize=False` |
| Training very slow | Too many demo cycles | Use 50-100 for testing |

---

**Remember:** This is TEST MODE - change parameters back before production run!
