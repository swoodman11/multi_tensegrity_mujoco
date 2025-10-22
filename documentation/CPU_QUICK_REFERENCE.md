# CPU Training Quick Reference Card

## 🚀 Quick Commands

```bash
# Train on CPU (automatic configuration based on RAM)
python cpu_pretraining_SAC_single.py

# Test trained model (auto-detects CPU/GPU)
python test_trained_model_SAC_single.py

# Demo simulation (no RL, no GPU)
python run_single.py

# Generate gait patterns
python single_tensegrity_gait_generation.py

# Monitor training
tensorboard --logdir ./sac_single_cpu_tensorboard_cpu_medium/
```

## 📊 Automatic Configuration Selection

| Your RAM | Config Selected | Network Size | Timesteps | Expected Time |
|----------|----------------|--------------|-----------|---------------|
| 8-12 GB  | cpu_small      | [128, 128]   | 20,000    | 30-60 min     |
| 12-16 GB | cpu_medium     | [256, 256]   | 30,000    | 1-2 hours     |
| 16+ GB   | cpu_large      | [512,256,128]| 50,000    | 2-4 hours     |

## ✅ What Works Without GPU

| Script | Requires GPU? | Notes |
|--------|---------------|-------|
| `cpu_pretraining_SAC_single.py` | ❌ NO | CPU training script |
| `test_trained_model_SAC_single.py` | ❌ NO | Auto-detects device |
| `run_single.py` | ❌ NO | Pure MuJoCo simulation |
| `single_tensegrity_gait_generation.py` | ❌ NO | No RL involved |
| `gpu_pretraining_SAC_single.py` | ✅ YES | GPU-only version |

## 🎯 Typical Workflow

### First Time Setup
```bash
# 1. Generate gait files
python single_tensegrity_gait_generation.py

# 2. Test simulator works
python run_single.py

# 3. Train on CPU
python cpu_pretraining_SAC_single.py

# 4. Test trained model
python test_trained_model_SAC_single.py
```

### Testing a Gait
```bash
# Run gait for 5 cycles with visualization
python run_single.py dual_robot_first_6.json 5
python run_single.py optimal_rolling_gait.json 10
```

### After Training
```bash
# Test latest model
python test_trained_model_SAC_single.py

# Test specific model
python test_trained_model_SAC_single.py --model models/cpu/sac_single_cpu_pretraining_cpu_medium_20251020_143022

# Run 10 episodes without visualization
python test_trained_model_SAC_single.py --episodes 10 --no-viz
```

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Close other apps, use cpu_small config |
| Training too slow | Reduce timesteps in script (e.g., 10000) |
| Model not found | Check `ls models/cpu/` |
| No gait file | Run `python single_tensegrity_gait_generation.py` |
| Visualization issues | Use `--no-viz` flag for testing |

## 📁 Where Things Are Saved

```
models/
  cpu/                    # CPU-trained models saved here
    sac_single_cpu_pretraining_*.zip
  gpu/                    # GPU-trained models (if any)
    sac_single_gpu_pretraining_*.zip

logs/
  best_model_single_cpu_*/   # Best models during training
  evals_single_cpu_*/        # Evaluation results

sac_single_cpu_tensorboard_*/  # TensorBoard logs
```

## ⚡ Speed Comparison

- **GPU training**: 10-20 minutes (RTX 4090)
- **CPU training**: 30 minutes - 4 hours (depends on RAM/CPU)
- **Speed difference**: CPU is 10-50x slower than GPU
- **Model quality**: Same (given enough time)

## 💾 Disk Space Usage

- Model file: ~10-50 MB
- Replay buffer: ~50-200 MB (in RAM during training)
- TensorBoard logs: ~10-100 MB
- Total: ~100-500 MB per training run

## 🎮 Test Script Options

```bash
# All available options
python test_trained_model_SAC_single.py \
  --model path/to/model.zip \
  --config cpu_medium \
  --episodes 10 \
  --max-steps 1000 \
  --no-viz \
  --save-frames
```

## 📚 Documentation Files

- **`CPU_TRAINING_README.md`** - Comprehensive CPU training guide
- **`CPU_IMPLEMENTATION_SUMMARY.md`** - Technical implementation details
- **`SINGLE_TENSEGRITY_README.md`** - General single robot documentation
- **`CPU_QUICK_REFERENCE.md`** - This file!

## 🆘 Getting Help

1. Check `CPU_TRAINING_README.md` for detailed guide
2. Verify requirements: `pip install -r requirements.txt`
3. Test simulator: `python run_single.py`
4. Check Python version: `python --version` (need 3.8+)
5. Verify MuJoCo: `python -c "import mujoco; print(mujoco.__version__)"`

## ⏰ When to Use What

**Use `cpu_pretraining_SAC_single.py` when:**
- You don't have a GPU
- Training on a laptop
- Learning/experimenting with RL

**Use `gpu_pretraining_SAC_single.py` when:**
- You have an NVIDIA GPU
- Need fast training
- Running many experiments

**Use `run_single.py` when:**
- Testing simulator without RL
- Demonstrating gait patterns
- Debugging physics issues

**Use `test_trained_model_SAC_single.py` when:**
- Evaluating trained models
- Creating demo videos
- Comparing different models

---

**Ready to start?** Run: `python cpu_pretraining_SAC_single.py`
