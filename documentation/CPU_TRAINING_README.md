# CPU Training Guide for Single Tensegrity

This guide explains how to train and test the single tensegrity robot on systems **without GPU**.

## 🚨 Important Notes

- **CPU training is significantly slower than GPU training** (typically 10-50x slower)
- Recommended for laptops or systems without NVIDIA GPUs
- Training is automatic based on available system RAM
- All scripts automatically detect and use CPU when GPU is unavailable

## 📋 Files Overview

### CPU-Compatible Scripts

1. **`cpu_pretraining_SAC_single.py`** - Main CPU training script
   - Automatically detects available RAM
   - Selects appropriate model size (small/medium/large)
   - Uses smaller batch sizes and networks for CPU efficiency
   - Saves models to `models/cpu/`

2. **`test_trained_model_SAC_single.py`** - Testing script (AUTO-DETECTS CPU/GPU)
   - Automatically uses CPU if GPU unavailable
   - Can load models from both `models/gpu/` and `models/cpu/`
   - Works with both GPU-trained and CPU-trained models

3. **`run_single.py`** - Demonstration script (ALWAYS CPU-COMPATIBLE)
   - No GPU required - pure MuJoCo simulation
   - Use to test simulator without RL

## 🚀 Quick Start - CPU Training

### Step 1: Ensure Gait Files Exist

Make sure you have gait pattern JSON files:

```bash
# Check if gait files exist
ls *.json
```

If not, generate them:

```bash
python single_tensegrity_gait_generation.py
```

### Step 2: Run CPU Training

Simply run the CPU training script:

```bash
python cpu_pretraining_SAC_single.py
```

**What happens automatically:**
- System checks available RAM
- Selects appropriate configuration:
  - **CPU Small** (8-12 GB RAM): 20k timesteps, 128-dim network
  - **CPU Medium** (12-16 GB RAM): 30k timesteps, 256-dim network  
  - **CPU Large** (16+ GB RAM): 50k timesteps, 512-dim network
- Loads `dual_robot_first_6.json` gait
- Performs behavioral cloning pretraining
- Fine-tunes with SAC reinforcement learning
- Saves model to `models/cpu/`

**Expected training time:**
- Small config: ~30-60 minutes
- Medium config: ~1-2 hours
- Large config: ~2-4 hours

(Times vary based on CPU speed)

### Step 3: Test Trained Model

Test your trained model:

```bash
# Test latest model (auto-detects CPU)
python test_trained_model_SAC_single.py

# Or specify a model
python test_trained_model_SAC_single.py --model models/cpu/sac_single_cpu_pretraining_cpu_medium_20251020_123456
```

## 🎛️ Configuration Options

### CPU Training Script Options

The CPU training script automatically configures based on RAM, but you can modify the code to customize:

**Small Configuration (8-12 GB RAM):**
```python
{
    "learning_rate": 3e-4,
    "batch_size": 64,
    "buffer_size": 50000,
    "net_arch": [128, 128]
}
```

**Medium Configuration (12-16 GB RAM):**
```python
{
    "learning_rate": 3e-4,
    "batch_size": 128,
    "buffer_size": 100000,
    "net_arch": [256, 256]
}
```

**Large Configuration (16+ GB RAM):**
```python
{
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 200000,
    "net_arch": [512, 256, 128]
}
```

### Testing Script Options

```bash
# Basic usage (auto-detects latest model and CPU/GPU)
python test_trained_model_SAC_single.py

# Specify model
python test_trained_model_SAC_single.py --model path/to/model.zip

# Find latest model with specific config name
python test_trained_model_SAC_single.py --config cpu_medium

# Custom number of episodes
python test_trained_model_SAC_single.py --episodes 10

# Run without visualization (faster)
python test_trained_model_SAC_single.py --no-viz

# Save visualization frames
python test_trained_model_SAC_single.py --save-frames
```

## 📊 Monitoring Training

### TensorBoard

Monitor training progress in real-time:

```bash
tensorboard --logdir ./sac_single_cpu_tensorboard_cpu_small/
# or cpu_medium / cpu_large depending on your config
```

Open browser to `http://localhost:6006`

**Key metrics to watch:**
- `train/loss` - Should decrease over time
- `rollout/ep_rew_mean` - Episode reward (should increase)
- `rollout/ep_len_mean` - Episode length
- `eval/mean_reward` - Evaluation performance

### Console Output

The script prints progress including:
- System RAM usage
- Training phase progress bars
- Timing breakdowns for each phase
- Final model save location

## 🔧 Troubleshooting

### Issue: Out of Memory

**Symptoms:** Process killed, "MemoryError", system freezes

**Solutions:**
1. Close other applications
2. Edit `cpu_pretraining_SAC_single.py` to force smaller config:
   ```python
   selected_config = "cpu_small"  # Force small config
   timesteps = 10000  # Reduce timesteps
   cycles = 10  # Reduce demo cycles
   ```

### Issue: Training Very Slow

**Expected behavior:** CPU training is 10-50x slower than GPU

**Optimization tips:**
1. Reduce timesteps: Change `timesteps=20000` to `timesteps=10000`
2. Reduce demonstration cycles: Change `cycles=20` to `cycles=10`
3. Use smaller network: Force `cpu_small` configuration
4. Close background applications

### Issue: Model Not Found During Testing

**Solutions:**
1. Check model was saved:
   ```bash
   ls models/cpu/
   ```

2. Specify full model path:
   ```bash
   python test_trained_model_SAC_single.py --model models/cpu/sac_single_cpu_pretraining_cpu_medium_20251020_123456.zip
   ```

3. If no models exist, train first:
   ```bash
   python cpu_pretraining_SAC_single.py
   ```

### Issue: Visualization Not Working

**For testing:**
```bash
# Try without visualization
python test_trained_model_SAC_single.py --no-viz
```

**For demonstration:**
The `run_single.py` script requires visualization. If you have issues:
1. Check MuJoCo installation
2. Ensure X server running (Linux) or display available
3. Try reducing render size in the script

## 🔄 CPU vs GPU Comparison

| Feature | GPU Training | CPU Training |
|---------|--------------|--------------|
| **Speed** | Fast (baseline) | 10-50x slower |
| **Hardware** | NVIDIA GPU required | Any CPU works |
| **RAM Usage** | GPU RAM (8-24 GB) | System RAM (8-32 GB) |
| **Batch Size** | Large (256-2048) | Small (64-256) |
| **Network Size** | Large (512-1024) | Small (128-512) |
| **Model Quality** | Best | Good |
| **Cost** | High (GPU needed) | Low (any laptop) |

## 💡 Best Practices

### For Limited Resources

1. **Start small:**
   ```bash
   # Run with minimal config first
   python cpu_pretraining_SAC_single.py
   ```

2. **Test incrementally:**
   ```bash
   # Test after short training
   python test_trained_model_SAC_single.py --episodes 2
   ```

3. **Monitor resources:**
   - Watch RAM usage in Task Manager (Windows) or `htop` (Linux)
   - Stop if RAM usage exceeds 90%

### For Overnight Training

1. **Use larger configs** if you have 16+ GB RAM
2. **Increase timesteps** by editing the script
3. **Enable longer evaluation periods** for better checkpoints
4. **Close all other applications** to free resources

### For Quick Testing

1. **Use the demo script** (no RL needed):
   ```bash
   python run_single.py
   ```

2. **Generate and test gaits** without training:
   ```bash
   python single_tensegrity_gait_generation.py
   python run_single.py optimal_rolling_gait.json 5
   ```

## 📁 Output Structure

After training, you'll have:

```
models/
  cpu/
    sac_single_cpu_pretraining_cpu_small_TIMESTAMP.zip
    sac_single_cpu_pretraining_cpu_medium_TIMESTAMP.zip
    sac_single_cpu_pretraining_cpu_large_TIMESTAMP.zip

logs/
  best_model_single_cpu_[config]/
    best_model.zip
  evals_single_cpu_[config]/
    evaluations.npz

sac_single_cpu_tensorboard_[config]/
  [TensorBoard logs]
```

## 🎯 Next Steps

1. **Train your first model:**
   ```bash
   python cpu_pretraining_SAC_single.py
   ```

2. **Test the trained model:**
   ```bash
   python test_trained_model_SAC_single.py
   ```

3. **Monitor with TensorBoard:**
   ```bash
   tensorboard --logdir ./sac_single_cpu_tensorboard_cpu_medium/
   ```

4. **Experiment with gaits:**
   ```bash
   python single_tensegrity_gait_generation.py
   python run_single.py optimal_rolling_gait.json 10
   ```

5. **Compare CPU vs GPU models** (if you have access to both):
   - Train on CPU, test on GPU-capable machine
   - Models are compatible across devices

## ❓ FAQ

**Q: Can I use CPU-trained models on GPU systems?**  
A: Yes! Models are device-independent. Load with `device="cuda"` parameter.

**Q: How long does CPU training take?**  
A: 30 minutes to 4 hours depending on RAM and configuration.

**Q: Can I pause and resume training?**  
A: Not directly. You can stop and restart, but it will start from scratch.

**Q: Is CPU training quality worse than GPU?**  
A: No, just slower. Given enough time, CPU training achieves similar quality.

**Q: Can I speed up CPU training?**  
A: Yes - reduce timesteps, use smaller networks, fewer demo cycles.

**Q: Does the test script require GPU?**  
A: No! It auto-detects and uses CPU if GPU unavailable.

**Q: Does the run script require GPU?**  
A: No! It's pure MuJoCo simulation, always runs on CPU.

---

## 🆘 Support

If you encounter issues:

1. Check this troubleshooting guide
2. Verify system meets minimum requirements (8GB RAM)
3. Try the smallest configuration first
4. Test with `run_single.py` to verify simulator works
5. Check Python dependencies are installed: `pip install -r requirements.txt`

For questions about the project, see `SINGLE_TENSEGRITY_README.md` for general documentation.
