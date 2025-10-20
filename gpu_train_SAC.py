import time
import gc
import torch
import numpy as np
import psutil
from datetime import datetime
from pathlib import Path
from tensegrity_env import TensegrityEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback


def check_system_requirements():
    """
    Comprehensive system check for GPU training requirements - matching gpu_training.py
    
    Returns:
        success: bool - Whether system meets minimum requirements
        gpu_name: str - GPU model name (lowercase)
        memory_specs: Tuple[float, float] - (GPU VRAM GB, System RAM GB)
    """
    print("🔧 System Requirements Check")
    print("=" * 50)
    
    # CUDA availability check
    if not torch.cuda.is_available():
        print("❌ CUDA not available - GPU training not possible")
        print("   Install CUDA toolkit: https://developer.nvidia.com/cuda-toolkit")
        return False, None, None
    
    # GPU specifications
    gpu_name = torch.cuda.get_device_name(0).lower()
    gpu_properties = torch.cuda.get_device_properties(0)
    gpu_memory_gb = gpu_properties.total_memory / 1e9
    
    # System RAM
    system_memory = psutil.virtual_memory()
    system_ram_gb = system_memory.total / 1e9
    
    # Minimum requirements check
    min_gpu_memory = 6.0  # Minimum 6GB VRAM
    min_system_ram = 16.0  # Minimum 16GB RAM
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU VRAM: {gpu_memory_gb:.1f}GB")
    print(f"✅ System RAM: {system_ram_gb:.1f}GB")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ PyTorch Version: {torch.__version__}")
    
    # Requirements validation
    if gpu_memory_gb < min_gpu_memory:
        print(f"❌ Insufficient GPU memory: {gpu_memory_gb:.1f}GB < {min_gpu_memory}GB required")
        return False, gpu_name, (gpu_memory_gb, system_ram_gb)
    
    if system_ram_gb < min_system_ram:
        print(f"⚠️  Low system RAM: {system_ram_gb:.1f}GB (recommend >{min_system_ram}GB)")
    
    # Clear GPU cache and check current usage
    torch.cuda.empty_cache()
    current_usage = torch.cuda.memory_allocated(0) / 1e9
    available_memory = gpu_memory_gb - current_usage - 1.0  # Reserve 1GB
    
    print(f"📊 Available GPU memory: ~{available_memory:.1f}GB")
    print(f"🚀 System ready for GPU training!")
    
    return True, gpu_name, (gpu_memory_gb, system_ram_gb)


def gpu_training_sac(config_name, model_params, total_timesteps=75000):
    """
    GPU-optimized SAC training WITHOUT pretraining - pure RL learning from scratch
    """
    print(f"\n{'='*60}")
    print(f"🚀 GPU Training (No Pretraining): {config_name}")
    print(f"{'='*60}")
    
    device = "cuda:0"
    timing_breakdown = {}
    
    try:
        # 1. Environment setup - CRITICAL: verify obs_dim consistency per coding guidelines
        print("\n1️⃣ Environment Setup...")
        start_time = time.time()
        
        env = TensegrityEnv(visualize=False)  # No visualization for GPU training
        
        # CRITICAL validation per coding guidelines
        expected_obs_dim = 96  # From coding guidelines
        actual_obs_dim = env.observation_space.shape[0]
        
        if actual_obs_dim != expected_obs_dim:
            print(f"⚠️  MISMATCH FOUND: Expected obs_dim={expected_obs_dim} but got {actual_obs_dim}")
            print("   Cross-referencing with simulator...")
            print(f"   Simulator obs_dim: {env.sim.obs_dim}")
        else:
            print(f"✅ Observation dimensions verified: {actual_obs_dim}")
        
        print(f"✅ Environment configured:")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        print(f"   Action bounds: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
        
        timing_breakdown['Environment Setup'] = time.time() - start_time
        print(f"   ✅ Environment setup completed ({timing_breakdown['Environment Setup']:.2f}s)")
        
        # 2. SAC Model Initialization
        print("\n2️⃣ SAC Model Initialization...")
        start_time = time.time()
        
        # Monitor GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(0) / 1e9
        max_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"   GPU Memory - Allocated: {initial_memory:.2f}GB")
        print(f"   GPU Memory - Available: {max_gpu_memory - initial_memory:.2f}GB")
        
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"./sac_tensegrity_tensorboard_{config_name}/",
            device=device,
            **model_params
        )
        
        model_memory = torch.cuda.memory_allocated(0) / 1e9
        memory_increase = model_memory - initial_memory
        
        print(f"   ✅ Model initialized")
        print(f"   GPU Memory after model: {model_memory:.2f}GB (+{memory_increase:.2f}GB)")
        print(f"   Learning rate: {model_params['learning_rate']}")
        print(f"   Batch size: {model_params['batch_size']}")
        print(f"   Network architecture: {model_params['policy_kwargs']['net_arch']}")
        
        timing_breakdown['Model Initialization'] = time.time() - start_time
        print(f"   ✅ SAC model initialization completed ({timing_breakdown['Model Initialization']:.2f}s)")
        
        # 3. RL training from scratch (NO pretraining)
        print("\n3️⃣ RL Training from Scratch...")
        start_time = time.time()
        
        # Monitor training memory usage
        pre_training_memory = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU memory before RL training: {pre_training_memory:.2f}GB")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   Training started at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Execute training with progress bar (NO EVALUATION - faster training)
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True
        )
        
        post_training_memory = torch.cuda.memory_allocated(0) / 1e9
        peak_memory = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"   GPU memory after RL training: {post_training_memory:.2f}GB")
        print(f"   Peak GPU memory: {peak_memory:.2f}GB")
        
        timing_breakdown['RL Training'] = time.time() - start_time
        print(f"   ✅ RL training completed ({timing_breakdown['RL Training']:.2f}s)")
        
        # 4. Save model with timestamp
        print("\n4️⃣ Saving Trained Model...")
        start_time = time.time()
        
        # Generate timestamp for unique model naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"models/gpu/sac_gpu_training_{config_name}_{timestamp}"
        Path("models/gpu").mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        
        timing_breakdown['Model Saving'] = time.time() - start_time
        print(f"   ✅ Model saved to: {save_path}")
        print(f"   ✅ Model saving completed ({timing_breakdown['Model Saving']:.2f}s)")
        
        # Print comprehensive timing summary
        total_time = sum(timing_breakdown.values())
        print(f"\n{'='*60}")
        print(f"🎯 SAC TRAINING SUMMARY - {config_name}")
        print(f"{'='*60}")
        for phase, duration in timing_breakdown.items():
            percentage = (duration / total_time) * 100
            print(f"{phase:25}: {duration:6.2f}s ({percentage:5.1f}%)")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f}GB")
        
        return {"success": True, "timing": timing_breakdown, "peak_memory": torch.cuda.max_memory_allocated(0) / 1e9}
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ GPU Out of Memory: {e}")
        print("💡 Try reducing batch_size or network size")
        return {"success": False, "error": "GPU OOM"}
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return {"success": False, "error": str(e)}
    
    finally:
        # Aggressive GPU cleanup following coding guidelines
        try:
            env.close()
            del model, env
        except:
            pass
        
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 GPU memory cleanup completed")


def gpu_optimized_configs():
    """GPU-optimized configurations for different hardware setups"""
    
    return {
        # Conservative preset for RTX 2080 Ti 11GB
        "rtx2080ti_11gb_safe": {
            "learning_rate": 2e-4,     # Lower LR for stable pure RL learning
            "batch_size": 512,         # smaller to fit 11GB comfortably
            "gamma": 0.99,
            "ent_coef": 0.15,          # Higher entropy for more exploration
            "policy_kwargs": dict(
                net_arch=[256, 256],   # compact net
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx4090_large": {
            "learning_rate": 2e-4,     # Lower LR for stable pure RL learning
            "batch_size": 2048,
            "gamma": 0.99,             # Lower gamma for better short-term learning
            "ent_coef": 0.2,           # Higher entropy for more exploration
            "policy_kwargs": dict(
                net_arch=[2048, 1024, 512, 256],  # Shared for all networks in SAC
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx4090_ultra": {
            "learning_rate": 2e-4,     # Lower LR for stable pure RL learning
            "batch_size": 2048,
            "gamma": 0.99,             # Lower gamma for better short-term learning
            "ent_coef": 0.2,           # Higher entropy for more exploration
            "policy_kwargs": dict(
                net_arch=[2048, 1024, 512, 256],
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx2080ti_32gb_balanced": {
            # Balanced config for 11GB VRAM + 32GB RAM, good throughput without OOM
            "learning_rate": 2e-4,     # Lower LR for stable pure RL learning
            "batch_size": 512,
            "gamma": 0.99,             # Lower gamma for better short-term learning
            "ent_coef": 0.15,          # Higher entropy for more exploration
            "policy_kwargs": dict(
                net_arch=[512, 512, 256],
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx2080ti_32gb_efficient": {
            # Faster training wall-clock; smaller nets and batches - optimized for pure RL
            "learning_rate": 2e-4,     # Lower LR for stable pure RL learning
            "batch_size": 384,
            "gamma": 0.99,
            "ent_coef": 0.15,          # Higher entropy for more exploration
            "policy_kwargs": dict(
                net_arch=[256, 256, 128],
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx5090_extreme": {
            "learning_rate": 2e-4,     # Lower LR for stable pure RL learning
            "batch_size": 2048,
            "gamma": 0.99,             # Lower gamma for better short-term learning
            "ent_coef": 0.2,           # Higher entropy for more exploration
            "policy_kwargs": dict(
                net_arch=[2048, 1024, 512, 256],
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx5090_ultra": {
            "learning_rate": 2e-4,     # Lower LR for stable pure RL learning
            "batch_size": 2048,
            "gamma": 0.99,             # Lower gamma for better short-term learning
            "ent_coef": 0.2,           # Higher entropy for more exploration
            "policy_kwargs": dict(
                net_arch=[2048, 1024, 512, 256],
                activation_fn=torch.nn.ReLU
            )
        }
    }


def main():
    """Main GPU training pipeline with automatic hardware detection"""
    
    gpu_available, gpu_name, memory_specs = check_system_requirements()
    if not gpu_available:
        print("❌ System requirements not met. Exiting.")
        return
    
    gpu_memory_gb, system_ram_gb = memory_specs
    configs = gpu_optimized_configs()
    
    # Auto-detect optimal configurations based on hardware
    if "5090" in gpu_name:
        selected_configs = {k: v for k, v in configs.items() if "rtx5090" in k}
        timesteps = 5_000_000  # 5M steps for pure RL on RTX 5090
        print(f"🚀 RTX 5090 detected! Using next-gen configurations for {gpu_memory_gb:.1f}GB VRAM")
        
    elif "4090" in gpu_name:
        selected_configs = {k: v for k, v in configs.items() if "rtx4090" in k}
        timesteps = 3_000_000  # 3M steps for pure RL on RTX 4090
        print(f"🚀 RTX 4090 detected! Using optimized configurations for {gpu_memory_gb:.1f}GB VRAM")
        
    elif "2080" in gpu_name and system_ram_gb >= 24:
        # Use efficient config with adequate timesteps for pure RL learning
        selected_configs = {"rtx2080ti_32gb_efficient": configs["rtx2080ti_32gb_efficient"]}
        timesteps = 2_000_000  # 2M steps minimum for pure RL learning
        print(f"🚀 RTX 2080 Ti + 32GB RAM detected! Using efficient configuration for 2M steps")
        
    else:
        # Default to RTX 4090 large as requested
        selected_configs = {"rtx4090_large": configs["rtx4090_large"]}
        timesteps = 3_000_000  # 3M steps for pure RL
        print(f"⚠️  Unknown/unsupported GPU '{gpu_name}' ({gpu_memory_gb:.1f}GB)")
        print(f"    Defaulting to RTX 4090 Large configuration as requested")
        print(f"    WARNING: This may cause GPU OOM if your hardware has insufficient VRAM")

    results = {}
    
    print(f"\n🚀 Starting GPU optimized training with {len(selected_configs)} configurations")
    print(f"   Training timesteps: {timesteps:,}")
    print(f"   NOTE: No pretraining - pure RL learning from scratch")
    
    for config_name, params in selected_configs.items():
        print(f"\n{'='*80}")
        print(f"🎯 Training Configuration: {config_name}")
        print(f"{'='*80}")
        
        result = gpu_training_sac(
            config_name, 
            params, 
            total_timesteps=timesteps
        )
        results[config_name] = result
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏁 GPU TRAINING RESULTS SUMMARY")
    print(f"{'='*80}")
    
    successful_configs = []
    failed_configs = []
    
    for config, result in results.items():
        if result["success"]:
            successful_configs.append(config)
            total_time = sum(result["timing"].values())
            peak_memory = result["peak_memory"]
            print(f"✅ {config}")
            print(f"   Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
            print(f"   Peak GPU usage: {peak_memory:.2f}GB")
            print(f"   Average training speed: {timesteps / result['timing']['RL Training']:.1f} timesteps/s")
        else:
            failed_configs.append(config)
            print(f"❌ {config}: {result['error']}")
    
    print(f"\n📊 Training Summary:")
    print(f"   Successful configurations: {len(successful_configs)}/{len(results)}")
    print(f"   Training timesteps per config: {timesteps:,}")
    print(f"   Training mode: Pure RL (no pretraining)")
    
    if successful_configs:
        print(f"\n🎮 To test your trained models:")
        for config in successful_configs:
            model_path = f"models/gpu/sac_gpu_training_{config}_*"
            print(f"   python test_trained_model_SAC.py --model {model_path}")
        
        print(f"\n📊 To view training progress:")
        print(f"   tensorboard --logdir ./sac_tensegrity_tensorboard_{successful_configs[0]}/")
    
    if failed_configs:
        print(f"\n⚠️  Failed configurations may need:")
        print(f"   - Reduced batch_size")
        print(f"   - Smaller network architectures")
        print(f"   - More system RAM or GPU VRAM")


if __name__ == "__main__":
    main()
