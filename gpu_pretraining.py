import time
import gc
import torch
import numpy as np
import psutil
from datetime import datetime
from pathlib import Path
from tensegrity_env import TensegrityEnv
from stable_baselines3 import PPO

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
    print("🚀 System ready for GPU training!")
    
    return True, gpu_name, (gpu_memory_gb, system_ram_gb)

def gpu_pretraining_with_roll_sequence(config_name, model_params, total_timesteps=75000, demo_cycles=None):
    """
    GPU-optimized pretraining using roll sequence - following coding guidelines
    """
    print(f"\n{'='*60}")
    print(f"🚀 GPU Pretraining: {config_name}")
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
        
        # 2. Roll sequence demonstration generation - preserving your exact pattern
        print("\n2️⃣ Generating Roll Sequence Demonstrations...")
        start_time = time.time()
        
        # Your exact roll_sequence from pretraining.py
        roll_sequence = np.array([
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,   0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,   0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,   0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,   0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 0.1, 1.0, 1.0, 0.1,   0.1, 0.8, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.8, 0.1,   1.0, 0.1, 1.0, 1.0, 0.1, 1.0],
            [1.0, 0.1, 1.0, 1.0, 0.1, 1.0,   0.0, 0.1, 0.8, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.8, 0.1, 1.0,   1.0, 1.0, 0.8, 1.0, 1.0, 0.1],
            [1.0, 1.0, 0.0, 0.8, 0.1, 1.0,   1.0, 1.0, 0.8, 1.0, 1.0, 0.1],
            [1.0, 0.1, 0.1, 1.0, 0.1, 0.1,   1.0, 0.8, 0.1, 1.0, 1.0, 0.0],
            [1.0, 0.5, 0.1, 1.0, 0.4, 0.1,   0.1, 0.8, 0.0, 1.0, 1.0, 0.5],
            [0.5, 0.5, 1.0, 1.0, 0.4, 0.1,   0.1, 0.8, 0.0, 0.5, 1.0, 1.0],
            [0.5, 0.5, 1.0, 1.0, 0.4, 0.1,   0.1, 0.8, 0.0, 0.5, 1.0, 1.0],
            [0.5, 0.5, 1.0, 1.0, 0.4, 0.8,   0.1, 0.8, 0.8, 0.5, 1.0, 1.0],
            [0.5, 0.5, 1.0, 0.4, 0.4, 0.8,   0.4, 0.8, 0.8, 0.5, 1.0, 1.0],
            [0.5, 0.5, 1.0, 0.4, 0.4, 0.8,   0.4, 0.8, 0.8, 0.5, 1.0, 1.0],
            [0.5, 0.5, 1.0, 0.4, 0.4, 0.8,   0.4, 0.8, 0.8, 0.5, 1.0, 1.0],
            [0.5, 0.5, 1.0, 0.1, 0.4, 0.8,   0.1, 0.8, 0.8, 0.5, 1.0, 1.0],
            [0.5, 0.5, 1.0, 0.1, 0.4, 0.8,   0.1, 0.8, 0.8, 0.5, 1.0, 1.0],
            [0.5, 0.5, 1.0, 0.1, 0.4, 0.8,   0.1, 0.8, 0.8, 0.5, 1.0, 1.0]
        ])
        
        print(f"   Using roll sequence: {roll_sequence.shape[0]} steps × {roll_sequence.shape[1]} actuators")
        
        # Validate actuator count per coding guidelines
        expected_actuators = 12
        if roll_sequence.shape[1] != expected_actuators:
            print(f"⚠️  ACTUATOR MISMATCH: roll_sequence has {roll_sequence.shape[1]} but expected {expected_actuators}")
        
        # Generate demonstration data - auto-adjust cycles if not specified
        obs, _ = env.reset()
        trajectory = {"observations": [], "actions": [], "rewards": []}
        
        if demo_cycles is None:
            # Auto-determine cycles based on config name
            if "rtx5090" in config_name:
                num_cycles = 2000
            elif "rtx4090" in config_name:
                num_cycles = 100 #1000
            elif "rtx2080ti_32gb" in config_name:
                num_cycles = 300
            else:
                num_cycles = 100
        else:
            num_cycles = demo_cycles
        
        print(f"   Generating {num_cycles} demonstration cycles for {config_name}")
        
        for cycle in range(num_cycles):
            if cycle % 100 == 0 and cycle > 0:
                print(f"     Progress: {cycle}/{num_cycles} cycles completed")
                
            for step_idx, action in enumerate(roll_sequence):
                action = np.array(action, dtype=np.float32)
                action = np.clip(action, 0.0, 1.0)  # Normalized cable lengths per coding guidelines
                
                trajectory["observations"].append(obs.copy())
                trajectory["actions"].append(action.copy())
                
                obs, reward, terminated, truncated, info = env.step(action)
                trajectory["rewards"].append(reward)
                
                if terminated or truncated:
                    obs, _ = env.reset()
        
        timing_breakdown['Roll Sequence Generation'] = time.time() - start_time
        print(f"   Roll sequence generation completed in {timing_breakdown['Roll Sequence Generation']:.2f} seconds")
        print(f"   Generated {len(trajectory['observations'])} demonstration samples")
        
        # 3. RTX 4090-optimized PPO model - leveraging full GPU power
        print("\n3️⃣ PPO Model Initialization...")
        start_time = time.time()
        
        # Monitor GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(0) / 1e9
        max_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"   GPU Memory - Allocated: {initial_memory:.2f}GB")
        print(f"   GPU Memory - Available: {max_gpu_memory - initial_memory:.2f}GB")
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"./ppo_tensegrity_tensorboard_{config_name}/",
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
        print(f"   ✅ PPO model initialization completed ({timing_breakdown['Model Initialization']:.2f}s)")
        
        # 4. GPU-accelerated behavioral cloning
        print("\n4️⃣ GPU-Accelerated Behavioral Cloning...")
        start_time = time.time()
        
        all_observations = np.array(trajectory["observations"], dtype=np.float32)
        all_actions = np.array(trajectory["actions"], dtype=np.float32)
        
        print(f"   Training on {len(all_observations)} samples")
        print(f"   Obs shape: {all_observations.shape}, Action shape: {all_actions.shape}")
        
        # RTX 4090-optimized behavioral cloning
        num_bc_epochs = 30  # More epochs leveraging GPU speed
        batch_size = model_params.get('batch_size', 256)  # Large batches for RTX 4090
        
        for epoch in range(num_bc_epochs):
            indices = np.random.permutation(len(all_observations))
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_obs = all_observations[batch_indices]
                batch_actions = all_actions[batch_indices]
                
                # GPU tensor operations
                obs_tensor = torch.FloatTensor(batch_obs).to(device)
                action_tensor = torch.FloatTensor(batch_actions).to(device)
                
                with torch.no_grad():
                    actions_pred, _, _ = model.policy(obs_tensor)
                
                loss = torch.nn.functional.mse_loss(actions_pred, action_tensor)
                epoch_loss += loss.item()
                num_batches += 1
            
            if epoch % 10 == 0:
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                current_memory = torch.cuda.memory_allocated(0) / 1e9
                print(f"   Epoch {epoch}: Loss={avg_loss:.4f}, GPU={current_memory:.2f}GB")
        
        timing_breakdown['Behavioral Cloning'] = time.time() - start_time
        print(f"   ✅ Behavioral cloning completed ({timing_breakdown['Behavioral Cloning']:.2f}s)")
        
        # 5. RL fine-tuning with RTX 4090 power
        print("\n5️⃣ RL Fine-Tuning...")
        start_time = time.time()
        
        # Monitor training memory usage
        pre_training_memory = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU memory before RL training: {pre_training_memory:.2f}GB")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   Training started at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Execute training with progress bar
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
        
        # 6. Save model with timestamp
        print("\n6️⃣ Saving Trained Model...")
        start_time = time.time()
        
        # Generate timestamp for unique model naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"models/gpu/ppo_gpu_pretraining_{config_name}_{timestamp}"
        Path("models/gpu").mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        
        timing_breakdown['Model Saving'] = time.time() - start_time
        print(f"   ✅ Model saved to: {save_path}")
        print(f"   ✅ Model saving completed ({timing_breakdown['Model Saving']:.2f}s)")
        
        # Print comprehensive timing summary
        total_time = sum(timing_breakdown.values())
        print(f"\n{'='*60}")
        print(f"🎯 RTX 4090 TRAINING SUMMARY - {config_name}")
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
        # RTX 4090 configurations (24GB VRAM)
        "rtx4090_large": {
            "learning_rate": 2e-4,
            "n_steps": 16384,        # Large rollout buffer for RTX 4090
            "batch_size": 1024,      # Big batches for GPU efficiency
            "n_epochs": 15,         # More epochs with GPU speed, might overfit
            "gamma": 0.999,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01, # was 0.01, promotes exploration
            "vf_coef": 1.0,
            "max_grad_norm": 0.5,
            "policy_kwargs": dict(
                net_arch=dict(pi=[1024, 1024, 512], vf=[1024, 1024, 512]),  # Large networks
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx4090_ultra": {
            "learning_rate": 2e-4,
            "n_steps": 16384,       # Very large rollout for maximum GPU utilization
            "batch_size": 1024,     # Massive batches
            "n_epochs": 5,         # Fewer epochs to prevent overfitting
            "gamma": 0.999,         # Long-term focus
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,      # was 0.01, promotes exploration
            "vf_coef": 1.0,
            "max_grad_norm": 0.5,
            "policy_kwargs": dict(
                net_arch=dict(pi=[2048, 1024, 512, 256], vf=[2048, 1024, 512, 256]),  # Very deep
                activation_fn=torch.nn.ReLU
            )
        },
        
        # RTX 2080 Ti configurations (11GB VRAM + 32GB RAM) - EXPLORATION FOCUSED
        "rtx2080ti_32gb_balanced": {
            "learning_rate": 5e-4,   # Higher LR for faster exploration of action space
            "n_steps": 8192,         # Larger rollout buffer with 32GB RAM
            "batch_size": 512,       # Moderate batch size for 11GB VRAM
            "n_epochs": 8,           # Fewer epochs to prevent over-fitting to early patterns
            "gamma": 0.99,           # Shorter discount factor to encourage immediate exploration
            "gae_lambda": 0.9,       # Reduced GAE lambda for less bias toward long-term rewards
            "clip_range": 0.3,       # Larger clip range allows bigger policy updates
            "ent_coef": 0.15,        # MAJOR INCREASE: Much higher entropy bonus for exploration
            "vf_coef": 0.5,          # Reduced value function weight to prioritize policy exploration
            "max_grad_norm": 1.0,    # Higher gradient norm allows larger updates
            "policy_kwargs": dict(
                net_arch=dict(pi=[768, 768, 384], vf=[768, 768, 384]),  # Larger than typical 2080 Ti
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx2080ti_32gb_efficient": {
            "learning_rate": 4e-4,   # Higher LR for exploration
            "n_steps": 6144,         # Conservative GPU memory usage
            "batch_size": 384,       # Safe for 11GB VRAM
            "n_epochs": 6,           # Fewer epochs to avoid exploitation too early
            "gamma": 0.985,          # Shorter horizon to focus on immediate exploration
            "gae_lambda": 0.9,       # Reduced for less long-term bias
            "clip_range": 0.25,      # Increased clip range for larger policy changes
            "ent_coef": 0.12,        # MAJOR INCREASE: High entropy for exploration
            "vf_coef": 0.4,          # Lower value function emphasis
            "max_grad_norm": 0.8,    # Higher gradient norm for larger updates
            "policy_kwargs": dict(
                net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),  # Conservative GPU memory
                activation_fn=torch.nn.ReLU
            )
        },
        
        # RTX 5090 configurations (32GB VRAM + 32GB RAM) - Next-gen powerhouse
        "rtx5090_extreme": {
            "learning_rate": 3e-4,
            "n_steps": 32768,        # Massive rollout buffer for 32GB VRAM
            "batch_size": 2048,      # Huge batches for ultimate efficiency
            "n_epochs": 20,          # More epochs with massive GPU power
            "gamma": 0.999,          # Long-term planning
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.08,        # High exploration with powerful GPU
            "vf_coef": 1.0,
            "max_grad_norm": 0.5,
            "policy_kwargs": dict(
                net_arch=dict(pi=[4096, 2048, 1024, 512], vf=[4096, 2048, 1024, 512]),  # Massive networks
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx5090_ultra": {
            "learning_rate": 3e-4,
            "n_steps": 24576,        # Very large but not maximum
            "batch_size": 1536,      # Large batches
            "n_epochs": 12,          # Balanced epochs
            "gamma": 0.998,          
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.06,        
            "vf_coef": 1.0,
            "max_grad_norm": 0.5,
            "policy_kwargs": dict(
                net_arch=dict(pi=[2048, 2048, 1024, 256], vf=[2048, 2048, 1024, 256]),  # Very deep
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
        timesteps = 150000  # Extended training for RTX 5090
        cycles = 2000       # Massive demonstration cycles
        print(f"🚀 RTX 5090 detected! Using next-gen configurations for {gpu_memory_gb:.1f}GB VRAM")
        
    elif "4090" in gpu_name:
        selected_configs = {k: v for k, v in configs.items() if "rtx4090" in k}
        timesteps = 500000  # Standard for RTX 4090
        cycles = 1515       # Large demonstration cycles
        print(f"🚀 RTX 4090 detected! Using optimized configurations for {gpu_memory_gb:.1f}GB VRAM")
        
    elif "2080" in gpu_name and system_ram_gb >= 24:
        selected_configs = {k: v for k, v in configs.items() if "rtx2080ti_32gb" in k}
        timesteps = 80000   # Moderate for RTX 2080 Ti + 32GB RAM
        cycles = 300        # Conservative GPU memory, more system RAM
        print(f"🚀 RTX 2080 Ti + 32GB RAM detected! Using enhanced configurations")
        
    else:
        # Default to RTX 4090 large as requested
        selected_configs = {"rtx4090_large": configs["rtx4090_large"]}
        timesteps = 300000
        cycles = 1500 #was 1000
        print(f"⚠️  Unknown/unsupported GPU '{gpu_name}' ({gpu_memory_gb:.1f}GB)")
        print(f"    Defaulting to RTX 4090 Large configuration as requested")
        print(f"    WARNING: This may cause GPU OOM if your hardware has insufficient VRAM")
    
    results = {}
    
    print(f"\n🚀 Starting GPU optimized training with {len(selected_configs)} configurations")
    print(f"   Training timesteps: {timesteps:,}")
    print(f"   Demonstration cycles: {cycles}")
    
    for config_name, params in selected_configs.items():
        print(f"\n{'='*80}")
        print(f"🎯 Training Configuration: {config_name}")
        print(f"{'='*80}")
        
        result = gpu_pretraining_with_roll_sequence(
            config_name, 
            params, 
            total_timesteps=timesteps,
            demo_cycles=cycles
        )
        results[config_name] = result
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏁 GPU PRETRAINING RESULTS SUMMARY")
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
    print(f"   Demonstration cycles per config: {cycles}")
    
    if successful_configs:
        print(f"\n🎮 To test your trained models:")
        for config in successful_configs:
            model_path = f"models/gpu/ppo_gpu_pretraining_{config}_*"
            print(f"   python test_trained_model.py --model {model_path}")
        
        print(f"\n📊 To view training progress:")
        print(f"   tensorboard --logdir ./ppo_tensegrity_tensorboard_{successful_configs[0]}/")
    
    if failed_configs:
        print(f"\n⚠️  Failed configurations may need:")
        print(f"   - Reduced batch_size or n_steps")
        print(f"   - Smaller network architectures")
        print(f"   - More system RAM or GPU VRAM")

if __name__ == "__main__":
    main()