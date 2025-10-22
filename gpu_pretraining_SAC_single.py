"""GPU-accelerated SAC pretraining for single tensegrity robot.

Adapted from gpu_pretraining_SAC.py for single 3-bar tensegrity with 6 actuators.
Loads gait patterns from JSON files for behavioral cloning pretraining.
"""

import time
import gc
import torch
import numpy as np
import psutil
import json
from datetime import datetime
from pathlib import Path
from single_tensegrity_env import SingleTensegrityEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback


def check_system_requirements():
    """
    Comprehensive system check for GPU training requirements.
    
    Returns
    -------
    success : bool
        Whether system meets minimum requirements
    gpu_name : str
        GPU model name (lowercase)
    memory_specs : Tuple[float, float]
        (GPU VRAM GB, System RAM GB)
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


def load_gait_from_json(json_path: str | Path) -> np.ndarray:
    """Load gait sequence from JSON file.
    
    Parameters
    ----------
    json_path : str or Path
        Path to JSON file containing gait sequence
    
    Returns
    -------
    gait_sequence : np.ndarray, shape (num_steps, 6)
        Gait sequence as numpy array
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Gait JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        gait_data = json.load(f)
    
    # Extract actions from JSON
    if 'actions' in gait_data:
        actions = np.array(gait_data['actions'], dtype=np.float32)
    else:
        # Assume JSON is just the action array
        actions = np.array(gait_data, dtype=np.float32)
    
    # Validate shape
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    
    if actions.shape[1] != 6:
        raise ValueError(f"Expected 6 actuators, got {actions.shape[1]} from {json_path}")
    
    print(f"✅ Loaded gait from {json_path.name}:")
    print(f"   Sequence length: {actions.shape[0]} steps")
    print(f"   Description: {gait_data.get('description', 'N/A')}")
    
    return actions


def gpu_pretraining_with_gait(config_name, model_params, gait_json_path, total_timesteps=75000, 
                               demo_cycles=None, reset_each_cycle=True):
    """
    GPU-optimized pretraining for single tensegrity using gait from JSON.
    
    Parameters
    ----------
    config_name : str
        Configuration name for logging
    model_params : dict
        SAC model parameters
    gait_json_path : str or Path
        Path to gait JSON file
    total_timesteps : int, default=75000
        Total RL training timesteps
    demo_cycles : int, optional
        Number of demonstration cycles (auto-determined if None)
    reset_each_cycle : bool, default=True
        Whether to reset environment each cycle
    
    Returns
    -------
    result : dict
        Training results with timing and success info
    """
    print(f"\n{'='*60}")
    print(f"🚀 GPU Pretraining (Single Tensegrity): {config_name}")
    print(f"{'='*60}")
    
    device = "cuda:0"
    timing_breakdown = {}
    
    try:
        # 1. Environment setup
        print("\n1️⃣ Environment Setup...")
        start_time = time.time()
        
        env = SingleTensegrityEnv(visualize=False, max_episode_steps=50)  # No visualization for training
        
        # CRITICAL validation per coding guidelines
        expected_actuators = 6
        actual_actuators = env.action_space.shape[0]
        
        if actual_actuators != expected_actuators:
            print(f"⚠️  ACTUATOR MISMATCH: Expected {expected_actuators} but got {actual_actuators}")
        else:
            print(f"✅ Actuator count verified: {actual_actuators}")
        
        print(f"✅ Environment configured:")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        print(f"   Action bounds: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
        print(f"   Simulator obs_dim: {env.sim.obs_dim}")
        
        timing_breakdown['Environment Setup'] = time.time() - start_time
        print(f"   ✅ Environment setup completed ({timing_breakdown['Environment Setup']:.2f}s)")
        
        # 2. Load gait sequence from JSON
        print("\n2️⃣ Loading Gait Sequence from JSON...")
        start_time = time.time()
        
        gait_sequence = load_gait_from_json(gait_json_path)
        
        # Validate actuator count matches
        if gait_sequence.shape[1] != expected_actuators:
            raise ValueError(f"Gait sequence has {gait_sequence.shape[1]} actuators, expected {expected_actuators}")
        
        print(f"   Gait shape: {gait_sequence.shape}")
        
        # 3. Generate demonstration data
        print("\n3️⃣ Generating Demonstrations...")
        
        obs, _ = env.reset()
        trajectory = {"observations": [], "actions": [], "rewards": []}
        
        # Auto-determine cycles if not specified
        if demo_cycles is None:
            if "rtx5090" in config_name.lower():
                num_cycles = 2000
            elif "rtx4090" in config_name.lower():
                num_cycles = 1000
            elif "rtx2080ti_32gb" in config_name.lower():
                num_cycles = 300
            else:
                num_cycles = 100
        else:
            num_cycles = demo_cycles
        
        print(f"   Generating {num_cycles} demonstration cycles")
        
        for cycle in range(num_cycles):
            # Reset for each cycle if needed
            if reset_each_cycle and cycle > 0:
                obs, _ = env.reset()
            
            if cycle % 100 == 0 and cycle > 0:
                print(f"     Progress: {cycle}/{num_cycles} cycles completed")
            
            # Execute gait sequence
            for step_idx, action in enumerate(gait_sequence):
                action = np.array(action, dtype=np.float32)
                action = np.clip(action, 0.0, 1.0)
                
                trajectory["observations"].append(obs.copy())
                trajectory["actions"].append(action.copy())
                
                obs, reward, terminated, truncated, info = env.step(action)
                trajectory["rewards"].append(reward)
                
                if terminated or truncated:
                    obs, _ = env.reset()
        
        timing_breakdown['Demonstration Generation'] = time.time() - start_time
        print(f"   ✅ Demo generation completed in {timing_breakdown['Demonstration Generation']:.2f}s")
        print(f"   Generated {len(trajectory['observations'])} demonstration samples")
        
        # 4. SAC Model Initialization
        print("\n4️⃣ SAC Model Initialization...")
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
            tensorboard_log=f"./sac_single_tensegrity_tensorboard_{config_name}/",
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
        
        # 5. GPU-accelerated behavioral cloning
        print("\n5️⃣ GPU-Accelerated Behavioral Cloning...")
        start_time = time.time()
        
        all_observations = np.array(trajectory["observations"], dtype=np.float32)
        all_actions = np.array(trajectory["actions"], dtype=np.float32)
        
        print(f"   Training on {len(all_observations)} samples")
        print(f"   Obs shape: {all_observations.shape}, Action shape: {all_actions.shape}")
        
        # Behavioral cloning parameters
        num_bc_epochs = 30
        batch_size = model_params.get('batch_size', 256)
        
        # Determine action scaling
        act_low = float(env.action_space.low[0])
        act_high = float(env.action_space.high[0])
        
        # Put actor in training mode
        model.policy.actor.train()
        
        for epoch in range(num_bc_epochs):
            indices = np.random.permutation(len(all_observations))
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_obs = all_observations[batch_indices]
                batch_actions = all_actions[batch_indices]
                
                # GPU tensor operations
                obs_tensor = torch.as_tensor(batch_obs, dtype=torch.float32, device=device)
                target_actions = torch.as_tensor(batch_actions, dtype=torch.float32, device=device)
                
                # Rescale if needed (env action space is [0,1], SAC typically uses [-1,1])
                if act_low == -1.0 and act_high == 1.0:
                    target_actions = target_actions * 2.0 - 1.0
                
                # Forward through actor
                actor_out = model.policy.actor(obs_tensor, deterministic=True)
                if isinstance(actor_out, (tuple, list)):
                    actions_pred = actor_out[0]
                else:
                    actions_pred = actor_out
                
                # Compute loss and update
                loss = torch.nn.functional.mse_loss(actions_pred, target_actions)
                model.policy.actor.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                model.policy.actor.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if epoch % 10 == 0:
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                current_memory = torch.cuda.memory_allocated(0) / 1e9
                print(f"   Epoch {epoch}: Loss={avg_loss:.4f}, GPU={current_memory:.2f}GB")
        
        timing_breakdown['Behavioral Cloning'] = time.time() - start_time
        print(f"   ✅ Behavioral cloning completed ({timing_breakdown['Behavioral Cloning']:.2f}s)")
        
        # 6. RL fine-tuning
        print("\n6️⃣ RL Fine-Tuning...")
        start_time = time.time()
        
        pre_training_memory = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU memory before RL training: {pre_training_memory:.2f}GB")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   Training started at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Set up evaluation environment
        eval_env = SingleTensegrityEnv(visualize=False, max_episode_steps=50)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./logs/best_model_single_{config_name}/",
            log_path=f"./logs/evals_single_{config_name}/",
            eval_freq=10000,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        # Execute training
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        post_training_memory = torch.cuda.memory_allocated(0) / 1e9
        peak_memory = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"   GPU memory after RL training: {post_training_memory:.2f}GB")
        print(f"   Peak GPU memory: {peak_memory:.2f}GB")
        
        timing_breakdown['RL Training'] = time.time() - start_time
        print(f"   ✅ RL training completed ({timing_breakdown['RL Training']:.2f}s)")
        
        # 7. Save model
        print("\n7️⃣ Saving Trained Model...")
        start_time = time.time()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"models/gpu/sac_single_gpu_pretraining_{config_name}_{timestamp}"
        Path("models/gpu").mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        
        timing_breakdown['Model Saving'] = time.time() - start_time
        print(f"   ✅ Model saved to: {save_path}")
        print(f"   ✅ Model saving completed ({timing_breakdown['Model Saving']:.2f}s)")
        
        # Print comprehensive timing summary
        total_time = sum(timing_breakdown.values())
        print(f"\n{'='*60}")
        print(f"🎯 SINGLE TENSEGRITY SAC TRAINING SUMMARY - {config_name}")
        print(f"{'='*60}")
        for phase, duration in timing_breakdown.items():
            percentage = (duration / total_time) * 100
            print(f"{phase:30}: {duration:6.2f}s ({percentage:5.1f}%)")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f}GB")
        print(f"Gait source: {Path(gait_json_path).name}")
        
        return {
            "success": True, 
            "timing": timing_breakdown, 
            "peak_memory": torch.cuda.max_memory_allocated(0) / 1e9,
            "model_path": save_path
        }
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ GPU Out of Memory: {e}")
        print("💡 Try reducing batch_size or network size")
        return {"success": False, "error": "GPU OOM"}
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    
    finally:
        # Aggressive GPU cleanup
        try:
            env.close()
            if 'eval_env' in locals():
                eval_env.close()
            del model, env
        except:
            pass
        
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 GPU memory cleanup completed")


def gpu_optimized_configs():
    """GPU-optimized configurations for different hardware setups."""
    
    return {
        # Conservative preset for RTX 2080 Ti 11GB
        "rtx2080ti_11gb_safe": {
            "learning_rate": 3e-4,
            "batch_size": 512,
            "gamma": 0.99,
            "ent_coef": 0.05,
            "policy_kwargs": dict(
                net_arch=[256, 256],
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx4090_large": {
            "learning_rate": 3e-4,
            "batch_size": 2048,
            "gamma": 0.999,
            "ent_coef": 0.1,
            "policy_kwargs": dict(
                net_arch=[2048, 1024, 512, 256],
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx4090_ultra": {
            "learning_rate": 3e-4,
            "batch_size": 2048,
            "gamma": 0.999,
            "ent_coef": 0.1,
            "policy_kwargs": dict(
                net_arch=[2048, 1024, 512, 256],
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx2080ti_32gb_efficient": {
            "learning_rate": 4e-4,
            "batch_size": 384,
            "gamma": 0.99,
            "ent_coef": 0.05,
            "policy_kwargs": dict(
                net_arch=[256, 256, 128],
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx5090_extreme": {
            "learning_rate": 3e-4,
            "batch_size": 2048,
            "gamma": 0.999,
            "ent_coef": 0.1,
            "policy_kwargs": dict(
                net_arch=[2048, 1024, 512, 256],
                activation_fn=torch.nn.ReLU
            )
        },
    }


def main():
    """Main GPU training pipeline with automatic hardware detection."""
    
    gpu_available, gpu_name, memory_specs = check_system_requirements()
    if not gpu_available:
        print("❌ System requirements not met. Exiting.")
        return
    
    gpu_memory_gb, system_ram_gb = memory_specs
    configs = gpu_optimized_configs()
    
    # Auto-detect optimal configuration
    if "5090" in gpu_name:
        selected_configs = {k: v for k, v in configs.items() if "rtx5090" in k}
        timesteps = 3000000
        cycles = 2000
        print(f"🚀 RTX 5090 detected! Using next-gen configs for {gpu_memory_gb:.1f}GB VRAM")
        
    elif "4090" in gpu_name:
        selected_configs = {k: v for k, v in configs.items() if "rtx4090" in k}
        timesteps = 200000
        cycles = 1000
        print(f"🚀 RTX 4090 detected! Using optimized configs for {gpu_memory_gb:.1f}GB VRAM")
        
    elif "2080" in gpu_name and system_ram_gb >= 24:
        selected_configs = {"rtx2080ti_32gb_efficient": configs["rtx2080ti_32gb_efficient"]}
        timesteps = 2_500_000
        cycles = 10
        print(f"🚀 RTX 2080 Ti + 32GB RAM detected! Using efficient config")
        
    else:
        # Default to RTX 4090 large
        selected_configs = {"rtx4090_large": configs["rtx4090_large"]}
        timesteps = 200000
        cycles = 1000
        print(f"⚠️  Unknown GPU '{gpu_name}' ({gpu_memory_gb:.1f}GB)")
        print(f"    Defaulting to RTX 4090 Large configuration")
    
    # Select gait JSON file
    gait_json = "dual_robot_first_6.json"  # Default gait
    # Alternatively, use: "optimal_rolling_gait.json" or any custom gait
    
    if not Path(gait_json).exists():
        print(f"\n❌ Gait file not found: {gait_json}")
        print("Please ensure gait JSON file exists or run single_tensegrity_gait_generation.py")
        return
    
    results = {}
    
    print(f"\n🚀 Starting GPU optimized training with {len(selected_configs)} configurations")
    print(f"   Training timesteps: {timesteps:,}")
    print(f"   Demonstration cycles: {cycles}")
    print(f"   Gait file: {gait_json}")
    
    for config_name, params in selected_configs.items():
        print(f"\n{'='*80}")
        print(f"🎯 Training Configuration: {config_name}")
        print(f"{'='*80}")
        
        result = gpu_pretraining_with_gait(
            config_name, 
            params,
            gait_json,
            total_timesteps=timesteps,
            demo_cycles=cycles,
            reset_each_cycle=True
        )
        results[config_name] = result
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏁 SINGLE TENSEGRITY GPU PRETRAINING RESULTS SUMMARY")
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
            print(f"   Model: {result['model_path']}")
        else:
            failed_configs.append(config)
            print(f"❌ {config}: {result['error']}")
    
    print(f"\n📊 Training Summary:")
    print(f"   Successful: {len(successful_configs)}/{len(results)}")
    print(f"   Timesteps per config: {timesteps:,}")
    print(f"   Demo cycles: {cycles}")
    print(f"   Gait: {gait_json}")
    
    if successful_configs:
        print(f"\n🎮 To test your trained models:")
        print(f"   python test_trained_model_SAC_single.py")
        
        print(f"\n📊 To view training progress:")
        print(f"   tensorboard --logdir ./sac_single_tensegrity_tensorboard_{successful_configs[0]}/")


if __name__ == "__main__":
    main()
