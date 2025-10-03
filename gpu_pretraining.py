import time
import gc
import torch
import numpy as np
from pathlib import Path
from tensegrity_env import TensegrityEnv
from stable_baselines3 import PPO

def check_rtx4090_setup():
    """Verify RTX 4090 configuration following coding guidelines"""
    print("🔧 RTX 4090 Setup Verification:")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"   ✅ GPU: {gpu_name}")
    print(f"   ✅ Total VRAM: {gpu_memory:.1f}GB")
    
    # Check current memory usage
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    
    print(f"   📊 Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    print(f"   🚀 Available for training: ~{gpu_memory - 1.0:.1f}GB")  # Account for system usage
    
    return True

def rtx4090_pretraining_with_roll_sequence(config_name, model_params, total_timesteps=75000):
    """
    RTX 4090-optimized pretraining using roll sequence - following coding guidelines
    """
    print(f"\n{'='*60}")
    print(f"🚀 RTX 4090 Pretraining: {config_name}")
    print(f"{'='*60}")
    
    device = "cuda:0"
    timing_breakdown = {}
    
    try:
        # 1. Environment setup - CRITICAL: verify obs_dim consistency per coding guidelines
        print("\n1. Setting up environment...")
        start_time = time.time()
        
        env = TensegrityEnv(visualize=False)  # No visualization for GPU training
        
        # CRITICAL validation per coding guidelines
        expected_obs_dim = 78  # From coding guidelines
        actual_obs_dim = env.observation_space.shape[0]
        
        if actual_obs_dim != expected_obs_dim:
            print(f"⚠️  MISMATCH FOUND: Expected obs_dim={expected_obs_dim} but got {actual_obs_dim}")
            print("   Cross-referencing with simulator...")
            print(f"   Simulator obs_dim: {env.sim.obs_dim}")
        
        timing_breakdown['Environment Setup'] = time.time() - start_time
        print(f"   Environment setup completed in {timing_breakdown['Environment Setup']:.2f} seconds")
        
        # 2. Roll sequence demonstration generation - preserving your exact pattern
        print("\n2. Generating roll sequence demonstrations...")
        start_time = time.time()
        
        # Your exact roll_sequence from pretraining.py
        roll_sequence = np.array([
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
        
        # Generate demonstration data
        obs, _ = env.reset()
        trajectory = {"observations": [], "actions": [], "rewards": []}
        
        # Execute roll sequence multiple cycles for richer training data
        num_cycles = 1000  # was 4, More cycles for RTX 4090, seems not to make a difference, maybe slightly better
        for cycle in range(num_cycles):
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
        print("\n3. Initializing RTX 4090-optimized PPO model...")
        start_time = time.time()
        
        # Monitor GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU memory before model init: {initial_memory:.2f}GB")
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"./ppo_tensegrity_tensorboard_{config_name}/",
            device=device,
            **model_params
        )
        
        model_memory = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU memory after model init: {model_memory:.2f}GB (Δ{model_memory - initial_memory:.2f}GB)")
        
        timing_breakdown['Model Initialization'] = time.time() - start_time
        print(f"   RTX 4090 PPO model initialization completed in {timing_breakdown['Model Initialization']:.2f} seconds")
        
        # 4. GPU-accelerated behavioral cloning
        print("\n4. GPU-accelerated behavioral cloning...")
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
        print(f"   Behavioral cloning completed in {timing_breakdown['Behavioral Cloning']:.2f} seconds")
        
        # 5. RL fine-tuning with RTX 4090 power
        print("\n5. RL fine-tuning with RTX 4090 acceleration...")
        start_time = time.time()
        
        # Monitor training memory usage
        pre_training_memory = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU memory before RL training: {pre_training_memory:.2f}GB")
        
        model.learn(total_timesteps=total_timesteps)
        
        post_training_memory = torch.cuda.memory_allocated(0) / 1e9
        print(f"   GPU memory after RL training: {post_training_memory:.2f}GB")
        
        timing_breakdown['RL Training'] = time.time() - start_time
        print(f"   RL training completed in {timing_breakdown['RL Training']:.2f} seconds")
        
        # 6. Save model
        print("\n6. Saving trained model...")
        start_time = time.time()
        
        save_path = f"models/ppo_tensegrity_rtx4090_{config_name}"
        Path("models").mkdir(exist_ok=True)
        model.save(save_path)
        
        timing_breakdown['Model Saving'] = time.time() - start_time
        print(f"   Model saved to {save_path} in {timing_breakdown['Model Saving']:.2f} seconds")
        
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

def rtx4090_optimized_configs():
    """RTX 4090-optimized configurations leveraging full 24GB VRAM"""
    
    return {
        "rtx4090_large": {
            "learning_rate": 3e-4,
            "n_steps": 16384,        # Large rollout buffer for RTX 4090
            "batch_size": 1024,      # Big batches for GPU efficiency
            "n_epochs": 15,         # More epochs with GPU speed, might overfit
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.05, # was 0.01, promotes exploration
            "vf_coef": 1.0,
            "max_grad_norm": 0.5,
            "policy_kwargs": dict(
                net_arch=dict(pi=[1024, 1024, 512], vf=[1024, 1024, 512]),  # Large networks
                activation_fn=torch.nn.ReLU
            )
        },
        "rtx4090_ultra": {
            "learning_rate": 3e-4,
            "n_steps": 16384,       # Very large rollout for maximum GPU utilization
            "batch_size": 1024,     # Massive batches
            "n_epochs": 5,         # Fewer epochs to prevent overfitting
            "gamma": 0.999,         # Long-term focus
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.05,      # was 0.01, promotes exploration
            "vf_coef": 1.0,
            "max_grad_norm": 0.5,
            "policy_kwargs": dict(
                net_arch=dict(pi=[2048, 1024, 512, 256], vf=[2048, 1024, 512, 256]),  # Very deep
                activation_fn=torch.nn.ReLU
            )
        }
    }

def main():
    """Main RTX 4090 training pipeline"""
    
    if not check_rtx4090_setup():
        print("❌ RTX 4090 setup failed")
        return
    
    configs = rtx4090_optimized_configs()
    results = {}
    
    print(f"\n🚀 Starting RTX 4090 optimized training with {len(configs)} configurations")
    
    for config_name, params in configs.items():
        print(f"\n{'='*80}")
        print(f"🎯 Training Configuration: {config_name}")
        print(f"{'='*80}")
        
        result = rtx4090_pretraining_with_roll_sequence(
            config_name, 
            params, 
            total_timesteps=100000  # Longer training leveraging GPU speed
        )
        results[config_name] = result
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏁 FINAL RTX 4090 TRAINING RESULTS")
    print(f"{'='*80}")
    
    for config, result in results.items():
        if result["success"]:
            total_time = sum(result["timing"].values())
            peak_memory = result["peak_memory"]
            print(f"{config:20}: ✅ {total_time:.1f}s, Peak: {peak_memory:.1f}GB")
        else:
            print(f"{config:20}: ❌ {result['error']}")

if __name__ == "__main__":
    main()