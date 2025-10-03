import time
import gc
import torch
import numpy as np
from pathlib import Path
from tensegrity_env import TensegrityEnv
from stable_baselines3 import PPO

def check_gpu_setup():
    """Verify GPU configuration following coding guidelines"""
    print("🔧 GPU Setup Verification:")
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        print(f"   ✅ CUDA available: {torch.version.cuda}")
        print(f"   🎯 GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    else:
        print("   ⚠️  No GPU available, falling back to CPU")
    return gpu_available

def gpu_pretraining_with_roll_sequence(config_name, model_params, gpu_id=0):
    """
    GPU-accelerated pretraining using your exact roll_sequence seeded gait
    """
    print(f"\n{'='*60}")
    print(f"🚀 GPU Pretraining: {config_name}")
    print(f"{'='*60}")
    
    # Set device following your patterns
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        print(f"Using device: {device}")
    else:
        device = "cpu"
        print("Using device: CPU (no GPU available)")
    
    timing_breakdown = {}
    
    try:
        # 1. Environment setup (following coding guidelines)
        print("\n1. Setting up environment...")
        start_time = time.time()
        
        # CRITICAL: Verify obs_dim consistency per coding guidelines
        env = TensegrityEnv(visualize=False)  # No visualization for GPU training
        expected_obs_dim = 78  # From coding guidelines
        actual_obs_dim = env.observation_space.shape[0]
        
        if actual_obs_dim != expected_obs_dim:
            print(f"⚠️  MISMATCH FOUND: Expected obs_dim={expected_obs_dim} but got {actual_obs_dim}")
            print("   Cross-referencing with simulator...")
            print(f"   Simulator obs_dim: {env.sim.obs_dim}")
        
        timing_breakdown['Environment Setup'] = time.time() - start_time
        print(f"   Environment setup completed in {timing_breakdown['Environment Setup']:.2f} seconds")
        
        # 2. Your exact roll sequence demonstration generation
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
        
        print(f"   Using roll sequence with {roll_sequence.shape[0]} steps, {roll_sequence.shape[1]} actuators")
        
        # Verify actuator count matches environment expectation
        expected_actuators = 12  # From coding guidelines
        if roll_sequence.shape[1] != expected_actuators:
            print(f"⚠️  ACTUATOR MISMATCH: roll_sequence has {roll_sequence.shape[1]} but expected {expected_actuators}")
        
        # Generate demonstration trajectories using your roll sequence
        demonstrations = []
        
        print("   Generating roll sequence demonstrations...")
        obs, _ = env.reset()
        trajectory = {"observations": [], "actions": [], "rewards": []}
        
        # Execute your roll sequence multiple times for more training data
        num_cycles = 3  # Repeat the sequence 3 times
        for cycle in range(num_cycles):
            print(f"     Cycle {cycle + 1}/{num_cycles}")
            
            for step_idx, action in enumerate(roll_sequence):
                # Ensure action is proper numpy array with correct bounds
                action = np.array(action, dtype=np.float32)
                action = np.clip(action, 0.0, 1.0)  # Ensure normalized cable lengths
                
                trajectory["observations"].append(obs.copy())
                trajectory["actions"].append(action.copy())
                
                obs, reward, terminated, truncated, info = env.step(action)
                trajectory["rewards"].append(reward)
                
                if terminated or truncated:
                    obs, _ = env.reset()
        
        demonstrations.append(trajectory)
        
        timing_breakdown['Roll Sequence Generation'] = time.time() - start_time
        print(f"   Roll sequence generation completed in {timing_breakdown['Roll Sequence Generation']:.2f} seconds")
        print(f"   Generated {len(trajectory['observations'])} demonstration samples")
        
        # 3. GPU-optimized PPO model initialization
        print("\n3. Initializing GPU-optimized PPO model...")
        start_time = time.time()
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"./ppo_tensegrity_tensorboard_{config_name}/",
            device=device,  # GPU acceleration
            **model_params
        )
        
        timing_breakdown['Model Initialization'] = time.time() - start_time
        print(f"   GPU PPO model initialization completed in {timing_breakdown['Model Initialization']:.2f} seconds")
        
        # 4. Behavioral cloning pretraining on roll sequence
        print("\n4. Behavioral cloning pretraining on roll sequence...")
        start_time = time.time()
        
        # Extract demonstration data
        all_observations = np.array(trajectory["observations"], dtype=np.float32)
        all_actions = np.array(trajectory["actions"], dtype=np.float32)
        
        print(f"   Training on {len(all_observations)} roll sequence samples")
        print(f"   Action shape: {all_actions.shape}, Observation shape: {all_observations.shape}")
        
        # Simple behavioral cloning with GPU acceleration
        num_bc_epochs = 25  # More epochs for better learning of roll sequence
        for epoch in range(num_bc_epochs):
            indices = np.random.permutation(len(all_observations))
            batch_size = model_params.get('batch_size', 128)
            
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_obs = all_observations[batch_indices]
                batch_actions = all_actions[batch_indices]
                
                # Convert to tensors and move to GPU
                obs_tensor = torch.FloatTensor(batch_obs).to(device)
                action_tensor = torch.FloatTensor(batch_actions).to(device)
                
                # Get policy predictions
                with torch.no_grad():
                    actions_pred, _, _ = model.policy(obs_tensor)
                
                # MSE loss for behavioral cloning
                loss = torch.nn.functional.mse_loss(actions_pred, action_tensor)
                epoch_loss += loss.item()
                num_batches += 1
            
            if epoch % 5 == 0:
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                print(f"   Epoch {epoch}: Avg BC Loss = {avg_loss:.4f}")
        
        timing_breakdown['Behavioral Cloning'] = time.time() - start_time
        print(f"   Behavioral cloning completed in {timing_breakdown['Behavioral Cloning']:.2f} seconds")
        
        # 5. RL fine-tuning (your main training loop)
        print("\n5. RL fine-tuning with GPU acceleration...")
        start_time = time.time()
        
        # Extract timesteps from params or use default
        total_timesteps = model_params.pop('total_timesteps', 50000)
        
        model.learn(total_timesteps=total_timesteps)
        
        timing_breakdown['RL Training'] = time.time() - start_time
        print(f"   RL training completed in {timing_breakdown['RL Training']:.2f} seconds")
        
        # 6. Save model (following your patterns)
        print("\n6. Saving trained model...")
        start_time = time.time()
        
        save_path = f"models/ppo_tensegrity_rollseq_{config_name}"
        Path("models").mkdir(exist_ok=True)
        model.save(save_path)
        
        timing_breakdown['Model Saving'] = time.time() - start_time
        print(f"   Model saved to {save_path} in {timing_breakdown['Model Saving']:.2f} seconds")
        
        # Print timing summary (following your current format)
        total_time = sum(timing_breakdown.values())
        print(f"\n{'='*60}")
        print(f"🎯 ROLL SEQUENCE TRAINING SUMMARY - {config_name}")
        print(f"{'='*60}")
        for phase, duration in timing_breakdown.items():
            percentage = (duration / total_time) * 100
            print(f"{phase:25}: {duration:6.2f}s ({percentage:5.1f}%)")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s on {device}")
        
        return {"success": True, "timing": timing_breakdown, "device": device}
        
    except Exception as e:
        print(f"❌ Training {config_name} failed: {e}")
        return {"success": False, "error": str(e), "device": device}
    
    finally:
        # GPU memory cleanup (following coding guidelines)
        try:
            env.close()
            del model, env
        except:
            pass
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 GPU memory cleanup completed")

def multi_gpu_roll_sequence_training():
    """
    Multi-GPU pretraining using your exact roll sequence
    """
    
    gpu_available = check_gpu_setup()
    num_gpus = torch.cuda.device_count() if gpu_available else 1
    
    # GPU-optimized configurations for roll sequence training
    configs = {
        "rollseq_baseline": {
            "learning_rate": 3e-4,
            "n_steps": 4096,        # Larger for GPU efficiency
            "batch_size": 128,      # GPU-optimized batch size
            "n_epochs": 10,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "total_timesteps": 75000,
            "policy_kwargs": dict(
                net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
                activation_fn=torch.nn.ReLU
            )
        },
        "rollseq_exploration": {
            "learning_rate": 3e-4,
            "n_steps": 4096,
            "batch_size": 128,
            "n_epochs": 10,
            "gamma": 0.99,          # Shorter horizon for locomotion
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.02,       # Higher exploration beyond roll sequence
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "total_timesteps": 75000,
            "policy_kwargs": dict(
                net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
                activation_fn=torch.nn.ReLU
            )
        }
    }
    
    results = {}
    config_list = list(configs.items())
    
    if gpu_available and num_gpus > 1:
        print(f"🚀 Starting multi-GPU roll sequence training on {num_gpus} GPUs")
        
        # Distribute configs across GPUs
        for i, (config_name, params) in enumerate(config_list):
            gpu_id = i % num_gpus
            result = gpu_pretraining_with_roll_sequence(config_name, params, gpu_id)
            results[config_name] = result
    else:
        print("🖥️  Sequential GPU/CPU roll sequence training")
        
        # Sequential training
        for config_name, params in config_list:
            result = gpu_pretraining_with_roll_sequence(config_name, params, gpu_id=0)
            results[config_name] = result
    
    return results

if __name__ == "__main__":
    # Set multiprocessing method for GPU compatibility
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    results = multi_gpu_roll_sequence_training()
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏁 FINAL ROLL SEQUENCE PRETRAINING RESULTS")
    print(f"{'='*80}")
    
    for config, result in results.items():
        if result["success"]:
            total_time = sum(result["timing"].values())
            print(f"{config:25}: ✅ {total_time:.1f}s on {result['device']}")
        else:
            print(f"{config:25}: ❌ {result['error']}")