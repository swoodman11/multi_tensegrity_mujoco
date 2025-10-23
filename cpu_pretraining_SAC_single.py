"""CPU-compatible SAC pretraining for single tensegrity robot.

Adapted from gpu_pretraining_SAC_single.py for CPU training.
Loads gait patterns from JSON files for behavioral cloning pretraining.
"""

import time
import gc
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
    Check system requirements for CPU training.
    
    Returns
    -------
    success : bool
        Whether system meets minimum requirements
    memory_specs : float
        System RAM GB
    """
    print("🔧 System Requirements Check (CPU Mode)")
    print("=" * 50)
    
    # System RAM
    system_memory = psutil.virtual_memory()
    system_ram_gb = system_memory.total / 1e9
    
    # Minimum requirements check
    min_system_ram = 8.0  # Minimum 8GB RAM for CPU training
    
    print(f"✅ System RAM: {system_ram_gb:.1f}GB")
    print(f"✅ CPU cores: {psutil.cpu_count()}")
    
    # Requirements validation
    if system_ram_gb < min_system_ram:
        print(f"⚠️  Low system RAM: {system_ram_gb:.1f}GB (recommend >{min_system_ram}GB)")
        print("   Training may be slow with limited RAM")
    
    available_memory = system_memory.available / 1e9
    print(f"📊 Available RAM: ~{available_memory:.1f}GB")
    print("🚀 System ready for CPU training!")
    print("⚠️  Note: CPU training will be slower than GPU training")
    
    return True, system_ram_gb


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


def cpu_pretraining_with_gait(config_name, model_params, gait_json_path, total_timesteps=50000, 
                               demo_cycles=None, reset_each_cycle=True):
    """
    CPU-optimized pretraining for single tensegrity using gait from JSON.
    
    Parameters
    ----------
    config_name : str
        Configuration name for logging
    model_params : dict
        SAC model parameters
    gait_json_path : str or Path
        Path to gait JSON file
    total_timesteps : int, default=50000
        Total RL training timesteps (reduced for CPU)
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
    print(f"🚀 CPU Pretraining (Single Tensegrity): {config_name}")
    print(f"{'='*60}")
    
    device = "cpu"
    timing_breakdown = {}
    
    try:
        # 1. Environment setup
        print("\n1️⃣ Environment Setup...")
        start_time = time.time()
        
        env = SingleTensegrityEnv(visualize=False, max_episode_steps=50)  # No visualization for training
        
        # Validation
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
        
        # 3. Generate demonstration data with rotation randomization
        print("\n3️⃣ Generating Demonstrations with Random Orientations...")
        
        obs, _ = env.reset()
        trajectory = {"observations": [], "actions": [], "rewards": []}
        
        # Fixed parameters for rotation randomization
        num_rotations = 20  # Number of different random orientations
        cycles_per_rotation = 5  # Gait cycles per orientation
        total_cycles = num_rotations * cycles_per_rotation  # 100 total cycles
        
        print(f"   Rotations: {num_rotations} random orientations")
        print(f"   Cycles per rotation: {cycles_per_rotation}")
        print(f"   Total cycles: {total_cycles}")
        
        for rotation_idx in range(num_rotations):
            # Reset and apply random Z-axis rotation
            obs, _ = env.reset()
            
            # Randomize robot orientation around Z-axis
            random_yaw = np.random.uniform(0, 2 * np.pi)
            
            # Apply rotation by modifying qpos (quaternion orientation)
            # Get current qpos
            qpos = env.sim.mjc_data.qpos.copy()
            
            # Create rotation quaternion for Z-axis rotation
            # quat = [w, x, y, z] for rotation angle theta around Z-axis:
            # w = cos(theta/2), x = 0, y = 0, z = sin(theta/2)
            quat_w = np.cos(random_yaw / 2)
            quat_z = np.sin(random_yaw / 2)
            
            # Update all body quaternions (assuming 7-element qpos per body: pos_xyz + quat_wxyz)
            num_bodies = len(qpos) // 7
            for body_idx in range(num_bodies):
                qpos_start = body_idx * 7
                # Keep position, update quaternion
                qpos[qpos_start + 3] = quat_w  # w
                qpos[qpos_start + 4] = 0.0      # x
                qpos[qpos_start + 5] = 0.0      # y
                qpos[qpos_start + 6] = quat_z   # z
            
            # Set new qpos
            env.sim.mjc_data.qpos[:] = qpos
            env.sim.forward()
            
            # Get observation after rotation
            obs = env.sim.get_observation()
            
            print(f"   Rotation {rotation_idx + 1}/{num_rotations}: yaw={np.degrees(random_yaw):.1f}°")
            
            # Run gait for multiple cycles with this orientation
            for cycle in range(cycles_per_rotation):
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
                        break
        
        timing_breakdown['Demonstration Generation'] = time.time() - start_time
        print(f"   ✅ Demo generation completed in {timing_breakdown['Demonstration Generation']:.2f}s")
        print(f"   Generated {len(trajectory['observations'])} demonstration samples")
        print(f"   From {num_rotations} rotations × {cycles_per_rotation} cycles = {total_cycles} total cycles")
        
        # 4. SAC Model Initialization
        print("\n4️⃣ SAC Model Initialization (CPU)...")
        start_time = time.time()
        
        # Monitor RAM
        memory_before = psutil.virtual_memory().used / 1e9
        
        print(f"   System RAM used: {memory_before:.2f}GB")
        
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"./sac_single_cpu_tensorboard_{config_name}/",
            device=device,
            **model_params
        )
        
        memory_after = psutil.virtual_memory().used / 1e9
        memory_increase = memory_after - memory_before
        
        print(f"   ✅ Model initialized on CPU")
        print(f"   RAM after model: {memory_after:.2f}GB (+{memory_increase:.2f}GB)")
        print(f"   Learning rate: {model_params['learning_rate']}")
        print(f"   Batch size: {model_params['batch_size']}")
        print(f"   Network architecture: {model_params['policy_kwargs']['net_arch']}")
        
        timing_breakdown['Model Initialization'] = time.time() - start_time
        print(f"   ✅ SAC model initialization completed ({timing_breakdown['Model Initialization']:.2f}s)")
        
        # 5. CPU behavioral cloning
        print("\n5️⃣ CPU Behavioral Cloning...")
        start_time = time.time()
        
        all_observations = np.array(trajectory["observations"], dtype=np.float32)
        all_actions = np.array(trajectory["actions"], dtype=np.float32)
        
        print(f"   Training on {len(all_observations)} samples")
        print(f"   Obs shape: {all_observations.shape}, Action shape: {all_actions.shape}")
        
        # Behavioral cloning parameters (reduced for CPU)
        num_bc_epochs = 20  # Reduced from 30 for faster CPU training
        batch_size = model_params.get('batch_size', 128)
        
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
                
                # CPU tensor operations
                import torch
                obs_tensor = torch.as_tensor(batch_obs, dtype=torch.float32, device=device)
                target_actions = torch.as_tensor(batch_actions, dtype=torch.float32, device=device)
                
                # Rescale if needed
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
            
            if epoch % 5 == 0:
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                memory_used = psutil.virtual_memory().used / 1e9
                print(f"   Epoch {epoch}: Loss={avg_loss:.4f}, RAM={memory_used:.2f}GB")
        
        timing_breakdown['Behavioral Cloning'] = time.time() - start_time
        print(f"   ✅ Behavioral cloning completed ({timing_breakdown['Behavioral Cloning']:.2f}s)")
        
        # 6. RL fine-tuning
        print("\n6️⃣ RL Fine-Tuning (CPU)...")
        start_time = time.time()
        
        memory_before_rl = psutil.virtual_memory().used / 1e9
        print(f"   RAM before RL training: {memory_before_rl:.2f}GB")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   Training started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"   ⏰ Estimated time: ~{total_timesteps*100/100:.0f} seconds of physics simulation")
        
        # Set up evaluation environment
        eval_env = SingleTensegrityEnv(visualize=False, max_episode_steps=50)
        
        # Set eval frequency - evaluate frequently for ~45 min training
        # With 12-18k timesteps, eval every 500 steps = 24-36 evaluations total
        eval_freq = 500  # Evaluate every 500 timesteps (frequent monitoring)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./logs/best_model_single_cpu_{config_name}/",
            log_path=f"./logs/evals_single_cpu_{config_name}/",
            eval_freq=eval_freq,
            n_eval_episodes=10,  # Evaluate with 10 episodes per evaluation
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
        
        memory_after_rl = psutil.virtual_memory().used / 1e9
        print(f"   RAM after RL training: {memory_after_rl:.2f}GB")
        
        timing_breakdown['RL Training'] = time.time() - start_time
        print(f"   ✅ RL training completed ({timing_breakdown['RL Training']:.2f}s)")
        
        # 7. Save model
        print("\n7️⃣ Saving Trained Model...")
        start_time = time.time()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"models/cpu/sac_single_cpu_pretraining_{config_name}_{timestamp}"
        Path("models/cpu").mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        
        timing_breakdown['Model Saving'] = time.time() - start_time
        print(f"   ✅ Model saved to: {save_path}")
        print(f"   ✅ Model saving completed ({timing_breakdown['Model Saving']:.2f}s)")
        
        # Print comprehensive timing summary
        total_time = sum(timing_breakdown.values())
        print(f"\n{'='*60}")
        print(f"🎯 SINGLE TENSEGRITY CPU SAC TRAINING SUMMARY - {config_name}")
        print(f"{'='*60}")
        for phase, duration in timing_breakdown.items():
            percentage = (duration / total_time) * 100
            print(f"{phase:30}: {duration:6.2f}s ({percentage:5.1f}%)")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        print(f"Gait source: {Path(gait_json_path).name}")
        
        return {
            "success": True, 
            "timing": timing_breakdown, 
            "model_path": save_path
        }
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    
    finally:
        # Cleanup
        try:
            env.close()
            if 'eval_env' in locals():
                eval_env.close()
            del model, env
        except:
            pass
        
        gc.collect()
        print("🧹 Memory cleanup completed")


def cpu_optimized_configs():
    """CPU-optimized configurations with smaller networks and batches."""
    
    import torch
    
    return {
        "cpu_small": {
            "learning_rate": 3e-4,
            "batch_size": 64,
            "gamma": 0.99,
            "ent_coef": 0.05,
            "buffer_size": 50000,  # Smaller buffer for CPU
            "policy_kwargs": dict(
                net_arch=[128, 128],  # Small network for CPU
                activation_fn=torch.nn.ReLU
            )
        },
        "cpu_medium": {
            "learning_rate": 3e-4,
            "batch_size": 128,
            "gamma": 0.99,
            "ent_coef": 0.05,
            "buffer_size": 100000,
            "policy_kwargs": dict(
                net_arch=[256, 256],  # Medium network
                activation_fn=torch.nn.ReLU
            )
        },
        "cpu_large": {
            "learning_rate": 3e-4,
            "batch_size": 256,
            "gamma": 0.995,
            "ent_coef": 0.1,
            "buffer_size": 200000,
            "policy_kwargs": dict(
                net_arch=[512, 256, 128],  # Larger network (if RAM allows)
                activation_fn=torch.nn.ReLU
            )
        },
    }


def main():
    """Main CPU training pipeline."""
    
    success, system_ram_gb = check_system_requirements()
    if not success:
        print("❌ System requirements not met. Exiting.")
        return
    
    configs = cpu_optimized_configs()
    
    # Select configuration based on available RAM
    # Configured for ~1 hour training session with frequent evaluations
    # Target: 1 hour = 3600 seconds, allocate ~10% to BC, ~85% to RL, ~5% overhead
    # RL time budget: ~3060s, at ~0.1s/step on CPU = ~30k timesteps
    if system_ram_gb >= 16:
        selected_config = "cpu_large"
        timesteps = 30000  # 30k steps × 1 second = 30,000 seconds simulated (~50 min RL on CPU)
        cycles = 50
        print(f"🚀 Using CPU Large configuration ({system_ram_gb:.1f}GB RAM)")
    elif system_ram_gb >= 12:
        selected_config = "cpu_medium"
        timesteps = 30000  # 30k steps × 1 second = 30,000 seconds simulated (~50 min RL on CPU)
        cycles = 30
        print(f"🚀 Using CPU Medium configuration ({system_ram_gb:.1f}GB RAM)")
    else:
        selected_config = "cpu_small"
        timesteps = 30000  # 30k steps × 1 second = 30,000 seconds simulated (~50 min RL on CPU)
        cycles = 20
        print(f"🚀 Using CPU Small configuration ({system_ram_gb:.1f}GB RAM)")
    
    # Select gait JSON file - using BEST GAIT from 3bar_gaits.py
    gait_json = "gaits/handmade_gait_best_rolling.json"  # Best quasi-static rolling gait (6 steps, no rest)
    
    if not Path(gait_json).exists():
        print(f"\n❌ Gait file not found: {gait_json}")
        print("Please ensure handmade gait JSON files exist")
        return
    
    print(f"\n🚀 Starting CPU optimized training")
    print(f"   Configuration: {selected_config}")
    print(f"   Training timesteps: {timesteps:,}")
    print(f"   Demonstration cycles: {cycles}")
    print(f"   Gait file: {gait_json}")
    print(f"   ⚠️  CPU training is slower than GPU - this may take a while!")
    
    result = cpu_pretraining_with_gait(
        selected_config,
        configs[selected_config],
        gait_json,
        total_timesteps=timesteps,
        demo_cycles=cycles,
        reset_each_cycle=True
    )
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏁 SINGLE TENSEGRITY CPU PRETRAINING RESULTS")
    print(f"{'='*80}")
    
    if result["success"]:
        total_time = sum(result["timing"].values())
        print(f"✅ Training successful!")
        print(f"   Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        print(f"   Model: {result['model_path']}")
        print(f"\n🎮 To test your trained model:")
        print(f"   python test_trained_model_SAC_single.py --model {result['model_path']}")
        print(f"\n📊 To view training progress:")
        print(f"   tensorboard --logdir ./sac_single_cpu_tensorboard_{selected_config}/")
    else:
        print(f"❌ Training failed: {result['error']}")


if __name__ == "__main__":
    main()
