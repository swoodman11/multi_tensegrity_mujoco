from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from tensegrity_env import TensegrityEnv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
import gc
import psutil
from datetime import datetime
from pathlib import Path
import argparse

# Force CPU usage since the warning indicates PPO works better on CPU with MLP policy
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def check_system_requirements():
    """
    System check for CPU pretraining requirements
    
    Returns:
        success: bool - Whether system meets minimum requirements
        device: str - Training device (always "cpu")
        memory_specs: Tuple[float, float] - (0.0, System RAM GB)
    """
    print("🔧 System Requirements Check")
    print("=" * 50)
    
    # System RAM
    system_memory = psutil.virtual_memory()
    system_ram_gb = system_memory.total / 1e9
    min_system_ram = 8.0  # Minimum 8GB RAM for pretraining
    
    print(f"✅ System RAM: {system_ram_gb:.1f}GB")
    print(f"✅ PyTorch Version: {torch.__version__}")
    print("ℹ️  Using CPU for training (GPU disabled for stability)")
    
    # Requirements validation
    if system_ram_gb < min_system_ram:
        print(f"❌ Insufficient system RAM: {system_ram_gb:.1f}GB < {min_system_ram}GB required")
        return False, "cpu", (0.0, system_ram_gb)
    
    # Check current RAM usage
    memory_info = psutil.virtual_memory()
    available_ram = memory_info.available / 1e9
    
    print(f"📊 Available RAM: ~{available_ram:.1f}GB")
    print("🚀 System ready for CPU pretraining!")
    
    return True, "cpu", (0.0, system_ram_gb)

def main():
    """Main pretraining pipeline with comprehensive progress tracking"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pretrain tensegrity model with expert demonstrations")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total RL training timesteps after pretraining")
    parser.add_argument("--demo_steps", type=int, default=1000, help="Number of demonstration cycles to generate")
    parser.add_argument("--pretrain_epochs", type=int, default=50, help="Number of pretraining epochs")
    parser.add_argument("--obs_mode", type=str, default="tier2", choices=["tier2", "legacy104"], 
                       help="Observation mode: tier2 (96-D) or legacy104 (104-D)")
    args = parser.parse_args()
    
    print("🤖 PPO Pretraining for Tensegrity Robot Locomotion")
    print("=" * 70)
    print("This script pretrains a locomotion policy using expert demonstrations")
    print("followed by reinforcement learning fine-tuning.")
    print()
    
    training_start_time = time.time()
    training_results = {
        "success": False,
        "timing": {},
        "system_metrics": {},
        "training_metrics": {}
    }
    
    try:
        # 1. System Requirements Check
        system_ok, device, memory_specs = check_system_requirements()
        if not system_ok:
            print("\n❌ System requirements not met. Exiting.")
            return
        
        gpu_memory_gb, system_ram_gb = memory_specs
        print(f"\n📋 Training Configuration:")
        print(f"   Device: {device}")
        print(f"   Observation mode: {args.obs_mode}")
        print(f"   Demonstration cycles: {args.demo_steps:,}")
        print(f"   Pretraining epochs: {args.pretrain_epochs}")
        print(f"   RL training timesteps: {args.timesteps:,}")
        
        # 2. Environment Setup
        print("\n1️⃣ Environment Setup...")
        env_start_time = time.time()
        
        env = TensegrityEnv(obs_mode=args.obs_mode, visualize=False)
        
        # Verify observation space consistency (critical per coding guidelines)
        if args.obs_mode == "tier2":
            expected_obs_dim = 96
        else:  # legacy104
            expected_obs_dim = 104
            
        actual_obs_dim = env.observation_space.shape[0]
        
        if actual_obs_dim != expected_obs_dim:
            print(f"⚠️  OBSERVATION DIMENSION MISMATCH:")
            print(f"   Expected: {expected_obs_dim}")
            print(f"   Actual: {actual_obs_dim}")
            print(f"   Simulator obs_dim: {env.sim.obs_dim}")
        else:
            print(f"✅ Observation dimensions verified: {actual_obs_dim}")
        
        print(f"✅ Environment configured:")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        print(f"   Action bounds: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
        
        training_results["timing"]["environment_setup"] = time.time() - env_start_time
        print(f"   ✅ Environment setup completed ({training_results['timing']['environment_setup']:.2f}s)")
        
        # 3. TensorBoard Logging Setup
        print("\n2️⃣ Logging Setup...")
        log_setup_time = time.time()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"./ppo_tensegrity_tensorboard/"
        
        # Ensure directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        # Standardized models directory
        Path("models").mkdir(exist_ok=True)
        
        print(f"   Log directory: {log_dir}")
        print(f"   View with: tensorboard --logdir {log_dir}")
        
        training_results["timing"]["logging_setup"] = time.time() - log_setup_time
        training_results["log_dir"] = log_dir
        # 4. Memory Monitoring
        print("\n3️⃣ Memory Check...")
        memory_info = psutil.virtual_memory()
        available_ram = memory_info.available / 1e9
        
        print(f"   System RAM - Available: {available_ram:.2f}GB")
        print(f"   System RAM - Used: {(memory_info.total - memory_info.available) / 1e9:.2f}GB")
        
        training_results["system_metrics"]["available_ram"] = available_ram
        
        # 5. Expert Demonstration Generation
        print("\n4️⃣ Generating Expert Demonstrations...")
        demo_start_time = time.time()
        
        # Define your rolling gait actions from 3bar_gaits.py
        # Based on one of the gaits that causes rolling
        # Steph trying shit
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
        
        # Create a dataset of observations and expert actions
        print(f"   Generating expert demonstration dataset with {args.demo_steps} cycles...")
        obs_dataset = []
        action_dataset = []
        
        obs, _ = env.reset()
        
        for cycle in range(args.demo_steps):
            if cycle % 100 == 0 and cycle > 0:
                print(f"     Progress: {cycle}/{args.demo_steps} cycles completed")
                
            for action in roll_sequence:
                # Store the current observation and the expert action
                obs_dataset.append(obs.copy())  # Use .copy() to avoid reference issues
                action_dataset.append(action.copy())
                
                # Take step in environment with expert action
                obs, _, done, _, _ = env.step(action)
                
                if done:
                    obs, _ = env.reset()
            
            obs, _ = env.reset()  # Reset at the end of each full gait cycle
        
        training_results["timing"]["demonstration_generation"] = time.time() - demo_start_time
        training_results["training_metrics"]["demo_samples"] = len(obs_dataset)
        print(f"   ✅ Demonstration generation completed ({training_results['timing']['demonstration_generation']:.2f}s)")
        print(f"   Generated {len(obs_dataset)} observation-action pairs")

        
        # 6. Data Processing and Augmentation
        print("\n5️⃣ Processing and Augmenting Dataset...")
        data_start_time = time.time()
        
        # Convert to numpy arrays
        X = np.array(obs_dataset)
        y = np.array(action_dataset)
        
        # Add small noise to prevent overfitting (data augmentation)
        X_noise = X + np.random.normal(0, 0.01, X.shape)  # Small Gaussian noise
        X_combined = np.vstack([X, X_noise])
        y_combined = np.vstack([y, y])
        
        # Shuffle the combined dataset
        shuffle_indices = np.random.permutation(len(X_combined))
        X = X_combined[shuffle_indices]
        y = y_combined[shuffle_indices]
        
        print(f"   Dataset size after augmentation: {len(X)} samples")
        
        # Verify observation dimension matches what the model will expect
        print(f"   Observation dimension: {X.shape[1]}")
        print(f"   Environment observation space shape: {env.observation_space.shape[0]}")
        
        training_results["timing"]["data_processing"] = time.time() - data_start_time
        training_results["training_metrics"]["augmented_samples"] = len(X)
        print(f"   ✅ Data processing completed ({training_results['timing']['data_processing']:.2f}s)")
        
        # 7. PPO Model Initialization
        print("\n6️⃣ PPO Model Initialization...")
        model_start_time = time.time()
        
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=log_dir, 
            device=device,
            # Improved hyperparameters for complex locomotion tasks
            learning_rate=3e-4,           # More aggressive learning rate
            n_steps=2048,                 # Larger rollout buffer for better data
            batch_size=32,                # Efficient batch size
            n_epochs=10,                  # More epochs per update
            gamma=0.995,                  # Higher discount for long-term rewards
            gae_lambda=0.95,              # Good GAE parameter for continuous control
            clip_range=0.2,               # Standard clipping
            ent_coef=0.01,                # Encourage exploration
            vf_coef=0.5,                  # Value function coefficient
            max_grad_norm=0.5,            # Gradient clipping for stability
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Larger networks
                activation_fn=torch.nn.ReLU
            )
        )
        
        print(f"   ✅ Model initialized")
        print(f"   Learning rate: 3e-4")
        print(f"   Batch size: 32")
        print(f"   Network architecture: [256, 256] × 2")
        
        training_results["timing"]["model_initialization"] = time.time() - model_start_time
        print(f"   ✅ PPO model initialization completed ({training_results['timing']['model_initialization']:.2f}s)")
        
        # 8. Pre-training Setup
        print("\n7️⃣ Setting Up Pre-Training Components...")
        pretrain_setup_time = time.time()
        
        # Access the policy network directly  
        policy = model.policy
        # Use same learning rate as PPO for consistency
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, weight_decay=1e-5)  # Added weight decay
        loss_fn = torch.nn.MSELoss()
        
        # Convert to PyTorch tensors on CPU explicitly
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        training_results["timing"]["pretrain_setup"] = time.time() - pretrain_setup_time
        print(f"   ✅ Pre-training setup completed ({training_results['timing']['pretrain_setup']:.2f}s)")
        
        # 9. Pre-training Execution
        print("\n8️⃣ Starting Pre-Training with Expert Demonstrations...")
        pretrain_start_time = time.time()
        
        print(f"   Pre-training epochs: {args.pretrain_epochs}")
        print(f"   Batch size: 32")
        print(f"   Training samples: {len(X)}")
        
        # Pre-training loop - reduced epochs to prevent overfitting
        num_epochs = args.pretrain_epochs
        batch_size = 32   # Kept consistent with PPO batch_size
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Create mini-batches
            indices = np.random.permutation(len(X))
            epoch_loss = 0.0
            num_batches = 0
            
            for start_idx in range(0, len(X), batch_size):
                end_idx = min(start_idx + batch_size, len(X))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]
                
                # FIXED: Use the actor part of the policy directly instead of evaluate_actions
                # The policy in SB3 has an actor network that we can access
                features = policy.extract_features(X_batch)
                latent_pi, _ = policy.mlp_extractor(features)
                predicted_actions = policy.action_net(latent_pi)
                
                # Compute loss between expert actions and predicted actions
                loss = loss_fn(predicted_actions, y_batch)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
            
            if epoch % 10 == 0:
                print(f"     Epoch {epoch}: Loss={avg_epoch_loss:.6f}, Best={best_loss:.6f}")
            
            # Early stopping if loss gets very low
            if avg_epoch_loss < 1e-6:
                print(f"     Early stopping at epoch {epoch} - loss converged")
                break
        
        training_results["timing"]["pretraining"] = time.time() - pretrain_start_time
        training_results["training_metrics"]["pretrain_epochs"] = epoch + 1
        training_results["training_metrics"]["final_pretrain_loss"] = avg_epoch_loss
        print(f"   ✅ Pre-training completed ({training_results['timing']['pretraining']:.2f}s)")
        print(f"   Final loss: {avg_epoch_loss:.6f}")
        
        # 10. Save Pre-trained Model
        print("\n9️⃣ Saving Pre-Trained Model...")
        save_pretrain_time = time.time()
        
        pretrained_path = f"models/tensegrity_gait_seeded_before_RL_{timestamp}"
        model.save(pretrained_path)
        
        training_results["timing"]["save_pretrained"] = time.time() - save_pretrain_time
        training_results["pretrained_model_path"] = pretrained_path
        print(f"   ✅ Pre-trained model saved to: {pretrained_path}")
        print(f"   ✅ Save completed ({training_results['timing']['save_pretrained']:.2f}s)")
        
        # 11. RL Training
        print(f"\n🔟 Starting RL Training...")
        rl_start_time = time.time()
        
        print(f"   RL training timesteps: {args.timesteps:,}")
        print(f"   Training started at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Execute RL training
        model.learn(total_timesteps=args.timesteps)
        
        training_results["timing"]["rl_training"] = time.time() - rl_start_time
        print(f"   ✅ RL training completed ({training_results['timing']['rl_training']:.2f}s)")
        
        # 12. Save Final Model
        print("\n1️⃣1️⃣ Saving Final Trained Model...")
        save_final_time = time.time()
        
        final_model_path = f"models/ppo_tensegrity_gait_seeded_{timestamp}"
        model.save(final_model_path)
        
        training_results["timing"]["save_final"] = time.time() - save_final_time
        training_results["final_model_path"] = final_model_path
        print(f"   ✅ Final model saved to: {final_model_path}")
        print(f"   ✅ Save completed ({training_results['timing']['save_final']:.2f}s)")
        
        # 13. Training Success
        total_training_time = time.time() - training_start_time
        training_results["timing"]["total_time"] = total_training_time
        training_results["success"] = True
        
        print(f"\n🎉 PRETRAINING COMPLETED SUCCESSFULLY!")
        print(f"   Total time: {total_training_time:.2f}s ({total_training_time/3600:.2f}h)")
        print(f"   Average pretraining speed: {len(X) * args.pretrain_epochs / training_results['timing']['pretraining']:.1f} samples/s")
        print(f"   Average RL speed: {args.timesteps / training_results['timing']['rl_training']:.1f} timesteps/s")
        
        # 14. Training Results Summary
        print(f"\n{'='*70}")
        print("🏁 PRETRAINING RESULTS SUMMARY")
        print(f"{'='*70}")
        
        print(f"✅ Pretraining completed successfully!")
        print(f"   Device: {device}")
        print(f"   Observation mode: {args.obs_mode} ({actual_obs_dim}-D)")
        print(f"   Demonstration cycles: {args.demo_steps:,}")
        print(f"   Demo samples generated: {training_results['training_metrics']['demo_samples']:,}")
        print(f"   Augmented samples: {training_results['training_metrics']['augmented_samples']:,}")
        print(f"   Pretraining epochs: {training_results['training_metrics']['pretrain_epochs']}")
        print(f"   Final pretraining loss: {training_results['training_metrics']['final_pretrain_loss']:.6f}")
        print(f"   RL timesteps: {args.timesteps:,}")
        print(f"   Total time: {training_results['timing']['total_time']:.2f}s")
        print(f"   Pre-trained model: {training_results['pretrained_model_path']}")
        print(f"   Final model: {training_results['final_model_path']}")
        print(f"   TensorBoard logs: {training_results['log_dir']}")
        
        print(f"\n📊 Timing Breakdown:")
        timing_items = [
            ("Environment setup", "environment_setup"),
            ("Demonstration generation", "demonstration_generation"), 
            ("Data processing", "data_processing"),
            ("Model initialization", "model_initialization"),
            ("Pre-training setup", "pretrain_setup"),
            ("Pre-training execution", "pretraining"),
            ("RL training", "rl_training"),
            ("Model saving", "save_pretrained"),
            ("Final saving", "save_final")
        ]
        
        for name, key in timing_items:
            if key in training_results["timing"]:
                duration = training_results["timing"][key]
                percentage = (duration / total_training_time) * 100
                print(f"   {name:25}: {duration:6.2f}s ({percentage:5.1f}%)")
        
        print(f"\n🎮 To test your trained models:")
        print(f"   python test_trained_model.py --model {training_results['pretrained_model_path']}")
        print(f"   python test_trained_model.py --model {training_results['final_model_path']}")
        
        print(f"\n📊 To view training progress:")
        print(f"   tensorboard --logdir {training_results['log_dir']}")
        
        return training_results
        
    except Exception as e:
        print(f"\n❌ PRETRAINING ERROR: {e}")
        training_results["error"] = str(e)
        return training_results
    finally:
        try:
            env.close()
            if 'model' in locals():
                del model
        except Exception:
            pass
        gc.collect()
        print("🧹 Cleanup completed")

    # Removed legacy duplicate timing/reporting block
