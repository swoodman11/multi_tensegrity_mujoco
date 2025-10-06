import time
import gc
import torch
import numpy as np
import psutil
from stable_baselines3 import PPO
from tensegrity_env import TensegrityEnv
from datetime import datetime
from pathlib import Path
import argparse

def debug_print(message, filename="train.py", debug_enabled=False):
    """Print debug messages with filename prefix if debug is enabled"""
    if debug_enabled:
        print(f"DEBUG {filename}: {message}")

def check_system_requirements():
    """
    System check for CPU training requirements
    
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
    min_system_ram = 8.0  # Minimum 8GB RAM for CPU training
    
    print(f"✅ System RAM: {system_ram_gb:.1f}GB")
    print(f"✅ PyTorch Version: {torch.__version__}")
    print("ℹ️  Using CPU for training")
    
    # Requirements validation
    if system_ram_gb < min_system_ram:
        print(f"❌ Insufficient system RAM: {system_ram_gb:.1f}GB < {min_system_ram}GB required")
        return False, "cpu", (0.0, system_ram_gb)
    
    # Check current RAM usage
    memory_info = psutil.virtual_memory()
    available_ram = memory_info.available / 1e9
    
    print(f"📊 Available RAM: ~{available_ram:.1f}GB")
    print("🚀 System ready for CPU training!")
    
    return True, "cpu", (0.0, system_ram_gb)

def main():
    """
    Main training pipeline with comprehensive progress tracking
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train tensegrity model with progress tracking")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--obs_mode", type=str, default="tier2", choices=["tier2", "legacy104"], 
                       help="Observation mode: tier2 (96-D) or legacy104 (104-D)")
    args = parser.parse_args()
    
    print("🤖 PPO Training for Tensegrity Robot Locomotion")
    print("=" * 60)
    print("This script trains a locomotion policy using PPO algorithm.")
    print()
    
    training_start_time = time.time()
    training_results = {
        "success": False,
        "timing": {},
        "system_metrics": {}
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
        print(f"   Total timesteps: {args.timesteps:,}")
        print(f"   Debug mode: {args.debug}")
        
        # 2. Environment Setup
        print("\n1️⃣ Environment Setup...")
        env_start_time = time.time()
        
        env = TensegrityEnv(obs_mode=args.obs_mode, debug_enabled=args.debug)
        debug_print("Using Tier-2 observation mode (96-D)", "train.py", args.debug)
        
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
        
        # 5. PPO Model Initialization
        print("\n4️⃣ PPO Model Initialization...")
        model_start_time = time.time()
        
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=log_dir,
            device=device
        )
        
        # Monitor memory after model creation
        print(f"   ✅ Model initialized on CPU")
        
        training_results["timing"]["model_initialization"] = time.time() - model_start_time
        print(f"   ✅ PPO model initialization completed ({training_results['timing']['model_initialization']:.2f}s)")
        
        # 6. Training Execution
        print(f"\n5️⃣ Starting PPO Training...")
        training_exec_time = time.time()

        # Create save directory (standardized to 'models')
        save_dir = Path("models")
        save_dir.mkdir(exist_ok=True)
        
        print(f"   Training will save models to: {save_dir}")
        print(f"   Monitor progress with: tensorboard --logdir {log_dir}")
        print(f"   Training started at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Execute training
        model.learn(total_timesteps=args.timesteps)
        
        training_results["timing"]["training_execution"] = time.time() - training_exec_time
        print(f"   ✅ Training completed ({training_results['timing']['training_execution']:.2f}s)")
        
        # 7. Model Saving
        print("\n6️⃣ Saving Trained Model...")
        save_start_time = time.time()
        
        model_filename = f"ppo_tensegrity_gait_{timestamp}"
        final_model_path = save_dir / model_filename
        
        model.save(final_model_path)
        
        training_results["timing"]["model_saving"] = time.time() - save_start_time
        training_results["model_path"] = str(final_model_path)
        
        print(f"   ✅ Model saved to: {final_model_path}")
        print(f"   ✅ Model saving completed ({training_results['timing']['model_saving']:.2f}s)")
        
        # 8. Final Memory Check
        final_memory_info = psutil.virtual_memory()
        final_available_ram = final_memory_info.available / 1e9
        
        print(f"\n📊 Memory Summary:")
        print(f"   Final available RAM: {final_available_ram:.2f}GB")
        print(f"   Memory efficiency: Good (CPU training)")
        
        training_results["system_metrics"]["final_available_ram"] = final_available_ram
        
        # 9. Training Success
        total_training_time = time.time() - training_start_time
        training_results["timing"]["total_time"] = total_training_time
        training_results["success"] = True
        
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"   Total time: {total_training_time:.2f}s ({total_training_time/3600:.2f}h)")
        print(f"   Average FPS: {args.timesteps / training_results['timing']['training_execution']:.1f}")
        
        # 10. Training Results Summary
        print(f"\n{'='*60}")
        print("🏁 TRAINING RESULTS SUMMARY")
        print(f"{'='*60}")
        
        print(f"✅ Training completed successfully!")
        print(f"   Device: {device}")
        print(f"   Observation mode: {args.obs_mode} ({actual_obs_dim}-D)")
        print(f"   Timesteps: {args.timesteps:,}")
        print(f"   Total time: {training_results['timing']['total_time']:.2f}s")
        print(f"   Model saved: {training_results['model_path']}")
        print(f"   TensorBoard logs: {training_results['log_dir']}")
        
        print(f"\n🎮 To test your trained model:")
        print(f"   python test_trained_model.py --model {training_results['model_path']}")
        
        print(f"\n📊 To view training progress:")
        print(f"   tensorboard --logdir {training_results['log_dir']}")
        
        return training_results
        
    except Exception as e:
        print(f"\n❌ TRAINING ERROR: {e}")
        training_results["error"] = str(e)
        return training_results
    
    finally:
        # Cleanup
        try:
            env.close()
            if 'model' in locals():
                del model
        except:
            pass
        
        # Memory cleanup
        gc.collect()
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    main()
