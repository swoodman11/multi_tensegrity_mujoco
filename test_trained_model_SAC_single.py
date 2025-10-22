"""Test trained SAC models for single tensegrity robot.

Loads a trained model and runs it in the environment with visualization.
"""

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from single_tensegrity_env import SingleTensegrityEnv
import time


def test_trained_model(model_path: str, num_timesteps: int = 2000, 
                       visualize: bool = True, save_frames: bool = False):
    """Test a trained SAC model on single tensegrity.
    
    Parameters
    ----------
    model_path : str
        Path to saved model
    num_timesteps : int, default=500
        Number of timesteps to run (single episode)
    visualize : bool, default=True
        Whether to enable visualization
    save_frames : bool, default=False
        Whether to save visualization frames
    """
    print("🤖 Testing Single Tensegrity Trained Model")
    print("=" * 60)
    
    # Load model (detect device automatically)
    print(f"\n📂 Loading model from: {model_path}")
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {device.upper()}")
        model = SAC.load(model_path, device=device)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Statistics tracking
    episode_reward = 0
    episode_length = 0
    
    # Run single episode
    print(f"\n🎮 Running single episode for {num_timesteps} timesteps...")
    print("=" * 60)
    
    # Create environment
    print(f"\n📍 Creating environment...")
    print("-" * 40)
    env = SingleTensegrityEnv(visualize=visualize)
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    
    obs, info = env.reset()
    
    for step in range(num_timesteps):
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute action (sim_step handles internal rendering if visualize=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Print step info every 10 steps
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}: reward={reward:.3f}, cumulative={episode_reward:.2f}")
            
            # Check if episode ended
            if terminated or truncated:
                print(f"  Episode ended early at step {episode_length}")
                break
    
    print(f"✅ Episode complete:")
    print(f"   Total reward: {episode_reward:.2f}")
    print(f"   Episode length: {episode_length} timesteps")
    
    # Print reward breakdown if available
    if 'endpoint_height_reward' in info:
        print(f"   Reward components:")
        reward_keys = [
            'endpoint_height_reward', 'lifted_centroid_xy_reward',
            'grounded_centroid_xy_reward', 'com_step_progress',
            'lifted_swing_reward', 'stall_penalty'
        ]
        for key in reward_keys:
            if key in info:
                print(f"     {key}: {info[key]:.3f}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("📊 TESTING SUMMARY")
    print("=" * 60)
    print(f"Timesteps completed: {episode_length}/{num_timesteps}")
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Average reward per step: {episode_reward/max(episode_length, 1):.3f}")
    
    # Close environment
    env.close()
    print("\n✅ Testing complete!")


def find_latest_model(config_name: str = None) -> Path:
    """Find the latest trained model.
    
    Parameters
    ----------
    config_name : str, optional
        Configuration name to filter by
    
    Returns
    -------
    model_path : Path
        Path to latest model
    """
    # Search for single tensegrity models (both GPU and CPU)
    if config_name:
        patterns = [
            f"sac_single_gpu_pretraining_{config_name}_*",
            f"sac_single_cpu_pretraining_{config_name}_*"
        ]
    else:
        patterns = [
            "sac_single_gpu_pretraining_*",
            "sac_single_cpu_pretraining_*"
        ]
    
    models = []
    
    # Check GPU models directory
    gpu_models_dir = Path("models/gpu")
    if gpu_models_dir.exists():
        for pattern in patterns:
            models.extend(gpu_models_dir.glob(pattern + ".zip"))
    
    # Check CPU models directory
    cpu_models_dir = Path("models/cpu")
    if cpu_models_dir.exists():
        for pattern in patterns:
            models.extend(cpu_models_dir.glob(pattern + ".zip"))
    
    if not models:
        raise FileNotFoundError(f"No models found matching patterns: {patterns}")
    
    # Sort by modification time and get latest
    latest = max(models, key=lambda p: p.stat().st_mtime)
    return latest


def main():
    """Main entry point for testing."""
    parser = argparse.ArgumentParser(description="Test trained single tensegrity SAC model")
    parser.add_argument("--model", type=str, help="Path to model file (if not provided, uses latest)")
    parser.add_argument("--config", type=str, help="Config name to find latest model")
    parser.add_argument("--timesteps", type=int, default=50, help="Number of timesteps to run (default: 50, equals 50 seconds)")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--save-frames", action="store_true", help="Save visualization frames")
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists() and not str(model_path).endswith('.zip'):
            model_path = Path(str(model_path) + '.zip')
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return
    else:
        try:
            model_path = find_latest_model(args.config)
            print(f"🔍 Using latest model: {model_path.name}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("\nAvailable models:")
            models_dir = Path("models/gpu")
            if models_dir.exists():
                for model in sorted(models_dir.glob("sac_single_*.zip")):
                    print(f"  - {model.name}")
            return
    
    # Run testing
    test_trained_model(
        str(model_path),
        num_timesteps=args.timesteps,
        visualize=not args.no_viz,
        save_frames=args.save_frames
    )


if __name__ == "__main__":
    main()
