from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from tensegrity_env import TensegrityEnv
import os
from datetime import datetime

def make_env(rank=0, seed=0):
    """Factory function to create environment instances"""
    def _init():
        # Only first environment gets visualization
        visualize = (rank == 0)
        env = TensegrityEnv(obs_dim=78, visualize=visualize)
        env.seed(seed + rank)  # Different seed for each env
        return env
    return _init

def train_parallel():
    # Configuration
    n_envs = 8  # Number of parallel environments
    total_timesteps = 100_000
    
    print(f"Training with {n_envs} parallel environments...")
    
    # Create vectorized environments
    env = make_vec_env(
        make_env, 
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv  # Use separate processes for true parallelism
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        tensorboard_log="./ppo_tensegrity_tensorboard/",
        device="cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    )
    
    # Train the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=f"parallel_run_{timestamp}"
    )
    
    # Save the trained model
    model.save(f"ppo_tensegrity_parallel_{timestamp}")
    print(f"Model saved as ppo_tensegrity_parallel_{timestamp}")
    
    env.close()

if __name__ == "__main__":
    train_parallel()