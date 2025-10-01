from stable_baselines3 import PPO
from tensegrity_env import TensegrityEnv
from datetime import datetime
import argparse

def debug_print(message, filename="train.py", debug_enabled=False):
    """Print debug messages with filename prefix if debug is enabled"""
    if debug_enabled:
        print(f"DEBUG {filename}: {message}")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train tensegrity model")
parser.add_argument("--debug", action="store_true", help="Enable debug output")
args = parser.parse_args()

# Use Tier-2 observation (96-D). For legacy 104-D, pass obs_mode="legacy104".
env = TensegrityEnv(obs_mode="tier2", debug_enabled=args.debug)
debug_print("Using Tier-2 observation mode (96-D)", "train.py", args.debug)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensegrity_tensorboard/",device='cpu')
model.learn(total_timesteps=100_000)

# Generate timestamp for unique model filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"ppo_tensegrity_gait_{timestamp}"

model.save(f"trained_models/{model_filename}")

print(f"Model saved as: {model_filename}.zip")
