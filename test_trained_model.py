# from stable_baselines3 import PPO
# from tensegrity_env import TensegrityEnv
# import time

# # Load the trained model
# model = PPO.load("ppo_tensegrity_gait")
# env = TensegrityEnv()

# # Test for 3 episodes
# for episode in range(3):
#     obs, info = env.reset()
#     total_reward = 0
#     steps = 0
    
#     print(f"\n=== Episode {episode + 1} ===")
    
#     for step in range(1000):  # Max steps per episode  #tensorboard --logdir=./ppo_tensegrity_tensorboard/ --port=6007
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, truncated, info = env.step(action)
        
#         total_reward += reward
#         steps += 1
        
#         # Render if you want to see it (might be slow)
#         # env.render()
#         # time.sleep(0.01)
        
#         if done or truncated:
#             break
    
#     print(f"Episode {episode + 1}: {steps} steps, Total reward: {total_reward:.2f}")

# # env.close()

from stable_baselines3 import PPO
from tensegrity_env import TensegrityEnv
import time
import os
import glob
import argparse
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def find_most_recent_model(models_dir="models", pattern="*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].zip"):
    """Find the most recent trained model based on timestamp in filename.
    
    Default pattern matches any file ending with timestamp format: YYYYMMDD_HHMMSS.zip
    This covers models from train.py, pretraining.py, train_parallel.py, etc.
    """
    # First try the timestamp pattern
    pattern_path = os.path.join(models_dir, pattern)
    model_files = glob.glob(pattern_path)
    
    # Alternative: use regex to find any .zip file with a timestamp pattern
    if not model_files:
        print("Timestamp pattern not found, searching for any files with timestamp format...")
        all_zip_files = glob.glob(os.path.join(models_dir, "*.zip"))
        timestamp_pattern = re.compile(r'.*_(\d{8}_\d{6})\.zip$')
        
        model_files = []
        for file in all_zip_files:
            if timestamp_pattern.match(file):
                model_files.append(file)
    
    if not model_files:
        # Fallback to old pattern for backwards compatibility
        old_pattern = "ppo_tensegrity_gait_*.zip"
        old_pattern_path = os.path.join(models_dir, old_pattern)
        model_files = glob.glob(old_pattern_path)
        
        if not model_files:
            raise FileNotFoundError(f"No models found matching timestamp pattern: {pattern_path} or fallback pattern: {old_pattern_path}")
        else:
            print(f"Using fallback pattern: {old_pattern}")
    
    # Sort by modification time (most recent first)
    model_files.sort(key=os.path.getmtime, reverse=True)
    
    # Remove the .zip extension for PPO.load()
    most_recent = model_files[0]
    model_path = os.path.splitext(most_recent)[0]
    
    print(f"Found {len(model_files)} model(s), using most recent: {os.path.basename(most_recent)}")
    return model_path


def create_observation_plots(obs_np, obs_mode, timestamp, save_plots=False):
    """
    Create comprehensive observation plots based on observation mode.
    
    Args:
        obs_np: numpy array of shape (timesteps, obs_dim)
        obs_mode: "tier2" or "legacy104"
        timestamp: string timestamp for filenames
        save_plots: whether to save plots as PNG files
    """
    print(f"Creating observation plots for {obs_mode} mode...")
    print(f"Observation shape: {obs_np.shape}")
    
    if obs_mode == "tier2":
        create_tier2_observation_plots(obs_np, timestamp, save_plots)
    else:  # legacy104 or other
        create_legacy_observation_plots(obs_np, timestamp, save_plots)


def create_tier2_observation_plots(obs_np, timestamp, save_plots=False):
    """
    Create plots for Tier2 96D observation mode.
    
    Tier2 structure (96D total):
    - Cable lengths normalized (12D): 0-11
    - Cable rates normalized (12D): 12-23  
    - Previous action (12D): 24-35
    - Strain extensions (18D): 36-53
    - IMU gravity vectors (18D): 54-71
    - IMU angular velocities (18D): 72-89
    - COM linear velocity (3D): 90-92
    - COM angular velocity (3D): 93-95
    """
    
    # Define observation groups for Tier2
    groups = {
        "Cable Lengths (Normalized)": (0, 12),
        "Cable Length Rates": (12, 24),  
        "Previous Actions": (24, 36),
        "Strain Extensions": (36, 54),
        "IMU Gravity Vectors": (54, 72),
        "IMU Angular Velocities": (72, 90),
        "COM Linear Velocity": (90, 93),
        "COM Angular Velocity": (93, 96)
    }
    
    t = np.arange(obs_np.shape[0])
    
    # Create comprehensive figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Plot each group
    for i, (group_name, (start, end)) in enumerate(groups.items(), 1):
        if end > obs_np.shape[1]:
            print(f"Warning: {group_name} indices {start}-{end} exceed observation dimension {obs_np.shape[1]}")
            continue
            
        ax = plt.subplot(4, 2, i)
        
        group_data = obs_np[:, start:end]
        group_size = end - start
        
        # Use different plotting strategies based on group size
        if group_size <= 3:
            # Plot individual lines for small groups
            labels = []
            if "COM" in group_name:
                if "Linear" in group_name:
                    labels = ['X', 'Y', 'Z']
                else:  # Angular
                    labels = ['Roll', 'Pitch', 'Yaw']
            else:
                labels = [f'{i}' for i in range(group_size)]
                
            for j in range(group_size):
                ax.plot(t, group_data[:, j], label=labels[j] if j < len(labels) else f'{j}', 
                       linewidth=1.5, alpha=0.8)
            ax.legend(fontsize=8)
            
        elif group_size <= 12:
            # Plot individual lines for medium groups (cables, actions)
            colors = plt.cm.tab10(np.linspace(0, 1, min(10, group_size)))
            for j in range(group_size):
                color = colors[j % len(colors)]
                ax.plot(t, group_data[:, j], label=f'{j+1}', color=color, 
                       linewidth=1.2, alpha=0.7)
            
            # Create compact legend
            if group_size <= 6:
                ax.legend(fontsize=8, ncol=2)
            else:
                ax.legend(fontsize=7, ncol=3)
                
        else:
            # For large groups, show mean ± std and extremes
            mean_val = np.mean(group_data, axis=1)
            std_val = np.std(group_data, axis=1)
            min_val = np.min(group_data, axis=1)
            max_val = np.max(group_data, axis=1)
            
            ax.plot(t, mean_val, 'b-', linewidth=2, label='Mean', alpha=0.8)
            ax.fill_between(t, mean_val - std_val, mean_val + std_val, 
                           alpha=0.3, color='blue', label='±1 Std')
            ax.plot(t, min_val, 'r--', linewidth=1, alpha=0.6, label='Min')
            ax.plot(t, max_val, 'g--', linewidth=1, alpha=0.6, label='Max')
            ax.legend(fontsize=8)
        
        ax.set_title(f"{group_name} (dim {start}-{end-1})", fontsize=10, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.grid(True, alpha=0.3)
        
        # Add range information in corner
        data_min, data_max = np.min(group_data), np.max(group_data)
        ax.text(0.02, 0.98, f'Range: [{data_min:.3f}, {data_max:.3f}]', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.suptitle('Tier2 Observation Analysis (96D)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save plot only if requested
    if save_plots:
        filename = f"tier2_observations_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved Tier2 observation plot: {filename}")
    plt.show()
    plt.close()
    
    # Create detailed cable analysis
    create_detailed_cable_plots(obs_np, timestamp, save_plots, obs_mode="tier2")


def create_legacy_observation_plots(obs_np, timestamp, save_plots=False):
    """
    Create plots for Legacy 104D observation mode.
    
    Legacy structure (104D total):
    - Robot state (42D): positions + orientations of 6 bodies (6 × 7)
    - Cable state (24D): 12 cables × (length + velocity)
    - End effector positions (18D): 6 endpoints × 3D
    - Padding (20D): unused dimensions
    """
    
    groups = {
        "Robot Body States": (0, 42),
        "Cable Lengths & Velocities": (42, 66),
        "End Effector Positions": (66, 84),
        "Padding/Unused": (84, 104)
    }
    
    t = np.arange(obs_np.shape[0])
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    
    for i, (group_name, (start, end)) in enumerate(groups.items(), 1):
        if end > obs_np.shape[1]:
            print(f"Warning: {group_name} indices {start}-{end} exceed observation dimension {obs_np.shape[1]}")
            continue
            
        ax = plt.subplot(2, 2, i)
        group_data = obs_np[:, start:end]
        
        if "Padding" in group_name:
            # For padding, just show if there's any non-zero data
            non_zero_count = np.count_nonzero(group_data)
            ax.text(0.5, 0.5, f"Padding Dimensions\n{non_zero_count}/{group_data.size} non-zero values", 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            # Show mean ± std for large groups
            mean_val = np.mean(group_data, axis=1)
            std_val = np.std(group_data, axis=1)
            min_val = np.min(group_data, axis=1)
            max_val = np.max(group_data, axis=1)
            
            ax.plot(t, mean_val, 'b-', linewidth=2, label='Mean')
            ax.fill_between(t, mean_val - std_val, mean_val + std_val, 
                           alpha=0.3, color='blue', label='±1 Std')
            ax.plot(t, min_val, 'r--', linewidth=1, alpha=0.6, label='Min')
            ax.plot(t, max_val, 'g--', linewidth=1, alpha=0.6, label='Max')
            ax.legend(fontsize=8)
        
        ax.set_title(f"{group_name} (dim {start}-{end-1})", fontsize=10, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Legacy Observation Analysis (104D)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save plot only if requested
    if save_plots:
        filename = f"legacy_observations_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved Legacy observation plot: {filename}")
    plt.show()
    plt.close()
    
    # Create detailed cable analysis for legacy mode
    create_detailed_cable_plots(obs_np, timestamp, save_plots, obs_mode="legacy")


def create_detailed_cable_plots(obs_np, timestamp, save_plots=False, obs_mode="tier2"):
    """Create detailed analysis of cable-related observations."""
    
    if obs_mode == "tier2":
        # Tier2: Cable lengths (0-11), Cable rates (12-23), Previous actions (24-35)
        cable_lengths = obs_np[:, 0:12]
        cable_rates = obs_np[:, 12:24] 
        prev_actions = obs_np[:, 24:36]
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        t = np.arange(obs_np.shape[0])
        
        # Cable lengths
        ax = axes[0]
        for i in range(12):
            ax.plot(t, cable_lengths[:, i], label=f'Cable {i+1}', linewidth=1.2, alpha=0.8)
        ax.set_title('Cable Lengths (Normalized)', fontweight='bold')
        ax.set_ylabel('Normalized Length')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Cable rates  
        ax = axes[1]
        for i in range(12):
            ax.plot(t, cable_rates[:, i], label=f'Cable {i+1}', linewidth=1.2, alpha=0.8)
        ax.set_title('Cable Length Rates (Normalized)', fontweight='bold')
        ax.set_ylabel('Normalized Rate')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Previous actions
        ax = axes[2] 
        for i in range(12):
            ax.plot(t, prev_actions[:, i], label=f'Action {i+1}', linewidth=1.2, alpha=0.8)
        ax.set_title('Previous Actions', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
    else:  # legacy mode
        # Legacy: Cable state is in positions 42-66 (12 cables × 2 values each)
        cable_data = obs_np[:, 42:66]
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        t = np.arange(obs_np.shape[0])
        
        # Cable lengths (even indices: 0, 2, 4, ...)
        ax = axes[0]
        for i in range(12):
            if 2*i < cable_data.shape[1]:
                ax.plot(t, cable_data[:, 2*i], label=f'Cable {i+1}', linewidth=1.2, alpha=0.8)
        ax.set_title('Cable Lengths', fontweight='bold')
        ax.set_ylabel('Length (m)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Cable velocities (odd indices: 1, 3, 5, ...)
        ax = axes[1]
        for i in range(12):
            if 2*i+1 < cable_data.shape[1]:
                ax.plot(t, cable_data[:, 2*i+1], label=f'Cable {i+1}', linewidth=1.2, alpha=0.8)
        ax.set_title('Cable Velocities', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Velocity (m/s)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Detailed Cable Analysis ({obs_mode.upper()} mode)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save plot only if requested
    if save_plots:
        filename = f"cable_details_{obs_mode}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved detailed cable plot: {filename}")
    plt.show()
    plt.close()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test trained tensegrity model")
parser.add_argument("--model", type=str, default=None, 
                   help="Specific model path to load (without .zip extension). If not provided, uses most recent model.")
parser.add_argument("--no-vis", action="store_true", 
                   help="Disable visualization")
parser.add_argument("--save-plots", action="store_true", 
                   help="Save observation and analysis plots as PNG files (disabled by default)")
parser.add_argument("--save-video", action="store_true", 
                   help="Save simulation as video file (disabled by default)")
args = parser.parse_args()

# Determine which model to load
if args.model:
    model_path = args.model
    print(f"Loading specified model: {model_path}")
else:
    model_path = find_most_recent_model()

# Load the trained model
model = PPO.load(model_path)
# interesting post physics etc 20250929_134907
# different configuration doesn't experience same physics freakout: ppo_tensegrity_gait_20250930_091454
env = TensegrityEnv(visualize=not args.no_vis)  # Visualization controlled by command line flag

print(f"Testing trained model: {os.path.basename(model_path)}")
print("Testing trained model with visualization..." if not args.no_vis else "Testing trained model without visualization...")
if not args.no_vis:
    print("Press 'q' in the render window to quit early")

# --- Video saving setup ---
video_frames = []
if args.save_video:
    print("Video saving enabled - frames will be collected during simulation")
    # Create output directory
    output_dir = Path('sim_output')
    output_dir.mkdir(exist_ok=True)
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"test_model_{timestamp}.mp4"
    video_path = output_dir / video_filename
else:
    print("Video saving disabled")
# --- End video saving setup ---

# Test for 1 episode with rendering
obs, info = env.reset()
total_reward = 0
steps = 0

# Logging containers
actions_log = []          # list of shape (T, num_actuators)
rewards_log = []          # list of shape (T,)
observations_log = []     # list of shape (T, obs_dim) - now storing full observations

# Store full observations instead of just selected indices
obs_dim = int(np.asarray(obs).shape[0]) if obs is not None else env.sim.obs_dim
num_actuators = getattr(env, 'num_actuators', 12)

print(f"Obs dim: {obs_dim}, storing full observation vector")
print(f"Action dim: {num_actuators}")
print(f"Observation mode: {env.sim.obs_mode}")

print(f"\n=== Visualizing Robot Gait ===")

for step in range(100):  # Max steps per episode (matches new env length)
    # mujoco.mj_step(model, data)
    action, _ = model.predict(obs, deterministic=True)
    # Ensure action shape is (num_actuators,)
    action = np.asarray(action).reshape(-1)
    if action.shape[0] != num_actuators:
        # Try to squeeze to correct size if batched
        action = action[:num_actuators]

    obs, reward, done, truncated, info = env.step(action)
    
    total_reward += reward
    steps += 1

    # Log after stepping (log the action we applied and the new obs/reward)
    actions_log.append(action.astype(float).tolist())
    rewards_log.append(float(reward))
    if obs is not None:
        obs_arr = np.asarray(obs).reshape(-1)
        observations_log.append(obs_arr.astype(float).tolist())  # Store full observation
    else:
        observations_log.append([float('nan')] * obs_dim)
    
    # Rendering is now handled internally by the simulator when visualize=True
    # No external env.render() or time.sleep() needed
    
    # Collect frames for video saving (if enabled)
    if args.save_video:
        # Use offscreen renderer to get frames even when visualization is disabled
        frame = env.sim.render_frame()
        if frame is not None:
            video_frames.append(frame)
    
    # Print progress every 50 steps
    if step % 50 == 0:
        print(f"Step {step}: Reward = {reward:.3f}, Total = {total_reward:.2f}")
    
    if done or truncated:
        print("Episode ended!")
        break

print(f"Final: {steps} steps, Total reward: {total_reward:.2f}")

# Keep window open for a bit (only if visualization is enabled)
if not args.no_vis:
    time.sleep(2)

# Save video if enabled and frames were collected
if args.save_video and video_frames:
    print(f"Saving video with {len(video_frames)} frames...")
    env.sim.save_video(video_path, video_frames)
    print(f"Video saved to: {video_path}")
elif args.save_video:
    print("Warning: Video saving was enabled but no frames were collected")

# ----- Post-run: Save plots and CSVs -----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_dir = os.path.join("sim_output", f"eval_{timestamp}")
# os.makedirs(output_dir, exist_ok=True)

actions_np = np.asarray(actions_log, dtype=float) if len(actions_log) else np.zeros((0, num_actuators))
rewards_np = np.asarray(rewards_log, dtype=float) if len(rewards_log) else np.zeros((0,))
obs_np = np.asarray(observations_log, dtype=float) if len(observations_log) else np.zeros((0, obs_dim))

# Save CSVs for reproducibility
# np.savetxt(os.path.join(output_dir, "actions.csv"), actions_np, delimiter=",", fmt="%.6f")
# np.savetxt(os.path.join(output_dir, "rewards.csv"), rewards_np.reshape(-1, 1), delimiter=",", fmt="%.6f")
# np.savetxt(os.path.join(output_dir, "selected_observations.csv"), obs_np, delimiter=",", fmt="%.6f")

if actions_np.shape[0] > 0:
    # Actions plot: 3x4 subplots if 12 actuators, else a single plot overlay
    if num_actuators == 12:
        fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey=True)
        axes = axes.ravel()
        t = np.arange(actions_np.shape[0])
        for i in range(num_actuators):
            axes[i].plot(t, actions_np[:, i], lw=1.5)
            axes[i].set_title(f"Actuator {i}")
            axes[i].set_ylim(-0.1, 1.1)
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.suptitle("Actions per actuator")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # fig.savefig(os.path.join(output_dir, "actions.png"), dpi=200)
        plt.show()
        plt.close(fig)

if obs_np.shape[0] > 0 and obs_np.shape[1] > 0:
    # Create comprehensive observation plots based on observation mode
    create_observation_plots(obs_np, env.sim.obs_mode, timestamp, args.save_plots)

if rewards_np.shape[0] > 0:
    # Reward plots: instantaneous and cumulative
    t = np.arange(rewards_np.shape[0])
    cum = np.cumsum(rewards_np)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, rewards_np, color='tab:blue')
    axes[0].set_title("Reward per step")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, cum, color='tab:green')
    axes[1].set_title("Cumulative reward")
    axes[1].set_xlabel("step")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    # fig.savefig(os.path.join(output_dir, "rewards.png"), dpi=200)
    plt.show()
    plt.close(fig)

# print(f"Saved plots to {output_dir}")

# Close properly
try:
    import cv2
    cv2.destroyAllWindows()
except ImportError:
    pass  # cv2 not available
except Exception:
    pass

#Zac: run this for tensor flow charts: tensorboard --logdir=./ppo_tensegrity_tensorboard/ --port=6007