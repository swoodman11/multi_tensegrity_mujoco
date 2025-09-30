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
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


# Load the trained model
model = PPO.load("ppo_tensegrity_gait_seeded")  
# interesting post physics etc 20250929_134907
# different configuration doesn't experience same physics freakout: ppo_tensegrity_gait_20250930_091454
env = TensegrityEnv(visualize=True)  # Now created with visualization enabled

print("Testing trained model with visualization...")
print("Press 'q' in the render window to quit early")

# Test for 1 episode with rendering
obs, info = env.reset()
total_reward = 0
steps = 0

# Logging containers
actions_log = []          # list of shape (T, num_actuators)
rewards_log = []          # list of shape (T,)
observations_log = []     # list of shape (T, len(OBSERVATION_INDICES))

# Choose which observation indices to plot (customize if desired)
# If None, we'll default to first min(8, obs_dim)
OBSERVATION_INDICES = None

obs_dim = int(np.asarray(obs).shape[0]) if obs is not None else env.sim.obs_dim
num_actuators = getattr(env, 'num_actuators', 12)
if OBSERVATION_INDICES is None:
    OBSERVATION_INDICES = list(range(min(8, obs_dim)))

print(f"Obs dim: {obs_dim}, plotting obs indices: {OBSERVATION_INDICES}")
print(f"Action dim: {num_actuators}")

print(f"\n=== Visualizing Robot Gait ===")

for step in range(1000):  # Max steps per episode
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
        sel = obs_arr[OBSERVATION_INDICES]
        observations_log.append(sel.astype(float).tolist())
    else:
        observations_log.append([float('nan')] * len(OBSERVATION_INDICES))
    
    # Render the robot
    env.render()
    time.sleep(0.05)  # Slow down for better viewing
    
    # Print progress every 50 steps
    if step % 50 == 0:
        print(f"Step {step}: Reward = {reward:.3f}, Total = {total_reward:.2f}")
    
    if done or truncated:
        print("Episode ended!")
        break

print(f"Final: {steps} steps, Total reward: {total_reward:.2f}")

# Keep window open for a bit
time.sleep(2)

# ----- Post-run: Save plots and CSVs -----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_dir = os.path.join("sim_output", f"eval_{timestamp}")
# os.makedirs(output_dir, exist_ok=True)

actions_np = np.asarray(actions_log, dtype=float) if len(actions_log) else np.zeros((0, num_actuators))
rewards_np = np.asarray(rewards_log, dtype=float) if len(rewards_log) else np.zeros((0,))
obs_np = np.asarray(observations_log, dtype=float) if len(observations_log) else np.zeros((0, len(OBSERVATION_INDICES)))

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
    # Selected observations plot
    plt.figure(figsize=(12, 4))
    t = np.arange(obs_np.shape[0])
    for j in range(obs_np.shape[1]):
        plt.plot(t, obs_np[:, j], label=f"obs[{OBSERVATION_INDICES[j]}]")
    plt.title("Selected observations over time")
    plt.xlabel("step")
    plt.ylabel("value")
    plt.grid(True, alpha=0.3)
    if obs_np.shape[1] <= 12:
        plt.legend(ncol=3)
    plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "observations.png"), dpi=200)
    plt.show()
    plt.close()

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
except:
    pass

#Zac: run this for tensor flow charts: tensorboard --logdir=./ppo_tensegrity_tensorboard/ --port=6007