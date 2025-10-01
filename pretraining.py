from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from tensegrity_env import TensegrityEnv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime

# Force CPU usage since the warning indicates PPO works better on CPU with MLP policy
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Define your rolling gait actions from 3bar_gaits.py
# Based on one of the gaits that causes rolling
# roll_sequence = np.array([
#     [1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1],  # Step 1


#     [0.0, 1.0, 1.0, 0.0, 1.0, 0.1, 0.0, 1.0, 1.0, 0.0, 1.0, 0.1],  # Step 2

#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 3
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 3
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 3
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 3
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 3
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 3

#     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],

#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],

#     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 0.1, 1.0, 1.0],
#     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 0.1, 1.0, 1.0],
#     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 0.1, 1.0, 1.0],
#     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 0.1, 1.0, 1.0],
#     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 0.1, 1.0, 1.0],
#     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 0.1, 1.0, 1.0],

#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],


    
#     # Add the rest of your gait sequence here
#     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0],  # Step 4
#     [1.0, 1.0, 0.0, 1.0, 0.1, 0.0, 1.0, 1.0, 0.0, 1.0, 0.1, 0.0],  # Step 5
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 6
#     # Additional steps as needed
#     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0],  # Step 4
#     [1.0, 1.0, 0.0, 1.0, 0.1, 0.0, 1.0, 1.0, 0.0, 1.0, 0.1, 0.0],  # Step 5
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # Step 6

#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 4
#     [1.0, 1.0, 0.0, 1.0, 0.1, 0.0, 1.0, 1.0, 0.0, 1.0, 0.1, 0.0],  # Step 5
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# ])

roll_sequence = np.array([
        [1.0, 1.0, 0.1, 1.0, 1.0, 0.1,   0.1, 0.8, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.8, 0.1,   1.0, 0.1, 1.0, 1.0, 0.1, 1.0],
        [1.0, 0.1, 1.0, 1.0, 0.1, 1.0,   0.0, 0.1, 0.8, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.8, 0.1, 0.0,   1.0, 1.0, 0.1, 1.0, 1.0, 0.1],
        [0.1, 1.0, 1.0, 0.1, 1.0, 1.0,   0.8, 0.0, 0.1, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.1, 0.0, 0.8,   0.1, 1.0, 1.0, 0.1, 1.0, 1.0]
    ])

# Create environment
env = TensegrityEnv(obs_mode="tier2", visualize=False)

# Create a dataset of observations and expert actions
obs_dataset = []
action_dataset = []

obs, _ = env.reset()
for _step in range(5):  # Run for enough steps to cover the gait sequence multiple times
    for action in roll_sequence:
        # Store the current observation and the expert action
        obs_dataset.append(obs)
        action_dataset.append(action)
        
        # Take step in environment with expert action
        obs, _, done, _, _ = env.step(action)
        
        if done:
            obs, _ = env.reset()

# Convert to numpy arrays
X = np.array(obs_dataset)
y = np.array(action_dataset)

# Verify observation dimension matches what the model will expect
print(f"Observation dimension: {X.shape[1]}")
print(f"Environment observation space shape: {env.observation_space.shape[0]}")

# Initialize the PPO model with explicit device=cpu
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensegrity_tensorboard/", device="cpu")

# Pre-train the policy network using your expert demonstrations
# Access the policy network directly
policy = model.policy
optimizer = torch.optim.Adam(policy.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()

# Convert to PyTorch tensors on CPU explicitly
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Pre-training loop
print("Pre-training policy network with expert demonstrations...")
num_epochs = 100
batch_size = 32
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
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {avg_epoch_loss:.6f}")

# Now continue with regular RL training
print("Starting RL training...")
model.learn(total_timesteps=100_000)

# Save the model with a timestamp to prevent overwriting files

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"ppo_tensegrity_gait_seeded_{timestamp}")