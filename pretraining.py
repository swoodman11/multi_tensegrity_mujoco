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

# roll_sequence = np.array([
#         [1.0, 1.0, 0.1, 1.0, 1.0, 0.1,   0.1, 0.8, 0.0, 1.0, 1.0, 0.0],
#         [0.0, 1.0, 1.0, 0.0, 0.8, 0.1,   1.0, 0.1, 1.0, 1.0, 0.1, 1.0],
#         [1.0, 0.1, 1.0, 1.0, 0.1, 1.0,   0.0, 0.1, 0.8, 0.0, 1.0, 1.0],
#         [1.0, 1.0, 0.0, 0.8, 0.1, 0.0,   1.0, 1.0, 0.1, 1.0, 1.0, 0.1],
#         [0.1, 1.0, 1.0, 0.1, 1.0, 1.0,   0.8, 0.0, 0.1, 1.0, 0.0, 1.0],
#         [1.0, 0.0, 1.0, 0.1, 0.0, 0.8,   0.1, 1.0, 1.0, 0.1, 1.0, 1.0]
#     ])

# Shuffling gait for dual tensegrity robot
# Strategy: Alternate contraction patterns between tensegrities to create shuffling motion
# First 6 actuators control first tensegrity, last 6 control second tensegrity
# roll_sequence = np.array([
#     # Step 1: Both tensegrities at rest position
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    
#     # Step 2: First tensegrity contracts front cables, second stays extended
#     [0.3, 0.3, 1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    
#     # Step 3: First tensegrity contracts more, second starts to contract rear
#     [0.2, 0.2, 0.8, 0.8, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0, 0.3, 0.3],
    
#     # Step 4: Transition - first extends rear, second contracts front
#     [0.2, 0.2, 1.0, 1.0, 0.3, 0.3,   0.3, 0.3, 1.0, 1.0, 0.2, 0.2],
    
#     # Step 5: First extends, second fully contracts front
#     [0.8, 0.8, 1.0, 1.0, 0.8, 0.8,   0.2, 0.2, 0.8, 0.8, 0.2, 0.2],
    
#     # Step 6: Both extend to transition state
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   0.8, 0.8, 1.0, 1.0, 0.8, 0.8],
    
#     # Step 7: Second tensegrity extends fully, first starts new cycle
#     [0.3, 0.3, 1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    
#     # Step 8: Return to rest for cycle completion
#     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# ])

# roll_sequence = np.array([
#         [1.0, 1.0, 0.1, 1.0, 1.0, 0.1,   0.1, 0.8, 0.0, 1.0, 1.0, 0.0],
#         [0.0, 1.0, 1.0, 0.0, 0.8, 0.1,   1.0, 0.1, 1.0, 1.0, 0.1, 1.0],
#         [1.0, 0.1, 1.0, 1.0, 0.1, 1.0,   0.0, 0.1, 0.8, 0.0, 1.0, 1.0],
#         [1.0, 1.0, 0.0, 0.8, 0.1, 0.0,   1.0, 1.0, 0.1, 1.0, 1.0, 0.1],
#         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#     ])

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

# Start overall timing
start_time_total = time.time()
timing_breakdown = {}

print("=== TENSEGRITY ROBOT PRE-TRAINING AND RL TRAINING ===")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Create environment
print("\n1. Initializing environment...")
start_time = time.time()
env = TensegrityEnv(obs_mode="tier2", visualize=False)
timing_breakdown['Environment Setup'] = time.time() - start_time
print(f"   Environment setup completed in {timing_breakdown['Environment Setup']:.2f} seconds")

# Create a dataset of observations and expert actions
print("\n2. Generating expert demonstration dataset...")
start_time = time.time()
obs_dataset = []
action_dataset = []

obs, _ = env.reset()
# Increased dataset size for better coverage and reduced overfit risk
for _step in range(20000):  # Increased from 15000 for more diverse data
    for action in roll_sequence:
        # Store the current observation and the expert action
        obs_dataset.append(obs.copy())  # Use .copy() to avoid reference issues
        action_dataset.append(action.copy())
        
        # Take step in environment with expert action
        obs, _, done, _, _ = env.step(action)
        
        if done:
            obs, _ = env.reset()

timing_breakdown['Dataset Generation'] = time.time() - start_time
print(f"   Dataset generation completed in {timing_breakdown['Dataset Generation']:.2f} seconds")
print(f"   Generated {len(obs_dataset)} observation-action pairs")

# Convert to numpy arrays
print("\n3. Processing and augmenting dataset...")
start_time = time.time()
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

print(f"Dataset size after augmentation: {len(X)} samples")

# Verify observation dimension matches what the model will expect
print(f"Observation dimension: {X.shape[1]}")
print(f"Environment observation space shape: {env.observation_space.shape[0]}")

timing_breakdown['Data Processing'] = time.time() - start_time
print(f"   Data processing completed in {timing_breakdown['Data Processing']:.2f} seconds")

# Initialize the PPO model with tuned hyperparameters for locomotion
print("\n4. Initializing PPO model...")
start_time = time.time()
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./ppo_tensegrity_tensorboard/", 
    device="cpu",
    # Improved hyperparameters for complex locomotion tasks
    learning_rate=3e-4,           # More aggressive learning rate
    n_steps=2048,                 # Larger rollout buffer for better data
    batch_size=64,                # Efficient batch size
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

timing_breakdown['Model Initialization'] = time.time() - start_time
print(f"   PPO model initialization completed in {timing_breakdown['Model Initialization']:.2f} seconds")

# Pre-train the policy network using expert demonstrations
print("\n5. Setting up pre-training components...")
start_time = time.time()
# Access the policy network directly  
policy = model.policy
# Use same learning rate as PPO for consistency
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, weight_decay=1e-5)  # Added weight decay
loss_fn = torch.nn.MSELoss()

# Convert to PyTorch tensors on CPU explicitly
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

timing_breakdown['Pre-training Setup'] = time.time() - start_time
print(f"   Pre-training setup completed in {timing_breakdown['Pre-training Setup']:.2f} seconds")

# Pre-training loop - reduced epochs to prevent overfitting
print("\n6. Starting pre-training with expert demonstrations...")
start_time = time.time()
print("Pre-training policy network with expert demonstrations...")
num_epochs = 150  # Reduced from 200 to prevent overfitting
batch_size = 64   # Kept consistent with PPO batch_size
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
    
    # Early stopping if loss gets very low
    if avg_epoch_loss < 1e-6:
        print(f"Early stopping at epoch {epoch} - loss converged")
        break

timing_breakdown['Pre-training'] = time.time() - start_time
print(f"\n   Pre-training completed in {timing_breakdown['Pre-training']:.2f} seconds")

print("\n7. Saving pre-trained model...")
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"trained_models/tensegrity_gait_seeded_before_RL_{timestamp}")
timing_breakdown['Model Save (Pre-trained)'] = time.time() - start_time
print(f"   Pre-trained model saved in {timing_breakdown['Model Save (Pre-trained)']:.2f} seconds")

# Now continue with regular RL training
print("\n8. Starting RL training...")
start_time = time.time()
print("Starting RL training...")
# Reduced from 5M to prevent overtraining and allow for faster iteration
model.learn(total_timesteps=2_000_000)
timing_breakdown['RL Training'] = time.time() - start_time
print(f"\n   RL training completed in {timing_breakdown['RL Training']:.2f} seconds")

# Save the model with a timestamp to prevent overwriting files
print("\n9. Saving final trained model...")
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"trained_models/ppo_tensegrity_gait_seeded_{timestamp}")
timing_breakdown['Model Save (Final)'] = time.time() - start_time
print(f"Model saved as: ppo_tensegrity_gait_seeded_{timestamp}.zip")
print(f"   Final model saved in {timing_breakdown['Model Save (Final)']:.2f} seconds")

# Calculate total time and print comprehensive breakdown
total_time = time.time() - start_time_total
timing_breakdown['Total Time'] = total_time

print("\n" + "="*60)
print("           TRAINING TIMING BREAKDOWN")
print("="*60)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Print timing breakdown
for component, duration in timing_breakdown.items():
    if component != 'Total Time':
        percentage = (duration / total_time) * 100
        print(f"{component:<25}: {duration:>8.2f}s ({percentage:>5.1f}%)")

print("-" * 60)
print(f"{'TOTAL TIME':<25}: {total_time:>8.2f}s (100.0%)")
print("-" * 60)

# Convert to hours, minutes, seconds for readability
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

if hours > 0:
    time_str = f"{hours}h {minutes}m {seconds}s"
elif minutes > 0:
    time_str = f"{minutes}m {seconds}s"
else:
    time_str = f"{seconds}s"

print(f"\nTotal training time: {time_str}")
print("="*60)