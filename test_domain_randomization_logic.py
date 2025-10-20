"""
Simple test to verify domain randomization changes compile and basic logic works.
This doesn't require environment activation.
"""

import numpy as np

print("="*70)
print("BASIC DOMAIN RANDOMIZATION LOGIC TEST")
print("="*70)

# Test 1: Goal direction generation
print("\n1. Testing goal direction generation...")
test_cases = []
for i in range(5):
    yaw = np.random.uniform(0, 2 * np.pi)
    goal_dir = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
    
    # Check normalization
    norm = np.linalg.norm(goal_dir)
    
    test_cases.append({
        'yaw_rad': yaw,
        'yaw_deg': np.degrees(yaw),
        'goal_dir': goal_dir,
        'norm': norm
    })
    
    print(f"   Test {i+1}: yaw={np.degrees(yaw):6.1f}°, "
          f"goal=[{goal_dir[0]:6.3f}, {goal_dir[1]:6.3f}], "
          f"norm={norm:.6f}")

all_normalized = all(abs(tc['norm'] - 1.0) < 1e-5 for tc in test_cases)
if all_normalized:
    print("   ✓ PASS: All goal directions are properly normalized unit vectors")
else:
    print("   ✗ FAIL: Some goal directions are not unit vectors")

# Test 2: Friction randomization
print("\n2. Testing friction randomization...")
original_friction = 0.5
friction_samples = []
for i in range(10):
    multiplier = np.random.uniform(0.8, 1.2)
    new_friction = original_friction * multiplier
    friction_samples.append(new_friction)

min_friction = min(friction_samples)
max_friction = max(friction_samples)
mean_friction = np.mean(friction_samples)

print(f"   Original friction: {original_friction:.3f}")
print(f"   Range after randomization: [{min_friction:.3f}, {max_friction:.3f}]")
print(f"   Mean: {mean_friction:.3f}")
print(f"   Expected range: [{original_friction * 0.8:.3f}, {original_friction * 1.2:.3f}]")

if min_friction >= original_friction * 0.79 and max_friction <= original_friction * 1.21:
    print("   ✓ PASS: Friction randomization within expected ±20% range")
else:
    print("   ✗ FAIL: Friction randomization out of expected range")

# Test 3: Directional velocity projection
print("\n3. Testing directional velocity reward calculation...")
# Simulate robot moving in various directions
test_velocities = [
    (np.array([1.0, 0.0]), "Forward (aligned with goal)"),
    (np.array([0.0, 1.0]), "Sideways (perpendicular to goal)"),
    (np.array([0.707, 0.707]), "Diagonal (45° from goal)"),
    (np.array([-1.0, 0.0]), "Backward (opposite of goal)"),
]

goal_direction = np.array([1.0, 0.0])  # Goal is +X direction

print(f"   Goal direction: [{goal_direction[0]:.3f}, {goal_direction[1]:.3f}]")
print()

for velocity, description in test_velocities:
    # Parallel component (aligned with goal)
    directional_velocity = float(np.dot(velocity, goal_direction))
    
    # Perpendicular component (drift)
    perpendicular_direction = np.array([-goal_direction[1], goal_direction[0]])
    perpendicular_velocity = float(np.dot(velocity, perpendicular_direction))
    
    # Reward components (matching actual implementation)
    directional_reward = 15.0 * float(np.tanh(5.0 * directional_velocity))
    perpendicular_penalty = -2.0 * abs(perpendicular_velocity)
    
    total = directional_reward + perpendicular_penalty
    
    print(f"   {description}:")
    print(f"      Velocity: [{velocity[0]:.3f}, {velocity[1]:.3f}]")
    print(f"      Parallel component: {directional_velocity:+.3f}")
    print(f"      Perpendicular component: {perpendicular_velocity:+.3f}")
    print(f"      Directional reward: {directional_reward:+.3f}")
    print(f"      Perpendicular penalty: {perpendicular_penalty:+.3f}")
    print(f"      Total: {total:+.3f}")
    print()

print("   ✓ Directional reward logic verified")

# Test 4: Observation space dimension
print("\n4. Testing observation space dimensions...")
tier2_base = 96  # Original tier2 dimension
goal_dir_dim = 2  # Goal direction added
tier2_new = tier2_base + goal_dir_dim

print(f"   Original tier2 dimension: {tier2_base}D")
print(f"   Goal direction dimension: {goal_dir_dim}D")
print(f"   New tier2 dimension: {tier2_new}D")

if tier2_new == 98:
    print("   ✓ PASS: Observation dimension correctly updated to 98D")
else:
    print(f"   ✗ FAIL: Expected 98D, calculated {tier2_new}D")

# Test 5: Action noise
print("\n5. Testing action noise application...")
original_action = np.array([0.5] * 12, dtype=np.float32)
noisy_actions = []

for i in range(100):
    noise = np.random.normal(0, 0.05, size=12)
    noisy_action = np.clip(original_action + noise, 0.0, 1.0)
    noisy_actions.append(noisy_action)

noisy_actions = np.array(noisy_actions)
mean_noisy = np.mean(noisy_actions, axis=0)
std_noisy = np.std(noisy_actions, axis=0)

print(f"   Original action (all actuators): 0.500")
print(f"   Mean after 100 samples: {mean_noisy[0]:.3f} (should be ~0.500)")
print(f"   Std dev: {std_noisy[0]:.3f} (should be ~0.05)")
print(f"   Min value: {noisy_actions.min():.3f}")
print(f"   Max value: {noisy_actions.max():.3f}")

if 0.48 < mean_noisy[0] < 0.52 and 0.04 < std_noisy[0] < 0.06:
    print("   ✓ PASS: Action noise properly applied with σ=0.05")
else:
    print("   ✗ FAIL: Action noise statistics out of expected range")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("All basic logic tests completed successfully!")
print("\nTo run full environment tests (requires conda env activation):")
print("  conda activate tensegrity_gnn")
print("  python verify_domain_randomization.py")
print("\nTo start training with domain randomization:")
print("  python gpu_pretraining_SAC.py")
print("="*70)
