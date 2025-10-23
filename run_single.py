"""Simple demonstration script for single tensegrity robot.

Loads the single tensegrity simulator and runs basic test patterns.
Can load gait patterns from JSON files or use built-in test sequences.
"""

import numpy as np
from pathlib import Path
import json
from mujoco_physics_engine.single_tensegrity_mjc_simulation import SingleTensegrityMuJoCoSimulator


def load_gait_from_json(json_path: str) -> np.ndarray:
    """Load gait sequence from JSON file."""
    with open(json_path, 'r') as f:
        gait_data = json.load(f)
    
    if 'actions' in gait_data:
        actions = np.array(gait_data['actions'], dtype=np.float32)
    else:
        actions = np.array(gait_data, dtype=np.float32)
    
    print(f"✅ Loaded gait from {Path(json_path).name}")
    print(f"   Sequence length: {len(actions)} steps")
    if 'description' in gait_data:
        print(f"   Description: {gait_data['description']}")
    
    return actions


def run_simple_test():
    """Run simple test pattern without gait file."""
    print("🤖 Single Tensegrity Demonstration")
    print("=" * 60)
    
    # Initialize simulator
    print("\n1️⃣ Initializing simulator...")
    xml_path = Path("mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml")
    sim = SingleTensegrityMuJoCoSimulator(
        xml_path=xml_path,
        visualize=True,
        render_size=(800, 600),
        render_fps=30,
        debug_enabled=False
    )
    
    print(f"✅ Simulator initialized")
    print(f"   Actuators: {sim.n_actuators}")
    print(f"   Observation dim: {sim.obs_dim}")
    print(f"   Cable sites: {len(sim.cable_sites)}")
    
    # Define a simple test pattern
    print("\n2️⃣ Running simple alternating test pattern...")
    test_pattern = np.array([
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Neutral
        [0.2, 0.8, 0.2, 0.8, 0.2, 0.8],  # Alternating 1
        [0.8, 0.2, 0.8, 0.2, 0.8, 0.2],  # Alternating 2
        [0.2, 0.8, 0.2, 0.8, 0.2, 0.8],  # Alternating 1
        [0.8, 0.2, 0.8, 0.2, 0.8, 0.2],  # Alternating 2
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Neutral
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # Extreme 1
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],  # Extreme 2
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Neutral
    ], dtype=np.float32)
    
    # Run pattern multiple times
    num_cycles = 1
    total_reward = 0.0
    
    # Note: sim_step now internally executes 100 physics steps per action (1 second)
    dt = sim.dt
    print(f"   Physics timestep: {dt}s")
    print(f"   Each action held for 1.0 second (100 physics steps internally)")
    
    # --- Logging for plotting ---
    # action_log: stores the input gait sequence (desired control action), values in [0,1]
    # pid_log: stores the PID output (as used for cable control), values in [-1,1]
    # cable_length_log: actual cable lengths
    # reward_log: reward per step
    #
    # The desired control action is simply the input action from the gait sequence or test pattern,
    # which is passed directly to sim.sim_step(action). It is not transformed or normalized here.
    action_log = []
    pid_log = []
    cable_length_log = []
    reward_log = []
    desired_cable_length_log = []

    for cycle in range(num_cycles):
        print(f"\n   Cycle {cycle + 1}/{num_cycles}")
        for action_idx, action in enumerate(test_pattern):
            print(f"     Executing action {action_idx + 1}/{len(test_pattern)}...")
            obs, reward, done, info = sim.sim_step(action)
            total_reward += reward

            # Log data for plotting
            action_log.append(np.array(action))  # input action, [0,1]
            pid_log.append(np.array(info['controls']) if info['controls'] is not None else np.zeros(sim.n_actuators))
            cable_length_log.append(sim._get_actuated_cable_lengths())
            reward_log.append(reward)
            desired_lengths = sim.min_cable_length + action * (sim.max_cable_length - sim.min_cable_length)
            desired_cable_length_log.append(desired_lengths)

    print(f"\n✅ Test complete!")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Average reward per action: {total_reward / (num_cycles * len(test_pattern)):.3f}")

    # --- Plotting ---
    import matplotlib.pyplot as plt
    action_log = np.array(action_log)
    pid_log = np.array(pid_log)
    cable_length_log = np.array(cable_length_log)
    reward_log = np.array(reward_log)
    desired_cable_length_log = np.array(desired_cable_length_log)
    steps = np.arange(len(action_log))

    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    colors = plt.cm.tab10.colors
    # 1. Input Action
    for i in range(sim.n_actuators):
        axs[0].plot(steps, action_log[:, i], label=f'Input Action {i+1}', linestyle='-', color=colors[i % len(colors)], linewidth=1.5)
    axs[0].set_ylabel('Action [0,1]')
    axs[0].set_title('Input Action (Desired Control)')
    axs[0].legend(loc='upper right', ncol=2, fontsize=8)
    axs[0].grid(True)
    # 2. PID Output
    for i in range(sim.n_actuators):
        axs[1].plot(steps, pid_log[:, i], label=f'PID Output {i+1}', linestyle='-', color=colors[i % len(colors)], linewidth=1.5)
    axs[1].set_ylabel('PID Output [-1,1]')
    axs[1].set_title('PID Control Signal')
    axs[1].legend(loc='upper right', ncol=2, fontsize=8)
    axs[1].grid(True)
    # 3. Cable Lengths
    for i in range(sim.n_actuators):
        axs[2].plot(steps, cable_length_log[:, i], label=f'Actual Cable {i+1}', linestyle='-', color=colors[i % len(colors)], linewidth=1.5)
        axs[2].plot(steps, desired_cable_length_log[:, i], label=f'Desired Cable {i+1}', linestyle='--', color=colors[i % len(colors)], linewidth=1.2)
    axs[2].set_ylabel('Cable Length (m)')
    axs[2].set_title('Actual (solid) and Desired (dashed) Cable Lengths')
    axs[2].legend(loc='upper right', ncol=2, fontsize=8)
    axs[2].grid(True)
    # 4. Reward
    axs[3].plot(steps, reward_log, label='Reward', color='k')
    axs[3].set_ylabel('Reward')
    axs[3].set_xlabel('Step')
    axs[3].set_title('Reward per Step')
    axs[3].legend()
    axs[3].grid(True)
    plt.tight_layout()
    plt.show()


def run_with_gait_file(gait_json: str, num_cycles: int = 3):
    """Run simulation with gait pattern from JSON file."""
    print("🤖 Single Tensegrity Demonstration (with Gait File)")
    print("=" * 60)
    
    # Load gait
    # gait_json = Path("gaits") / gait_json
    print(f"\n1️⃣ Loading gait from {gait_json}...")
    gait_sequence = load_gait_from_json(gait_json)
    
    # Initialize simulator
    print("\n2️⃣ Initializing simulator...")
    xml_path = Path("mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml")
    sim = SingleTensegrityMuJoCoSimulator(
        xml_path=xml_path,
        visualize=True,
        render_size=(800, 600),
        render_fps=30,
        debug_enabled=False
    )
    
    print(f"✅ Simulator initialized")
    
    # Run gait
    print(f"\n3️⃣ Running gait for {num_cycles} cycles...")
    total_reward = 0.0
    
    # Note: sim_step now internally executes 100 physics steps per action (1 second)
    dt = sim.dt
    print(f"   Physics timestep: {dt}s")
    print(f"   Each action held for 1.0 second (100 physics steps internally)")
    
    # --- Logging for plotting ---
    # action_log: stores the input gait sequence (desired control action), values in [0,1]
    # pid_log: stores the PID output (as used for cable control), values in [-1,1]
    # cable_length_log: actual cable lengths
    # reward_log: reward per step
    #
    # The desired control action is simply the input action from the gait sequence or test pattern,
    # which is passed directly to sim.sim_step(action). It is not transformed or normalized here.
    action_log = []
    pid_log = []
    cable_length_log = []
    desired_cable_length_log = []
    reward_log = []

    for cycle in range(num_cycles):
        print(f"\n   Cycle {cycle + 1}/{num_cycles}")
        for action_idx, action in enumerate(gait_sequence):
            obs, reward, done, info = sim.sim_step(action)
            total_reward += reward

            # Log data for plotting
            action_log.append(np.array(action))  # input action, [0,1]
            pid_log.append(np.array(info['controls']) if info['controls'] is not None else np.zeros(sim.n_actuators))
            cable_length_log.append(sim._get_actuated_cable_lengths())
            desired_cable_length_log.append(sim.min_cable_length + np.array(action) * (sim.max_cable_length - sim.min_cable_length))
            reward_log.append(reward)

            if action_idx % 5 == 0:
                print(f"     Action {action_idx}/{len(gait_sequence)}: cumulative reward={total_reward:.3f}")
        print(f"   Cycle reward: {total_reward:.2f}")

    print(f"\n✅ Demonstration complete!")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Average reward per action: {total_reward / (num_cycles * len(gait_sequence)):.3f}")

    # Print final robot position
    robot_pos = sim.get_robot_position()
    print(f"\n📍 Final robot position: [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")

    # --- Plotting ---
    import matplotlib.pyplot as plt
    action_log = np.array(action_log)
    pid_log = np.array(pid_log)
    cable_length_log = np.array(cable_length_log)
    reward_log = np.array(reward_log)
    desired_cable_length_log = np.array(desired_cable_length_log)
    steps = np.arange(len(action_log))

    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    colors = plt.cm.tab10.colors
    # 1. Input Action
    for i in range(sim.n_actuators):
        axs[0].plot(steps, action_log[:, i], label=f'Input Action {i+1}', linestyle='-', color=colors[i % len(colors)], linewidth=1.5)
    axs[0].set_ylabel('Action [0,1]')
    axs[0].set_title('Input Action (Desired Control)')
    axs[0].legend(loc='upper right', ncol=2, fontsize=8)
    axs[0].grid(True)
    # 2. PID Output
    for i in range(sim.n_actuators):
        axs[1].plot(steps, pid_log[:, i], label=f'PID Output {i+1}', linestyle='-', color=colors[i % len(colors)], linewidth=1.5)
    axs[1].set_ylabel('PID Output [-1,1]')
    axs[1].set_title('PID Control Signal')
    axs[1].legend(loc='upper right', ncol=2, fontsize=8)
    axs[1].grid(True)
    # 3. Cable Lengths
    for i in range(sim.n_actuators):
        axs[2].plot(steps, cable_length_log[:, i], label=f'Actual Cable {i+1}', linestyle='-', color=colors[i % len(colors)], linewidth=1.5)
        axs[2].plot(steps, desired_cable_length_log[:, i], label=f'Desired Cable {i+1}', linestyle='--', color=colors[i % len(colors)], linewidth=1.2)
    axs[2].set_ylabel('Cable Length (m)')
    axs[2].set_title('Actual (solid) and Desired (dashed) Cable Lengths')
    axs[2].legend(loc='upper right', ncol=2, fontsize=8)
    axs[2].grid(True)
    # 4. Reward
    axs[3].plot(steps, reward_log, label='Reward', color='k')
    axs[3].set_ylabel('Reward')
    axs[3].set_xlabel('Step')
    axs[3].set_title('Reward per Step')
    axs[3].legend()
    axs[3].grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1:
        # Use gait file if provided
        gait_file = sys.argv[1]
        gait_file = Path("gaits") / gait_file
        num_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        
        if not Path(gait_file).exists():
            print(f"❌ Gait file not found: {gait_file}")
            print("\nAvailable gait files:")
            for gait in Path(".").glob("*.json"):
                print(f"  - {gait.name}")
            return
        
        run_with_gait_file(gait_file, num_cycles)
    else:
        # Run simple test
        print("Usage: python run_single.py [gait_file.json] [num_cycles]")
        print("Running with built-in test pattern...\n")
        run_simple_test()


if __name__ == "__main__":
    main()
