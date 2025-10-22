from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mujoco_physics_engine.tensegrity_mjc_simulation_config_1 import *


def visualize_reward_components():
    """
    Visualize how different motion patterns affect each reward component.
    This helps debug and understand the reward function behavior.
    """
    print("🎯 REWARD COMPONENT VISUALIZATION")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path('sim_output')
    output_dir.mkdir(exist_ok=True)
    
    # Load simulator with visualization
    xml = Path('mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml')
    sim = TensegrityMuJoCoSimulator(xml, visualize=True, controller_kp=10.0, controller_ki=0.2, controller_kd=2.0)
    
    # Define test scenarios with different motion patterns
    test_scenarios = {
        "1. Good Rolling Pattern": {
            "sequence": np.array([
                [1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 0.1, 0.8, 0.0, 1.0, 1.0, 0.0],  # Roll sequence step 1
                [0.0, 1.0, 1.0, 0.0, 0.8, 0.1, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0],  # Roll sequence step 2
                [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 0.0, 0.1, 0.8, 0.0, 1.0, 1.0],  # Roll sequence step 3
            ]),
            "description": "Known good rolling sequence that should complete rolling motion"
        },
        
        "2. Full Contraction (EXPLOIT)": {
            "sequence": np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # All cables fully contracted
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]),
            "description": "Full contraction exploit - should be heavily penalized"
        },
        
        "3. High Oscillation (EXPLOIT)": {
            "sequence": np.array([
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # Rapid alternating
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],  # Pattern changes
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            ]),
            "description": "High oscillation pattern that might cause sliding"
        },
        
        "4. Balanced Extension": {
            "sequence": np.array([
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # All moderate
                [0.6, 0.7, 0.5, 0.8, 0.6, 0.7, 0.5, 0.8, 0.6, 0.7, 0.5, 0.8],  # Slight variations
                [0.7, 0.6, 0.8, 0.5, 0.7, 0.6, 0.8, 0.5, 0.7, 0.6, 0.8, 0.5],
            ]),
            "description": "Moderate, balanced cable extensions - baseline comparison"
        }
    }
    
    # Data collection for plotting
    all_results = {}
    
    for scenario_name, scenario_data in test_scenarios.items():
        print(f"\n🔄 Testing: {scenario_name}")
        print(f"   {scenario_data['description']}")
        
        # Reset simulation
        sim.reset()
        sim.bring_to_grnd()
        
        # Initialize tracking variables
        results = {
            'step': [],
            'total_reward': [],
            'rolling_quality': [],
            'exploitation_penalties': [],
            'cumulative_rotation': [],
            'consistent_direction': [],
            'distance_reward': [],
            'position_x': [],
            'position_y': [],
            'position_z': []
        }
        
        sequence = scenario_data['sequence']
        
        # Run through the sequence
        for step_idx, target_lengths in enumerate(sequence):
            print(f"     Step {step_idx + 1}: Running target lengths...")
            # Each sim.sim_step now advances by 100 physics steps (1s per action)
            observation, reward, done, info = sim.sim_step(target_lengths)
            # Get robot position
            end_pts = sim.get_endpts()
            robot_pos = end_pts.mean(axis=0)
            # Calculate individual reward components manually for visualization
            controls = np.array(target_lengths)
            # Calculate each reward component separately
            rolling_quality = sim._reward_cumulative_x_axis_rotation()
            # Use new penalty + reward component logging from info dict
            control_penalty_total = info.get('control_penalty_total', 0.0)
            # For backward naming compatibility, treat control penalty as exploitation_penalties (negative expected)
            exploitation_penalties = control_penalty_total
            cumulative_rotation = info.get('cumulative_rotation_reward', sim._reward_cumulative_x_axis_rotation())
            consistent_direction = info.get('consistent_direction_reward', sim._reward_consistent_rolling_direction(window_size=15))
            distance_reward = info.get('distance_reward', sim.calculate_omnidirectional_distance_reward(robot_pos) * 0.3)
            # Store results
            current_step = step_idx  # Now each step is one action (1s)
            results['step'].append(current_step)
            results['total_reward'].append(reward)
            results['rolling_quality'].append(rolling_quality)
            results['exploitation_penalties'].append(exploitation_penalties)
            results['cumulative_rotation'].append(cumulative_rotation)
            results['consistent_direction'].append(consistent_direction)
            results['distance_reward'].append(distance_reward)
            results['position_x'].append(robot_pos[0])
            results['position_y'].append(robot_pos[1])
            results['position_z'].append(robot_pos[2])
        
        # Calculate summary statistics
        avg_total_reward = np.mean(results['total_reward'])
        avg_rolling_quality = np.mean(results['rolling_quality'])
        avg_exploitation_penalties = np.mean(results['exploitation_penalties'])
        total_x_displacement = results['position_x'][-1] - results['position_x'][0]
        
        print(f"     Results:")
        print(f"       Average Total Reward: {avg_total_reward:.3f}")
        print(f"       Average Rolling Quality: {avg_rolling_quality:.3f}")
        print(f"       Average Exploitation Penalties: {avg_exploitation_penalties:.3f}")
        print(f"       Total X Displacement: {total_x_displacement:.3f}")
        
        all_results[scenario_name] = results
    
    # Create comprehensive visualization plots
    print(f"\n📊 Creating reward component visualization plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Reward Component Analysis for Different Motion Patterns', fontsize=16)
    
    colors = ['green', 'red', 'orange', 'blue']
    
    # Plot 1: Total Reward Over Time
    ax = axes[0, 0]
    for i, (scenario_name, results) in enumerate(all_results.items()):
        ax.plot(results['step'], results['total_reward'], label=scenario_name.split('.')[1].strip(), 
                color=colors[i % len(colors)], linewidth=2)
    ax.set_title('Total Reward Over Time')
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Rolling Quality Component
    ax = axes[0, 1]
    for i, (scenario_name, results) in enumerate(all_results.items()):
        ax.plot(results['step'], results['rolling_quality'], label=scenario_name.split('.')[1].strip(),
                color=colors[i % len(colors)], linewidth=2)
    ax.set_title('Rolling Quality Reward')
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Rolling Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Exploitation Penalties
    ax = axes[1, 0]
    for i, (scenario_name, results) in enumerate(all_results.items()):
        ax.plot(results['step'], results['exploitation_penalties'], label=scenario_name.split('.')[1].strip(),
                color=colors[i % len(colors)], linewidth=2)
    ax.set_title('Anti-Exploitation Penalties')
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Penalties (negative)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Position Trajectory (X displacement)
    ax = axes[1, 1]
    for i, (scenario_name, results) in enumerate(all_results.items()):
        ax.plot(results['step'], results['position_x'], label=scenario_name.split('.')[1].strip(),
                color=colors[i % len(colors)], linewidth=2)
    ax.set_title('Forward Progress (X Position)')
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('X Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative Rotation Reward
    ax = axes[2, 0]
    for i, (scenario_name, results) in enumerate(all_results.items()):
        ax.plot(results['step'], results['cumulative_rotation'], label=scenario_name.split('.')[1].strip(),
                color=colors[i % len(colors)], linewidth=2)
    ax.set_title('Cumulative Rotation Reward')
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Rotation Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary Bar Chart
    ax = axes[2, 1]
    scenario_names = [name.split('.')[1].strip() for name in all_results.keys()]
    avg_rewards = [np.mean(results['total_reward']) for results in all_results.values()]
    bars = ax.bar(scenario_names, avg_rewards, color=colors[:len(scenario_names)])
    ax.set_title('Average Total Reward by Scenario')
    ax.set_ylabel('Average Reward')
    ax.tick_params(axis='x', rotation=45)
    
    # Color bars based on reward value
    for bar, reward in zip(bars, avg_rewards):
        if reward > 0:
            bar.set_color('green')
        elif reward < -10:
            bar.set_color('red')
        else:
            bar.set_color('orange')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = output_dir / f"reward_analysis_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📈 Reward analysis plot saved to: {plot_filename}")
    
    # Show plot
    plt.show()
    
    # Print final analysis
    print(f"\n🎯 REWARD SYSTEM ANALYSIS SUMMARY:")
    print("=" * 60)
    
    for scenario_name, results in all_results.items():
        avg_reward = np.mean(results['total_reward'])
        avg_rolling_quality = np.mean(results['rolling_quality'])
        avg_penalties = np.mean(results['exploitation_penalties'])
        final_x_pos = results['position_x'][-1] - results['position_x'][0]
        
        status = "✅ GOOD" if avg_reward > 0 else "❌ PENALIZED" if avg_reward < -10 else "⚠️  NEUTRAL"
        
        print(f"{scenario_name}:")
        print(f"  Average Reward: {avg_reward:8.3f} | {status}")
        print(f"  Rolling Quality: {avg_rolling_quality:7.3f}")
        print(f"  Penalties: {avg_penalties:12.3f}")
        print(f"  X Displacement: {final_x_pos:8.3f}")
        print()
    
    print("🔍 What to look for:")
    print("✅ Good Rolling Pattern should have positive rewards and forward progress")
    print("❌ Full Contraction should be heavily penalized (< -20)")
    print("❌ High Oscillation should be penalized for rapid control changes")
    print("⚠️  Balanced Extension should be neutral baseline")
    
    return all_results


def run_single_sim():
    output_dir = Path('sim_output')
    output_dir.mkdir(exist_ok=True)
    xml = Path('mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml')
    sim = TensegrityMuJoCoSimulator(xml)

    target_lengths = [1.0 for _ in range(sim.n_actuators)]
    frames1 = sim.run_target_lengths(target_lengths, vis_save_dir=output_dir, vis_prefix='gait1', save_frames_as_png=False)

    target_lengths = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    frames2 = sim.run_target_lengths(target_lengths, vis_save_dir=output_dir, vis_prefix='gait2', save_frames_as_png=False)

    frames = frames1 + frames2
    
    # Generate timestamp for unique video filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"vid_{timestamp}.mp4"
    
    sim.save_video(Path(output_dir, video_filename), frames=frames)


def run_roll_sequence(sequence_json: str | None = None, repeats: int = 1, visualize: bool = True, interactive: bool = False, fast_forward: int = 0):
    """
    Simulate the tensegrity robot rolling through a predefined sequence of actions.
    Now includes plotting of target lengths, actual lengths, and PID responses.
    """

    # Quasi-static rolling gait - expanded for dual tensegrity (12 actuators)
    # Two options:
    # 1. Mirror pattern (same movement for both tensegrities)
    # NOTE: This pattern is from a single 3-bar gait. It is duplicated for two robots, but doesn't work effectively.
    base_sequence = np.array([
        [1.0, 1.0, 0.1, 1.0, 1.0, 0.1],  # Step 1 - both tensegrities move identically
        [0.0, 1.0, 1.0, 0.0, 0.8, 0.1],  # Step 2
        [1.0, 0.1, 1.0, 1.0, 0.1, 1.0],  # Step 3
        [1.0, 1.0, 0.0, 0.8, 0.1, 0.0],  # Step 4
        [0.1, 1.0, 1.0, 0.1, 1.0, 1.0],  # Step 5
        [1.0, 0.0, 1.0, 0.1, 0.0, 0.8]   # Step 6
    ])

    # # 2. Mirror and flipped pattern (same movement for both tensegrities but flipped for second tensegrity)
    # base_sequence = np.array([
    #     [1.0, 1.0, 0.1, 1.0, 1.0, 0.1,   0.1, 1.0, 1.0, 0.1, 1.0, 1.0], # NOTE: the second set of 6 actuators is flipped
    #     [0.0, 1.0, 1.0, 0.0, 0.8, 0.1,   0.1, 0.8, 0.0, 1.0, 1.0, 0.0],
    #     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0,   1.0, 0.1, 1.0, 1.0, 0.1, 1.0],
    #     [1.0, 1.0, 0.0, 0.8, 0.1, 0.0,   0.0, 0.1, 0.8, 0.0, 1.0, 1.0],
    #     [0.1, 1.0, 1.0, 0.1, 1.0, 1.0,   1.0, 1.0, 0.1, 1.0, 1.0, 0.1],
    #     [1.0, 0.0, 1.0, 0.1, 0.0, 0.8,   0.8, 0.0, 0.1, 1.0, 0.0, 1.0]
    # ])

    # 3. Offset and flipped pattern repeated n times (this pattern shuffle locomotes)
    # base_sequence = np.array([
    #     [1.0, 1.0, 0.1, 1.0, 1.0, 0.1,   0.1, 0.8, 0.0, 1.0, 1.0, 0.0],
    #     [0.0, 1.0, 1.0, 0.0, 0.8, 0.1,   1.0, 0.1, 1.0, 1.0, 0.1, 1.0],
    #     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0,   0.0, 0.1, 0.8, 0.0, 1.0, 1.0],
    #     [1.0, 1.0, 0.0, 0.8, 0.1, 0.0,   1.0, 1.0, 0.1, 1.0, 1.0, 0.1],
    #     [0.1, 1.0, 1.0, 0.1, 1.0, 1.0,   0.8, 0.0, 0.1, 1.0, 0.0, 1.0],
    #     [1.0, 0.0, 1.0, 0.1, 0.0, 0.8,   0.1, 1.0, 1.0, 0.1, 1.0, 1.0]
    # ])

    # # Testing which cables belong to the connected nodes
    # base_sequence = np.array([
    #     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #     [1.0, 1.0, 1.0, 1.0, 0.2, 1.0,   1.0, 0.2, 1.0, 1.0, 1.0, 1.0]
    # ])

    # # Hand designing double tensegrity gait
    # base_sequence = np.array([
    #     [1.0, 1.0, 0.1, 1.0, 1.0, 0.1,   0.1, 0.8, 0.0, 1.0, 1.0, 0.0],
    #     [0.0, 1.0, 1.0, 0.0, 0.8, 0.1,   1.0, 0.1, 1.0, 1.0, 0.1, 1.0],
    #     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0,   0.0, 0.1, 0.8, 0.0, 1.0, 1.0],
    #     [1.0, 1.0, 0.0, 0.8, 0.1, 0.0,   1.0, 1.0, 0.1, 1.0, 1.0, 0.1],
    #     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # ])
    # Steph trying shit
    # base_sequence = np.array([
    #     [1.0, 1.0, 0.1, 1.0, 1.0, 0.1,   0.1, 0.8, 0.0, 1.0, 1.0, 0.0], #start 1=b,2=r,3=g
    #     [0.0, 1.0, 1.0, 0.0, 0.8, 0.1,   1.0, 0.1, 1.0, 1.0, 0.1, 1.0],
    #     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0,   0.0, 0.1, 0.8, 0.0, 1.0, 1.0],
    #     [1.0, 1.0, 0.0, 0.8, 0.1, 1.0,   1.0, 1.0, 0.8, 1.0, 1.0, 0.1],
    #     [1.0, 1.0, 0.0, 0.8, 0.1, 1.0,   1.0, 1.0, 0.8, 1.0, 1.0, 0.1],
    #     [1.0, 0.1, 0.1, 1.0, 0.1, 0.1,   1.0, 0.8, 0.1, 1.0, 1.0, 0.0],
    #     [1.0, 0.5, 0.1, 1.0, 0.4, 0.1,   0.1, 0.8, 0.0, 1.0, 1.0, 0.5],
    #     [0.5, 0.5, 1.0, 1.0, 0.4, 0.1,   0.1, 0.8, 0.0, 0.5, 1.0, 1.0],
    #     [0.5, 0.5, 1.0, 1.0, 0.4, 0.1,   0.1, 0.8, 0.0, 0.5, 1.0, 1.0],
    #     [0.5, 0.5, 1.0, 1.0, 0.4, 0.8,   0.1, 0.8, 0.8, 0.5, 1.0, 1.0],
    #     [0.5, 0.1, 1.0, 0.1, 0.4, 1.0,   0.1, 0.8, 1.0, 0.5, 0.5, 1.0], #6,3 this and next two were 0.4
    #     [0.5, 0.1, 1.0, 0.1, 0.4, 0.8,   0.1, 0.8, 0.8, 0.5, 0.5, 1.0], #1 was 0.5...did this fuck it up?
    #     [0.5, 0.1, 1.0, 0.1, 0.4, 0.8,   0.1, 0.8, 0.8, 0.5, 0.5, 1.0], #10 was 1.0
    #     [0.5, 0.5, 1.0, 0.1, 0.4, 0.8,   0.1, 0.8, 0.8, 0.5, 1.0, 1.0],
    #     [0.5, 0.5, 1.0, 0.1, 0.4, 0.8,   0.1, 0.8, 0.8, 0.5, 1.0, 1.0],
    #     [0.5, 0.5, 1.0, 0.1, 0.4, 0.8,   0.1, 0.8, 0.8, 0.5, 1.0, 1.0], #1=g,2=b,3=r by here and stable
    #     [0.5, 0.5, 1.0, 0.4, 0.4, 0.4,   0.4, 0.8, 0.4, 0.5, 1.0, 1.0],
    #     [0.5, 0.5, 1.0, 0.8, 0.4, 0.2,   0.8, 0.8, 0.2, 0.5, 1.0, 1.0],
    #     [1.0, 0.5, 1.0, 1.0, 0.4, 0.1,   1.0, 0.8, 0.1, 1.0, 1.0, 1.0],
    #     [1.0, 0.1, 1.0, 1.0, 0.0, 0.1,   1.0, 0.0, 0.1, 1.0, 1.0, 1.0],
    #     [1.0, 0.1, 1.0, 0.1, 0.6, 0.1,   0.1, 0.0, 0.6, 1.0, 0.7, 1.0],
    #     [1.0, 0.1, 1.0, 0.6, 0.6, 0.1,   0.6, 0.0, 0.6, 1.0, 0.7, 1.0],
    #     [1.0, 0.1, 1.0, 0.6, 0.6, 0.1,   0.6, 0.0, 0.6, 1.0, 0.7, 1.0], #1=r,2=g,3=b
    #     [0.4, 0.4, 1.0, 0.6, 0.6, 0.1,   0.6, 0.0, 0.6, 1.0, 0.7, 1.0],
    #     [0.1, 1.0, 1.0, 0.6, 0.6, 0.1,   0.6, 0.0, 0.6, 1.0, 0.7, 1.0],
    #     [0.1, 1.0, 1.0, 0.6, 0.6, 0.1,   0.6, 0.0, 0.6, 1.0, 0.7, 1.0], #1=b,2=r,3=g
    #     [0.5, 1.0, 1.0, 0.6, 0.6, 0.5,   0.6, 0.5, 0.6, 1.0, 0.7, 1.0], #bring to neutral?
    #     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,   0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    #     [1.0, 1.0, 0.1, 1.0, 1.0, 0.1,   0.1, 0.8, 0.0, 1.0, 1.0, 0.0], #start 1=b,2=r,3=g
    # ])

    # Shuffling gait for dual tensegrity robot
    # Strategy: Alternate contraction patterns between tensegrities to create shuffling motion
    # First 6 actuators control first tensegrity, last 6 control second tensegrity
    # base_sequence = np.array([
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
    
    n = 4  # Number of times to repeat the sequence
    roll_sequence = np.tile(base_sequence, (n, 1))
    
    # Create output directory
    output_dir = Path('sim_output')
    output_dir.mkdir(exist_ok=True)
    
    # Load the XML model and create simulator
    xml = Path('mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml')
    sim = TensegrityMuJoCoSimulator(xml, controller_kp=10.0, controller_ki=0.2, controller_kd=2.0, visualize=visualize)
    
    # Configure camera for better robot visibility
    if sim.visualize and hasattr(sim, 'viewer') and sim.viewer is not None:
        sim.viewer.cam.distance = 20.0  # Zoom out
        sim.viewer.cam.elevation = -30  # Better viewing angle
        sim.viewer.cam.lookat[0] = 0.0  # Center on origin
        sim.viewer.cam.lookat[1] = 0.0
        sim.viewer.cam.lookat[2] = 1.0  # Slightly elevated view

    # init_viewer(sim)

    # Set up video capture
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"roll_sequence_{timestamp}.mp4"
    all_frames = []
    
    # Derive number of physics substeps per high-level action from dt.
    # Keep semantic: hold each action for action_hold_seconds simulated seconds.
    action_hold_seconds = 1.0  # Default preserves prior 200-step @ dt≈0.01s behavior.
    num_steps_per_sequence = max(1, round(action_hold_seconds / sim.dt))
    total_steps = len(roll_sequence) * num_steps_per_sequence
    print("----------------------------------")
    print(f"Simulation dt: {sim.dt:.6f}s, steps per action: {num_steps_per_sequence}, total simulation steps: {total_steps}")
    print("----------------------------------")

    # Initialize data arrays
    time_data = []
    target_lengths_data = []
    actual_lengths_data = []
    pid_responses_data = []
    reward_data = []
    test_reward_data = []  
    reward_components_data = []  # To collect reward component dictionaries
    sequence_boundaries = []  # To mark where each sequence step begins
    
    step_counter = 0
    
    print(f"Starting simulation with {len(roll_sequence)} sequence steps, {num_steps_per_sequence} substeps each (dt={sim.dt:.6f}s => {action_hold_seconds:.2f}s/action)")
    if fast_forward > 0:
        print(f"Fast-forward enabled: skipping visualization for first {fast_forward} sequence steps.")
    
    # Execute each action step in the sequence
    for i, target_lengths in enumerate(roll_sequence):
        skip_vis = fast_forward > 0 and i < fast_forward
        # Interactive prompt (only at sequence step granularity)
        if interactive:
            while True:
                user_in = input(
                    f"[Step {i+1}/{len(roll_sequence)}] Enter=run | s=skip | a=auto rest | q=quit > "
                ).strip().lower()
                if user_in in ('', 'c', 'run', 'r'):
                    # proceed with this step
                    break
                if user_in in ('s', 'skip'):
                    print(f"Skipping step {i+1} (no simulation run).")
                    sequence_boundaries.append(step_counter * sim.dt)
                    # treat as completed without simulation
                    break  # proceed to next step in outer loop
                if user_in in ('a', 'auto'):
                    print("Disabling interactive mode for remaining steps.")
                    interactive = False
                    break
                if user_in in ('q', 'quit', 'x'):
                    print("Quitting early at user request. Proceeding to plot collected data.")
                    # Jump to plotting with data collected so far
                    # Append boundary to mark termination point
                    sequence_boundaries.append(step_counter * sim.dt)
                    # Exit both loops by converting i loop to finished
                    # Use a flag
                    early_quit = True
                    break
                print("Unrecognized input. Use Enter, s, a, or q.")
            if 'early_quit' in locals() and early_quit:
                break
            # If user skipped step, continue to next sequence step without simulation
            if user_in in ('s', 'skip'):
                continue
        if skip_vis:
            if i == 0:
                print("(Fast-forwarding... no visualization until step", fast_forward + 1, ")")
        else:
            # First step after fast-forward boundary notification
            if fast_forward > 0 and i == fast_forward:
                print(f"Reached fast-forward boundary at step {i+1}; enabling visualization now.")
            print(f"Executing step {i+1}/{len(roll_sequence)} of rolling sequence")
        sequence_boundaries.append(step_counter * sim.dt)

        # Apply the target lengths and run simulation for multiple steps
        frames = [] if (sim.visualize and not skip_vis) else None
        
        # Reset if this is the first step (to ensure proper starting position)
        if i == 0:
            sim.reset()
            # Initialize exploration tracking right after reset
            end_pts_initial = sim.get_endpts()
            initial_robot_pos = end_pts_initial.mean(axis=0)
            max_distance_from_origin = 0.0
            prev_imu_grav = sim._get_IMU_gravity_vectors()
            sim.origin_pos = initial_robot_pos[:2].copy()
            sim.exploration_tracking_initialized = True
            print(f"Exploration tracking initialized after reset. Origin set to: {sim.origin_pos}")
        
        # Run the simulation for multiple steps with these target lengths
        for step in range(num_steps_per_sequence):
            # Only one sim_step per action; all sub-stepping and visualization are handled inside sim.sim_step
            current_time = step_counter * sim.dt
            time_data.append(current_time)

            end_pts = sim.get_endpts()
            prev_pos = end_pts.mean(axis=0)
            obs, reward, done, info = sim.sim_step(target_lengths)
            reward_data.append(reward)

            # Optionally collect diagnostic data here if needed (see original code for details)
            # For brevity, only main step logic is shown

            step_counter += 1
            if done:
                break
                
        # Add the frames to our collection
        if frames:
            all_frames.extend(frames)
    
    print("Simulation completed. Generating plots...")
    
    # Convert data to numpy arrays for easier plotting
    time_data = np.array(time_data)
    target_lengths_data = np.array(target_lengths_data)
    actual_lengths_data = np.array(actual_lengths_data)
    pid_responses_data = np.array(pid_responses_data)
    reward_data = np.array(reward_data)
    test_reward_data = np.array(test_reward_data)
    
    # Create comprehensive plots including rewards
    fig1, fig2 = create_comprehensive_analysis_plots(time_data, target_lengths_data, actual_lengths_data, 
                                      pid_responses_data, reward_data, test_reward_data, reward_components_data,
                                      sequence_boundaries, output_dir, timestamp,
                                      sim.min_cable_length, sim.max_cable_length)

# def init_viewer(sim):
#     """Initialize MuJoCo viewer with proper API"""
#     import mujoco.viewer
    
#     # Correct MuJoCo viewer initialization
#     viewer = mujoco.viewer.launch_passive(
#         sim.model, 
#         sim.data
#     )
    
#     # Configure camera for better robot visibility
#     if viewer is not None:
#         viewer.cam.distance = 10.0  # Zoom out to see full robot
#         viewer.cam.elevation = -30  # Better viewing angle
#         viewer.cam.lookat[0] = 0.0  # Center on origin
#         viewer.cam.lookat[1] = 0.0
#         viewer.cam.lookat[2] = 1.0  # Slightly elevated view
    
#     return viewer

def calculate_test_reward_function(sim, controls, target_lengths, prev_pos, max_distance_from_origin, prev_imu_grav):
    # This function for testing different potential reward functions

    ### Get end points for locomotion reward
    end_pts = sim.get_endpts()
    robot_pos = end_pts.mean(axis=0)  # Use the mean of end points as the robot's position

    # Debugging
    # print("Robot position: ", prev_pos)
    
    ### Calculate velocity magnitude reward
    
    if prev_pos is not None:
        # print("Inside velocity magnitude reward calculation")
        # Calculate XY plane displacement
        # print("Previous position: ", prev_pos)
        # print("Current position: ", robot_pos)
        xy_displacement = robot_pos[:2] - prev_pos[:2]  # [x, y] only
        xy_speed = np.linalg.norm(xy_displacement) / sim.dt
        
        velocity_magnitude_reward = xy_speed # Reward any XY movement with large magnitude
        # print("XY speed: ", xy_speed)

    ### Calculate positive velocity reward (forward movement)
    
    forward_direction = 0  # Assuming +x is forward direction
    positive_velocity_reward = 0.0
    if prev_pos is not None:
        # print("Inside positive velocity reward calculation")
        # Calculate XY plane displacement
        xy_displacement = robot_pos[:2] - prev_pos[:2]  # [x, y] only
        # Reward positive forward velocity (assuming +x is forward direction)
        forward_velocity = xy_displacement[forward_direction] / sim.dt  # x-component of velocity
        if forward_velocity > 0:
            positive_velocity_reward = forward_velocity # Additional reward for forward movement
    
    ### Add distance-based rewards (total distance from origin)
    
    # Tracking variables should already be initialized after reset
    if not hasattr(sim, 'exploration_tracking_initialized'):
        # print("WARNING: Exploration tracking not initialized! This shouldn't happen.")
        # Fallback initialization
        max_distance_from_origin = 0.0
        sim.origin_pos = robot_pos[:2].copy()
        sim.exploration_tracking_initialized = True
    
    # print(f"Max distance from origin: {max_distance_from_origin:.6f}")
    # Calculate distance from origin in XY plane
    xy_distance_from_origin = np.linalg.norm(robot_pos[:2] - sim.origin_pos)
    # print(f"Current distance from origin: {xy_distance_from_origin:.6f}")
    
    # Reward for reaching new maximum distances
    # Use a small tolerance to handle floating-point precision issues
    distance_tolerance = 1e-6
    exploration_reward = 0.0
    if xy_distance_from_origin > (max_distance_from_origin + distance_tolerance):
        distance_increase = xy_distance_from_origin - max_distance_from_origin
        # print(f"New max distance reached! {max_distance_from_origin:.6f} -> {xy_distance_from_origin:.6f} (+{distance_increase:.6f})")
        exploration_reward = distance_increase
        max_distance_from_origin = xy_distance_from_origin
    # else:
    #     print(f"No new max distance. Difference: {xy_distance_from_origin - max_distance_from_origin:.6f}")
    # Base distance reward (encourages staying away from origin)
    base_distance_reward = xy_distance_from_origin

    ### Calculate anti-exploit penalties

    # 1. Prevent excessive bouncing (z-axis exploitation)
    z_speed_penalty = 0.0
    if prev_pos is not None:
        z_speed = abs((robot_pos[2] - prev_pos[2]) / sim.dt)
        if z_speed > 1.0:  # Too much vertical movement
            z_speed_penalty = -1.0 * z_speed
    
    # 2. Prevent rapid oscillations in controls
    control_oscillation_penalty = 0.0
    if hasattr(sim, 'prev_controls'):
        control_change = np.sum(np.abs(controls - sim.prev_controls)) # max value should be 12?
        if control_change > 6.0:  # Too rapid changes
            control_oscillation_penalty = -1.0 * control_change
    sim.prev_controls = controls.copy()
    
    # 3. Energy efficiency penalty
    energy_cost = np.sum(np.abs(controls))
    energy_cost_penalty = -1.0 * energy_cost

    # 4. Rotation reward
    # Reward changes in IMU orientation - encourage rotation in one direction
    imu_reward = 0.0
    if hasattr(sim, 'prev_imu_grav'):
        current_imu_grav = sim._get_IMU_gravity_vectors()
        # Calculate change in orientation (gravity vector change)
        imu_reward = orientation_change = np.linalg.norm(current_imu_grav - prev_imu_grav)
        prev_imu_grav = current_imu_grav.copy()
    
    # Weighting individual reward components
    velocity_magnitude_reward *= 1.0
    positive_velocity_reward *= 1.0
    exploration_reward *= 1000.0
    base_distance_reward *= 1.0
    imu_reward *= 100.0
    # Weighting individual penalty components (quantities should already be negative)
    control_oscillation_penalty *= 0.0
    z_speed_penalty *= 0.0
    energy_cost_penalty *= 0.0

    reward = (
        velocity_magnitude_reward
        + positive_velocity_reward 
        + exploration_reward 
        + base_distance_reward 
        + imu_reward
        + control_oscillation_penalty
        + z_speed_penalty
        + energy_cost_penalty
    )

    # Create a container to store individual reward components
    reward_components = {
        "velocity_magnitude_reward": velocity_magnitude_reward,
        "positive_velocity_reward": positive_velocity_reward,
        "exploration_reward": exploration_reward,
        "base_distance_reward": base_distance_reward,
        "imu_reward": imu_reward,
        "control_oscillation_penalty": control_oscillation_penalty,
        "z_speed_penalty": z_speed_penalty,
        "energy_cost_penalty": energy_cost_penalty
    }

    # print("------")

    return reward, reward_components, max_distance_from_origin, prev_imu_grav

def create_comprehensive_analysis_plots(time_data, target_lengths_data, actual_lengths_data, 
                                       pid_responses_data, reward_data, test_reward_data, reward_components,
                                       sequence_boundaries, output_dir, timestamp, 
                                       min_cable_length, max_cable_length):
    """
    Create comprehensive plots including reward analysis.
    Creates two separate figures: one for cable/PID data, one for reward analysis.
    """
    n_actuators = target_lengths_data.shape[1]
    
    # Color map for different actuators
    colors = plt.cm.tab20(np.linspace(0, 1, n_actuators))
    
    # ===============================
    # FIGURE 1: Cable and PID Analysis (3 subplots)
    # ===============================
    fig1 = plt.figure(figsize=(16, 12))
    fig1.suptitle('Cable Length and PID Control Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Target vs Actual Cable Lengths for all actuators
    ax1 = plt.subplot(3, 1, 1)
    for i in range(n_actuators):
        # Convert normalized target to actual target length using simulator bounds
        target_actual = min_cable_length + (max_cable_length - min_cable_length) * target_lengths_data[:, i]
        
        ax1.plot(time_data, actual_lengths_data[:, i], color=colors[i], 
                linewidth=1.5, alpha=0.8, label=f'Actuator {i+1} (Actual)')
        ax1.plot(time_data, target_actual, color=colors[i], 
                linestyle='--', linewidth=1, alpha=0.6)
    
    # Add sequence boundaries
    for boundary in sequence_boundaries[1:]:  # Skip first boundary at t=0
        ax1.axvline(x=boundary, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax1.set_ylabel('Cable Length (m)')
    ax1.set_title('Target vs Actual Cable Lengths\n(Solid: Actual, Dashed: Target)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: PID Control Signals
    ax2 = plt.subplot(3, 1, 2)
    for i in range(n_actuators):
        ax2.plot(time_data, pid_responses_data[:, i], color=colors[i], 
                linewidth=1.5, alpha=0.8, label=f'Actuator {i+1}')
    
    # Add sequence boundaries
    for boundary in sequence_boundaries[1:]:
        ax2.axvline(x=boundary, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax2.set_ylabel('PID Control Signal')
    ax2.set_title('PID Control Responses')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Plot 3: Total Reward over Time
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(time_data, reward_data, 'purple', linewidth=2, label='Total Reward')
    ax3.plot(time_data, test_reward_data, 'red', linewidth=2, label='Total Test Reward')
    
    # Add sequence boundaries
    for boundary in sequence_boundaries[1:]:
        ax3.axvline(x=boundary, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Reward')
    ax3.set_title('Total Reward over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add text annotation about sequence boundaries
    fig1.text(0.02, 0.02, 'Red dotted lines indicate sequence step changes', 
             fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    # ===============================
    # FIGURE 2: Reward Analysis (Total + Individual Components)
    # ===============================
    
    # Determine number of reward components
    component_names = []
    if len(reward_components) > 0 and isinstance(reward_components, list):
        if isinstance(reward_components[0], dict):
            component_names = list(reward_components[0].keys())
    
    # Create figure with total reward + individual component subplots
    n_components = len(component_names)
    n_rows = max(2, n_components + 1)  # At least 2 rows (total + 1 component), or more if needed
    
    fig2 = plt.figure(figsize=(16, 4 * n_rows))
    fig2.suptitle('Detailed Reward Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Rewards (same as Plot 3 from Figure 1)
    ax_total = plt.subplot(n_rows, 1, 1)
    ax_total.plot(time_data, reward_data, 'purple', linewidth=2, label='Total Reward')
    ax_total.plot(time_data, test_reward_data, 'red', linewidth=2, label='Total Test Reward')
    
    # Add sequence boundaries
    for boundary in sequence_boundaries[1:]:
        ax_total.axvline(x=boundary, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax_total.set_ylabel('Reward')
    ax_total.set_title('Total Reward over Time')
    ax_total.grid(True, alpha=0.3)
    ax_total.legend()
    ax_total.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Individual component plots
    component_colors = plt.cm.Set3(np.linspace(0, 1, len(component_names)))
    
    for idx, component_name in enumerate(component_names):
        ax_comp = plt.subplot(n_rows, 1, idx + 2)
        
        component_values = [step_data.get(component_name, 0.0) for step_data in reward_components]
        
        if len(component_values) == len(time_data):
            ax_comp.plot(time_data, component_values, 
                        color=component_colors[idx], linewidth=2, 
                        label=component_name.replace('_', ' ').title())
            
            # Add sequence boundaries
            for boundary in sequence_boundaries[1:]:
                ax_comp.axvline(x=boundary, color='red', linestyle=':', alpha=0.5, linewidth=1)
            
            ax_comp.set_ylabel('Component Value')
            ax_comp.set_title(component_name.replace('_', ' ').title())
            ax_comp.grid(True, alpha=0.3)
            ax_comp.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            
            # Only add x-label to the last subplot
            if idx == len(component_names) - 1:
                ax_comp.set_xlabel('Time (seconds)')
    
    # Add text annotation about sequence boundaries
    fig2.text(0.02, 0.02, 'Red dotted lines indicate sequence step changes', 
             fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    # Show both figures
    plt.show()
    
    return fig1, fig2

def create_reward_statistics_summary(time_data, reward_data, reward_components_data, 
                                   sequence_boundaries, output_dir, timestamp):
    """
    Create a summary of reward statistics and save to file.
    """
    # Calculate statistics
    total_reward = np.sum(reward_data)
    mean_reward = np.mean(reward_data)
    std_reward = np.std(reward_data)
    min_reward = np.min(reward_data)
    max_reward = np.max(reward_data)
    
    # Calculate reward statistics per sequence step
    sequence_rewards = []
    for i in range(len(sequence_boundaries)):
        start_idx = int(sequence_boundaries[i] / (time_data[1] - time_data[0])) if i < len(sequence_boundaries) else 0
        end_idx = int(sequence_boundaries[i+1] / (time_data[1] - time_data[0])) if i+1 < len(sequence_boundaries) else len(reward_data)
        end_idx = min(end_idx, len(reward_data))
        
        if start_idx < len(reward_data):
            sequence_reward = np.sum(reward_data[start_idx:end_idx])
            sequence_rewards.append(sequence_reward)

    # Print key statistics to console
    print(f"\n=== REWARD ANALYSIS SUMMARY ===")
    print(f"Total Cumulative Reward: {total_reward:.3f}")
    print(f"Mean Reward per Step: {mean_reward:.3f}")
    print(f"Reward Standard Deviation: {std_reward:.3f}")
    print(f"Best Sequence Reward: {max(sequence_rewards):.3f} (Sequence {sequence_rewards.index(max(sequence_rewards))+1})")
    print(f"Worst Sequence Reward: {min(sequence_rewards):.3f} (Sequence {sequence_rewards.index(min(sequence_rewards))+1})")

def create_cable_analysis_plots(time_data, target_lengths_data, actual_lengths_data, 
                               pid_responses_data, sequence_boundaries, output_dir, timestamp,
                               min_cable_length, max_cable_length):
    """
    Create comprehensive plots showing target lengths, actual lengths, and PID responses.
    """
    n_actuators = target_lengths_data.shape[1]
    
    # Color map for different actuators
    colors = plt.cm.tab20(np.linspace(0, 1, n_actuators))
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Target vs Actual Cable Lengths for all actuators
    ax1 = plt.subplot(3, 1, 1)
    for i in range(n_actuators):
        # Convert normalized target to actual target length using simulator bounds
        target_actual = min_cable_length + (max_cable_length - min_cable_length) * target_lengths_data[:, i]
        
        ax1.plot(time_data, actual_lengths_data[:, i], color=colors[i], 
                linewidth=1.5, alpha=0.8, label=f'Actuator {i+1} (Actual)')
        ax1.plot(time_data, target_actual, color=colors[i], 
                linestyle='--', linewidth=1, alpha=0.6)
    
    # Add sequence boundaries
    for boundary in sequence_boundaries[1:]:  # Skip first boundary at t=0
        ax1.axvline(x=boundary, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax1.set_ylabel('Cable Length (m)')
    ax1.set_title('Target vs Actual Cable Lengths\n(Solid: Actual, Dashed: Target)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: PID Control Signals
    ax2 = plt.subplot(3, 1, 2)
    for i in range(n_actuators):
        ax2.plot(time_data, pid_responses_data[:, i], color=colors[i], 
                linewidth=1.5, alpha=0.8, label=f'Actuator {i+1}')
    
    # Add sequence boundaries
    for boundary in sequence_boundaries[1:]:
        ax2.axvline(x=boundary, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax2.set_ylabel('PID Control Signal')
    ax2.set_title('PID Control Responses')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Plot 3: Error (Target - Actual) for each actuator
    ax3 = plt.subplot(3, 1, 3)
    for i in range(n_actuators):
        target_actual = min_cable_length + (max_cable_length - min_cable_length) * target_lengths_data[:, i]
        error = target_actual - actual_lengths_data[:, i]
        ax3.plot(time_data, error, color=colors[i], 
                linewidth=1.5, alpha=0.8, label=f'Actuator {i+1}')
    
    # Add sequence boundaries
    for boundary in sequence_boundaries[1:]:
        ax3.axvline(x=boundary, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Error (Target - Actual) (m)')
    ax3.set_title('Cable Length Tracking Errors')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add text annotation about sequence boundaries
    fig.text(0.02, 0.02, 'Red dotted lines indicate sequence step changes', 
             fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"cable_analysis_{timestamp}.png"
    plt.savefig(output_dir / plot_filename, dpi=300, bbox_inches='tight')
    print(f"Cable analysis plot saved as {plot_filename}")
    
    # Create individual actuator plots for detailed analysis
    create_individual_actuator_plots(time_data, target_lengths_data, actual_lengths_data, 
                                   pid_responses_data, sequence_boundaries, output_dir, timestamp,
                                   min_cable_length, max_cable_length)
    
    plt.show()


def create_individual_actuator_plots(time_data, target_lengths_data, actual_lengths_data, 
                                   pid_responses_data, sequence_boundaries, output_dir, timestamp,
                                   min_cable_length, max_cable_length):
    """
    Create individual plots for each actuator showing detailed response.
    """
    n_actuators = target_lengths_data.shape[1]
    
    # Create plots for first 6 actuators (first tensegrity)
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    fig1.suptitle('First Tensegrity - Individual Actuator Responses', fontsize=16)
    
    # Create plots for last 6 actuators (second tensegrity)
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    fig2.suptitle('Second Tensegrity - Individual Actuator Responses', fontsize=16)
    
    for i in range(n_actuators):
        # Determine which figure and subplot to use
        if i < 6:
            ax = axes1[i // 3, i % 3]
            fig_num = 1
        else:
            ax = axes2[(i-6) // 3, (i-6) % 3]
            fig_num = 2
        
        # Convert normalized target to actual target length using simulator bounds
        target_actual = min_cable_length + (max_cable_length - min_cable_length) * target_lengths_data[:, i]
        error = target_actual - actual_lengths_data[:, i]
        
        # Plot actual length, target, and error
        ax_twin = ax.twinx()
        
        # Left y-axis: lengths
        line1 = ax.plot(time_data, actual_lengths_data[:, i], 'b-', linewidth=2, label='Actual Length')
        line2 = ax.plot(time_data, target_actual, 'r--', linewidth=2, label='Target Length')
        
        # Right y-axis: PID response
        line3 = ax_twin.plot(time_data, pid_responses_data[:, i], 'g-', linewidth=1.5, alpha=0.7, label='PID Response')
        
        # Add sequence boundaries
        for boundary in sequence_boundaries[1:]:
            ax.axvline(x=boundary, color='black', linestyle=':', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cable Length (m)', color='blue')
        ax_twin.set_ylabel('PID Response', color='green')
        ax.set_title(f'Actuator {i+1}')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=8)
        
        # Color the y-axis labels
        ax.tick_params(axis='y', labelcolor='blue')
        ax_twin.tick_params(axis='y', labelcolor='green')
    
    # Save the plots
    fig1.tight_layout()
    fig2.tight_layout()
    
    plot_filename1 = f"first_tensegrity_actuators_{timestamp}.png"
    plot_filename2 = f"second_tensegrity_actuators_{timestamp}.png"
    
    fig1.savefig(output_dir / plot_filename1, dpi=300, bbox_inches='tight')
    fig2.savefig(output_dir / plot_filename2, dpi=300, bbox_inches='tight')
    
    print(f"Individual actuator plots saved as {plot_filename1} and {plot_filename2}")
    
    plt.show()

    # Quasi-static rolling gait - expanded for dual tensegrity (12 actuators)
    # Two options:
    # 1. Mirror pattern (same movement for both tensegrities)
    # roll_sequence = np.array([
    #     [1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1],  # Step 1 - both tensegrities move identically
    #     [0.0, 1.0, 1.0, 0.0, 0.8, 0.1, 0.0, 1.0, 1.0, 0.0, 0.8, 0.1],  # Step 2
    #     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0],  # Step 3
    #     [1.0, 1.0, 0.0, 0.8, 0.1, 0.0, 1.0, 1.0, 0.0, 0.8, 0.1, 0.0],  # Step 4
    #     [0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0],  # Step 5
    #     [1.0, 0.0, 1.0, 0.1, 0.0, 0.8, 1.0, 0.0, 1.0, 0.1, 0.0, 0.8]   # Step 6
    # ])

    # 2. Alternating pattern (tensegrities move in sequence - potentially more stable)
    # roll_sequence = np.array([
    #     [1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 1 - first tensegrity moves
    #     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.8, 0.1],  # Step 2 - second tensegrity moves
    #     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 3 - first tensegrity moves
    #     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.8, 0.1, 0.0],  # Step 4 - second tensegrity moves
    #     [0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 5 - first tensegrity moves
    #     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.1, 0.0, 0.8]   # Step 6 - second tensegrity moves
    # ])

    # roll_sequence = np.array([[1.0,1.0,1.0,1.0,1.0,1.0],
        #                    [0.2,1.0,1.0,1.0,1.0,1.0],
        #                    [1.0,1.0,1.0,1.0,1.0,1.0],
        #                    [1.0,0.2,1.0,1.0,1.0,1.0],
        #                    [1.0,1.0,1.0,1.0,1.0,1.0],
        #                    [1.0,1.0,0.2,1.0,1.0,1.0],
        #                    [1.0,1.0,1.0,1.0,1.0,1.0],
        #                    [1.0,1.0,1.0,0.2,1.0,1.0],
        #                    [1.0,1.0,1.0,1.0,1.0,1.0],
        #                    [1.0,1.0,1.0,1.0,0.2,1.0],
        #                    [1.0,1.0,1.0,1.0,1.0,1.0],
        #                    [1.0,1.0,1.0,1.0,1.0,0.2]]) # testing one at a time

        # roll_sequence = np.array([[1.0, 1.0, 0.1, 1.0, 1.0, 0.1],[0.0, 1.0, 1.0, 0.0, 0.8, 0.1],[1.0, 0.1, 1.0, 1.0, 0.1, 1.0],[1.0, 1.0, 0.0, 0.8, 0.1, 0.0],[0.1, 1.0, 1.0, 0.1, 1.0, 1.0],[1.0, 0.0, 1.0, 0.1, 0.0, 0.8]]) # quasi-static rolling
    
    # Create output directory
    output_dir = Path('sim_output')
    output_dir.mkdir(exist_ok=True)
    
    # Load the XML model and create simulator
    xml = Path('mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml')
    sim = TensegrityMuJoCoSimulator(xml, visualize=True)
    
    # Set up video capture
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"roll_sequence_{timestamp}.mp4"
    all_frames = []
    
    
    print("Simulation completed. Generating plots...")
    
    # Convert data to numpy arrays for easier plotting
    time_data = np.array(time_data)
    target_lengths_data = np.array(target_lengths_data)
    actual_lengths_data = np.array(actual_lengths_data)
    pid_responses_data = np.array(pid_responses_data)
    
    # Create plots
    create_cable_analysis_plots(time_data, target_lengths_data, actual_lengths_data, 
                               pid_responses_data, sequence_boundaries, output_dir, timestamp)
    
    # Save the video if we have frames
    if all_frames:
        sim.save_video(Path(output_dir, video_filename), frames=all_frames)
        print(f"Rolling sequence simulation completed. Video saved as {video_filename}")
    else:
        print("No frames were captured. Check if visualization is enabled.")



def _list_available_sequences(directory: Path):
    if not directory.exists():
        print(f"No sequence directory found at {directory}")
        return
    print(f"Available action sequence JSON files in {directory}:")
    for p in sorted(directory.glob('*.json')):
        print(f"  - {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tensegrity roll sequence or analyses.")
    parser.add_argument('--sequence', '-s', type=str, default=None,
                        help='Path to JSON file with action sequence (defaults to steph_sequence.json).')
    parser.add_argument('--repeats', '-r', type=int, default=1, help='Number of times to repeat the loaded base sequence.')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization rendering.')
    parser.add_argument('--list', action='store_true', help='List available JSON sequences and exit.')
    parser.add_argument('--mode', choices=['roll','single','multi','analyze'], default='roll',
                        help='Execution mode: roll (default), single (simple demo), multi (multiprocess), analyze (reward component viz).')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Enable interactive stepping: prompt before each sequence step.')
    parser.add_argument('--fast-forward', type=int, default=0,
                        help='Number of initial sequence steps to simulate without visualization (still collects data).')
    args = parser.parse_args()

    seq_dir = Path('action_sequences')
    if args.list:
        _list_available_sequences(seq_dir)
        raise SystemExit(0)

    # Resolve sequence path if provided or default
    seq_path = args.sequence
    if seq_path is None:
        # default to Steph sequence
        seq_path = str(seq_dir / 'steph_sequence.json')
    else:
        # allow bare filename by searching in action_sequences
        cand = Path(seq_path)
        if not cand.exists():
            alt = seq_dir / seq_path
            if alt.exists():
                seq_path = str(alt)
            else:
                print(f"ERROR: Sequence file '{seq_path}' not found (also tried '{alt}'). Use --list to see options.")
                raise SystemExit(1)

    if args.mode == 'roll':
        run_roll_sequence(sequence_json=seq_path, repeats=args.repeats, visualize=not args.no_vis, interactive=args.interactive, fast_forward=args.fast_forward)
    elif args.mode == 'single':
        run_single_sim()
    elif args.mode == 'analyze':
        visualize_reward_components()
    else:
        print("Unknown mode.")