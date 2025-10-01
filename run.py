from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mujoco_physics_engine.tensegrity_mjc_simulation import *


def run_single_sim():
    output_dir = Path('sim_output')
    output_dir.mkdir(exist_ok=True)
    xml = Path('mujoco_physics_engine/xml_models/two_3bar_new_platform_config_1.xml')
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


def run_roll_sequence():
    """
    Simulate the tensegrity robot rolling through a predefined sequence of actions.
    Now includes plotting of target lengths, actual lengths, and PID responses.
    """
    # Define the rolling sequence
    # Testing one cable at a time - expanded for dual tensegrity (12 actuators)
    roll_sequence = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # All extended (baseline)
        [1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # First tensegrity, cable 2
        [1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # First tensegrity, cable 3
        [1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # First tensegrity, cable 4
        [1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # First tensegrity, cable 5
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # First tensegrity, cable 6
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0],  # Second tensegrity, cable 1
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0],  # Second tensegrity, cable 2
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0],  # Second tensegrity, cable 3
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0],  # Second tensegrity, cable 4
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0],  # Second tensegrity, cable 5
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2],   # Second tensegrity, cable 6
        [0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # First tensegrity, cable 1
    ])
    
    # Create output directory
    output_dir = Path('sim_output')
    output_dir.mkdir(exist_ok=True)
    
    # Load the XML model and create simulator
    xml = Path('mujoco_physics_engine/xml_models/two_3bar_new_platform_config_1.xml')
    sim = TensegrityMuJoCoSimulator(xml, visualize=True)
    
    # Set up video capture
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"roll_sequence_{timestamp}.mp4"
    all_frames = []
    
    # Data collection for plotting
    num_steps_per_sequence = 200
    total_steps = len(roll_sequence) * num_steps_per_sequence
    
    # Initialize data arrays
    time_data = []
    target_lengths_data = []
    actual_lengths_data = []
    pid_responses_data = []
    sequence_boundaries = []  # To mark where each sequence step begins
    
    step_counter = 0
    
    print(f"Starting simulation with {len(roll_sequence)} sequence steps, {num_steps_per_sequence} substeps each")
    
    # Execute each action step in the sequence
    for i, target_lengths in enumerate(roll_sequence):
        print(f"Executing step {i+1}/{len(roll_sequence)} of rolling sequence")
        sequence_boundaries.append(step_counter * sim.dt)
        
        # Apply the target lengths and run simulation for multiple steps
        frames = []
        
        # Reset if this is the first step (to ensure proper starting position)
        if i == 0:
            sim.reset()
        
        # Run the simulation for multiple steps with these target lengths
        for step in range(num_steps_per_sequence):
            # Record time
            current_time = step_counter * sim.dt
            time_data.append(current_time)
            
            # Record target lengths
            target_lengths_data.append(target_lengths.copy())
            
            # Get actual cable lengths before sim step
            actual_lengths = []
            for actuator_idx in range(sim.n_actuators):
                cable_idx = sim.actuated_ids[actuator_idx]
                s0 = sim.mjc_data.sensor(f"pos_{sim.cable_sites[cable_idx][0]}").data
                s1 = sim.mjc_data.sensor(f"pos_{sim.cable_sites[cable_idx][1]}").data
                current_length = np.linalg.norm(s1 - s0)
                actual_lengths.append(current_length)
            actual_lengths_data.append(actual_lengths.copy())
            
            # Get PID responses (control signals)
            pid_responses = []
            for actuator_idx in range(sim.n_actuators):
                # Access the PID controller's current output
                if hasattr(sim.pids[actuator_idx], 'u') and sim.pids[actuator_idx].u is not None:
                    if isinstance(sim.pids[actuator_idx].u, np.ndarray):
                        pid_output = sim.pids[actuator_idx].u.item() if sim.pids[actuator_idx].u.size == 1 else sim.pids[actuator_idx].u[0]
                    else:
                        pid_output = sim.pids[actuator_idx].u
                else:
                    pid_output = 0.0
                pid_responses.append(pid_output)
            pid_responses_data.append(pid_responses.copy())
            
            # Provide the target lengths as action input
            obs, reward, done, info = sim.sim_step(target_lengths)
            
            # Capture frame if visualization is enabled
            if sim.visualize:
                frame = sim.render()
                frames.append(frame)
            
            step_counter += 1
            
            if done:
                break

            # Small delay for visualization purposes
            if sim.visualize and step % 10 == 0:  # Reduce frequency of sleep to speed up
                import time
                time.sleep(0.001)
                
        # Add the frames to our collection
        all_frames.extend(frames)
    
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
    # if all_frames:
    #     sim.save_video(Path(output_dir, video_filename), frames=all_frames)
    #     print(f"Rolling sequence simulation completed. Video saved as {video_filename}")
    # else:
    #     print("No frames were captured. Check if visualization is enabled.")


def create_cable_analysis_plots(time_data, target_lengths_data, actual_lengths_data, 
                               pid_responses_data, sequence_boundaries, output_dir, timestamp):
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
        # Convert normalized target to actual target length for comparison
        min_length = 0.1  # From PID implementation
        max_length = 1.0
        target_actual = min_length + (max_length - min_length) * target_lengths_data[:, i]
        
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
        min_length = 0.1
        max_length = 1.0
        target_actual = min_length + (max_length - min_length) * target_lengths_data[:, i]
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
                                   pid_responses_data, sequence_boundaries, output_dir, timestamp)
    
    plt.show()


def create_individual_actuator_plots(time_data, target_lengths_data, actual_lengths_data, 
                                   pid_responses_data, sequence_boundaries, output_dir, timestamp):
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
        
        # Convert normalized target to actual target length
        min_length = 0.1
        max_length = 1.0
        target_actual = min_length + (max_length - min_length) * target_lengths_data[:, i]
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
    xml = Path('mujoco_physics_engine/xml_models/two_3bar_new_platform_config_1.xml')
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
    # if all_frames:
    #     sim.save_video(Path(output_dir, video_filename), frames=all_frames)
    #     print(f"Rolling sequence simulation completed. Video saved as {video_filename}")
    # else:
    #     print("No frames were captured. Check if visualization is enabled.")


def run_multi_sim():
    num_sim = 3
    xml = Path('mujoco_physics_engine/xml_models/two_3bar_new_platform_config_2.xml')
    multi_sim = MultiProcTensegrityMujocoSimulator(num_sim, xml)

    target_lengths = [
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    ]
    multi_sim.parallel_run_target_lengths(target_lengths)


if __name__ == "__main__":
    # Uncomment the function you want to run
    # run_single_sim()
    run_roll_sequence()
    # run_multi_sim()