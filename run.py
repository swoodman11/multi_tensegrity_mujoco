from pathlib import Path
from datetime import datetime
import numpy as np
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
    """
    # Define the rolling sequence
    # roll_sequence = np.array([
    #     [1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1],  # Step 1
    #     [0.0, 1.0, 1.0, 0.0, 1.0, 0.1, 0.0, 1.0, 1.0, 0.0, 1.0, 0.1],  # Step 2
    #     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 3
    #     [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0],  # Step 4
    #     [1.0, 1.0, 0.0, 1.0, 0.1, 0.0, 1.0, 1.0, 0.0, 1.0, 0.1, 0.0],  # Step 5
    #     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 6
    #     # [1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1],  # Step 1
    #     # [0.0, 1.0, 1.0, 0.0, 1.0, 0.1, 0.0, 1.0, 1.0, 0.0, 1.0, 0.1],  # Step 2
    #     # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 3
    #     # [1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0],  # Step 4
    #     # [1.0, 1.0, 0.0, 1.0, 0.1, 0.0, 1.0, 1.0, 0.0, 1.0, 0.1, 0.0],  # Step 5
    #     # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Step 6
    # ])

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
    
    # Check what methods are available in the simulator
    print(f"Available methods in simulator: {[method for method in dir(sim) if not method.startswith('_')]}")
    
    # Execute each action step in the sequence - using sim_step instead of run_target_lengths
    for i, target_lengths in enumerate(roll_sequence):
        print(f"Executing step {i+1}/{len(roll_sequence)} of rolling sequence")
        
        # Apply the target lengths and run simulation for multiple steps
        frames = []
        num_steps = 200
        
        # Reset if this is the first step (to ensure proper starting position)
        if i == 0:
            sim.reset()
        
        # Run the simulation for multiple steps with these target lengths
        for step in range(num_steps):
            # Provide the target lengths as action input
            obs, reward, done, info = sim.sim_step(target_lengths)
            
            # Capture frame if visualization is enabled
            if sim.visualize:
                frame = sim.render()
                frames.append(frame)
            
            if done:
                break

            pause = 0.01  # Small delay for visualization purposes
            if sim.visualize:
                import time
                time.sleep(pause)
                
        # Add the frames to our collection
        all_frames.extend(frames)
    
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