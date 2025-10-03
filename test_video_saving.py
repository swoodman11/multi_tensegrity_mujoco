"""
Quick test script to verify video saving functionality
"""
from pathlib import Path
from mujoco_physics_engine.tensegrity_mjc_simulation import TensegrityMuJoCoSimulator

def test_video_saving():
    """Test video saving with a short simulation"""
    print("Testing video saving functionality...")
    
    # Create output directory
    output_dir = Path('sim_output')
    output_dir.mkdir(exist_ok=True)
    
    # Load the XML model and create simulator
    xml = Path('mujoco_physics_engine/xml_models/two_3bar_new_platform_config_1.xml')
    sim = TensegrityMuJoCoSimulator(xml, visualize=True)
    
    # Run a short simulation and collect frames
    frames = []
    target_lengths = [1.0 for _ in range(sim.n_actuators)]
    
    print("Running short simulation...")
    for i in range(100):  # Just 100 steps for quick test
        obs, reward, done, info = sim.sim_step(target_lengths)
        
        if sim.visualize:
            frame = sim.render()
            frames.append(frame)
        
        if i % 20 == 0:
            print(f"Step {i}/100")
    
    # Test video saving
    video_path = output_dir / "test_video_saving.mp4"
    print(f"Saving video with {len(frames)} frames...")
    sim.save_video(video_path, frames)
    
    # Check if file was created
    if video_path.exists():
        print(f"✅ Success: Video saved to {video_path}")
        file_size = video_path.stat().st_size
        print(f"   File size: {file_size / 1024:.1f} KB")
    else:
        print("❌ Error: Video file was not created")

if __name__ == "__main__":
    test_video_saving()