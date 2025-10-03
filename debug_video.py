"""
Debug script to analyze frame data and video writing issues
"""
from pathlib import Path
import numpy as np
import cv2
from mujoco_physics_engine.tensegrity_mjc_simulation import TensegrityMuJoCoSimulator

def debug_frame_format():
    """Debug the frame format and video writing process"""
    print("Debugging frame format...")
    
    # Create output directory
    output_dir = Path('sim_output')
    output_dir.mkdir(exist_ok=True)
    
    # Load the XML model and create simulator
    xml = Path('mujoco_physics_engine/xml_models/two_3bar_new_platform_config_1.xml')
    sim = TensegrityMuJoCoSimulator(xml, visualize=True)
    
    # Get a single frame
    target_lengths = [1.0 for _ in range(sim.n_actuators)]
    obs, reward, done, info = sim.sim_step(target_lengths)
    frame = sim.render()
    
    print(f"Frame shape: {frame.shape}")
    print(f"Frame dtype: {frame.dtype}")
    print(f"Frame min/max values: {frame.min():.3f} / {frame.max():.3f}")
    print(f"Renderer size: {sim.renderer.width} x {sim.renderer.height}")
    
    # Test manual video writer setup
    frame_size = (sim.renderer.width, sim.renderer.height)
    print(f"Expected frame size: {frame_size}")
    
    # Test different codecs manually
    codecs = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
    ]
    
    for codec_name, fourcc in codecs:
        try:
            video_path = output_dir / f"test_{codec_name}.mp4"
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, frame_size)
            
            if video_writer.isOpened():
                print(f"✅ {codec_name}: Video writer opened successfully")
                
                # Test writing the frame
                if frame.dtype != np.uint8:
                    test_frame = (frame * 255).astype(np.uint8)
                else:
                    test_frame = frame.copy()
                
                # Convert RGB to BGR
                bgr_frame = cv2.cvtColor(test_frame, cv2.COLOR_RGB2BGR)
                print(f"   BGR frame shape: {bgr_frame.shape}, dtype: {bgr_frame.dtype}")
                
                # Check if frame dimensions match
                expected_height, expected_width = frame_size[1], frame_size[0]
                actual_height, actual_width = bgr_frame.shape[:2]
                
                if (actual_width, actual_height) != frame_size:
                    print(f"   ⚠️ Dimension mismatch: expected {frame_size}, got ({actual_width}, {actual_height})")
                    bgr_frame = cv2.resize(bgr_frame, frame_size)
                    print(f"   Resized to: {bgr_frame.shape}")
                
                success = video_writer.write(bgr_frame)
                print(f"   Write success: {success}")
                
                video_writer.release()
                
                # Check file size
                if video_path.exists():
                    file_size = video_path.stat().st_size
                    print(f"   File created: {file_size} bytes")
                else:
                    print(f"   ❌ File not created")
                    
            else:
                print(f"❌ {codec_name}: Failed to open video writer")
                video_writer.release()
                
        except Exception as e:
            print(f"❌ {codec_name}: Exception - {e}")
    
    # Save frame as image for comparison
    if frame.dtype != np.uint8:
        img_frame = (frame * 255).astype(np.uint8)
    else:
        img_frame = frame.copy()
    
    img_path = output_dir / "debug_frame.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(img_frame, cv2.COLOR_RGB2BGR))
    print(f"Frame saved as image: {img_path}")

if __name__ == "__main__":
    debug_frame_format()