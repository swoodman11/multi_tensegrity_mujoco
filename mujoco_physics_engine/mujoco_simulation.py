import json
import random
from pathlib import Path
from typing import List, Optional

import cv2
import mujoco
import numpy as np

def debug_print(message, filename="mujoco_simulation.py", debug_enabled=False):
    """Print debug messages with filename prefix if debug is enabled"""
    if debug_enabled:
        print(f"DEBUG {filename}: {message}")


class AbstractMuJoCoSimulator:
    """
    MuJoCo interface
    """

    def __init__(self,
                 xml_path: Path,
                 visualize: bool = False,
                 render_size: tuple[int, int] = (480, 640),
                 render_fps: int = 50):
        self.xml_path = xml_path
        self.visualize = visualize
        self.mjc_model = self._load_model_from_xml(xml_path)
        self.mjc_data = mujoco.MjData(self.mjc_model)
        # Store render configuration and lazy-create renderer when needed
        self._render_size = render_size
        self.render_fps = render_fps
        self.renderer = mujoco.Renderer(self.mjc_model, render_size[0], render_size[1]) if visualize else None
        self.render_fps = render_fps
        self.states = []
        self.time = 0
        self.dt = self.mjc_model.opt.timestep

    def reset(self):
        self.mjc_model = self._load_model_from_xml(self.xml_path)
        self.mjc_data = mujoco.MjData(self.mjc_model)
        # Recreate renderer if it already exists to bind to the new model
        if self.renderer is not None:
            try:
                self.renderer = mujoco.Renderer(self.mjc_model, self._render_size[0], self._render_size[1])
            except Exception:
                # If recreation fails, drop renderer to avoid stale references
                self.renderer = None

    def _load_model_from_xml(self, xml_path: Path) -> mujoco.MjModel:
        # Convert string to Path object if needed
        if isinstance(xml_path, str):
            xml_path = Path(xml_path)
        model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        return model

    def sim_step(self):
        mujoco.mj_step(self.mjc_model, self.mjc_data)

    def forward(self):
        mujoco.mj_forward(self.mjc_model, self.mjc_data)

    def render_frame(self, view='camera'):
        """Render a single RGB frame as a numpy array without requiring a visible viewer.

        Lazily creates a headless renderer if needed, so this works even when
        visualize=False (useful for --no-vis + --save-video runs).
        """
        try:
            if self.renderer is None:
                # Lazy-create a headless renderer
                self.renderer = mujoco.Renderer(self.mjc_model, self._render_size[0], self._render_size[1])
            self.renderer.update_scene(self.mjc_data, view)
            frame = self.renderer.render()
            # Some backends already return uint8; standardize to copy to detach buffer
            return frame.copy() if hasattr(frame, 'copy') else frame
        except Exception as e:
            debug_print(f"render_frame failed: {e}", "mujoco_simulation.py", False)
            return None

    def save_video(self, save_path: Path, frames: list):
        if not frames:
            print("Warning: No frames to save")
            return
        # Filter out any None frames defensively
        frames = [f for f in frames if f is not None]
        if not frames:
            print("Error: All collected frames were None; nothing to save")
            return
        
        # Get frame size from the actual frame dimensions, not renderer settings
        if frames:
            actual_frame = frames[0]
            if len(actual_frame.shape) == 3:
                frame_height, frame_width = actual_frame.shape[:2]
                frame_size = (frame_width, frame_height)
            else:
                print("Error: Invalid frame dimensions")
                return
        else:
            # Fallback to renderer size if available
            if hasattr(self, 'renderer') and self.renderer is not None:
                frame_size = (self.renderer.width, self.renderer.height)
            else:
                print("Error: Cannot determine frame size")
                return
        
        print(f"Detected frame size: {frame_size}")
        
        # Try multiple codecs in order of preference
        codecs_to_try = [
            cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 Part 2
            cv2.VideoWriter_fourcc(*'XVID'),  # Xvid
            cv2.VideoWriter_fourcc(*'MJPG'),  # Motion JPEG
        ]
        
        video_writer = None
        for fourcc in codecs_to_try:
            try:
                # Use string path instead of .as_posix() for better Windows compatibility
                video_writer = cv2.VideoWriter(str(save_path), fourcc, self.render_fps, frame_size)
                
                # Test if the video writer was initialized successfully
                if video_writer.isOpened():
                    break
                else:
                    video_writer.release()
                    video_writer = None
            except Exception as e:
                print(f"Failed to initialize video writer with codec: {e}")
                if video_writer:
                    video_writer.release()
                video_writer = None
        
        if video_writer is None:
            print("Error: Could not initialize video writer with any codec")
            return
        
        print(f"Saving video with {len(frames)} frames to {save_path}")
        
        successful_writes = 0
        for i, frame in enumerate(frames):
            try:
                # Ensure frame is in correct format
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                im = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Frame should already be the right size now, but double-check
                if im.shape[:2] != (frame_size[1], frame_size[0]):  # Note: OpenCV uses (height, width)
                    print(f"Warning: Frame {i} size mismatch, resizing")
                    im = cv2.resize(im, frame_size)
                
                success = video_writer.write(im)
                if success:
                    successful_writes += 1
                elif success is False:  # Explicitly check for False (not None)
                    print(f"Warning: Failed to write frame {i}")
                    
            except Exception as e:
                print(f"Error writing frame {i}: {e}")
        
        video_writer.release()
        print(f"Video saved successfully to {save_path} ({successful_writes}/{len(frames)} frames written)")


if __name__ == '__main__':
    import shutil

    xml_path = Path("xml_models/two_3bar_new_platform_config_2.xml")
    sim = AbstractMuJoCoSimulator(xml_path, visualize=True)

    for _ in range(500):
        sim.sim_step()
    sim.forward()

    qpos = sim.mjc_data.qpos.reshape(-1, 7)
    for i in range(qpos.shape[0]):
        debug_print(" ".join([str(round(qpos[i, j].item(), 7)) for j in range(7)]), "mujoco_simulation.py", True)  # Enable debug for main test
