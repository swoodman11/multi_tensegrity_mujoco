import json
import random
from pathlib import Path
from typing import List, Optional

import cv2
import mujoco
import numpy as np


class AbstractMuJoCoSimulator:
    """
    MuJoCo interface
    """

    def __init__(self,
                 xml_path: Path,
                 visualize: bool = False,
                 render_size: (int, int) = (480, 640),
                 render_fps: int = 50):
        self.xml_path = xml_path
        self.visualize = visualize
        self.mjc_model = self._load_model_from_xml(xml_path)
        self.mjc_data = mujoco.MjData(self.mjc_model)
        self.renderer = mujoco.Renderer(self.mjc_model, render_size[0], render_size[1]) if visualize else None
        self.render_fps = render_fps
        self.states = []
        self.time = 0
        self.dt = self.mjc_model.opt.timestep

    def reset(self):
        self.mjc_model = self._load_model_from_xml(self.xml_path)
        self.mjc_data = mujoco.MjData(self.mjc_model)

    def _load_model_from_xml(self, xml_path: Path) -> mujoco.MjModel:
        model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        return model

    def sim_step(self):
        mujoco.mj_step(self.mjc_model, self.mjc_data)

    def forward(self):
        mujoco.mj_forward(self.mjc_model, self.mjc_data)

    def render_frame(self, view='camera'):
        self.renderer.update_scene(self.mjc_data, view)
        frame = self.renderer.render().copy()
        return frame

    def save_video(self, save_path: Path, frames: list):
        frame_size = (self.renderer.width, self.renderer.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_path.as_posix(), fourcc, self.render_fps, frame_size)

        for i, frame in enumerate(frames):
            im = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(im)

        video_writer.release()


if __name__ == '__main__':
    import shutil

    xml_path = Path("xml_models/two_3bar_new_platform_config_1.xml")
    sim = AbstractMuJoCoSimulator(xml_path, visualize=True)

    for _ in range(500):
        sim.sim_step()
    sim.forward()

    qpos = sim.mjc_data.qpos.reshape(-1, 7)
    for i in range(qpos.shape[0]):
        print(" ".join([str(round(qpos[i, j].item(), 7)) for j in range(7)]))
