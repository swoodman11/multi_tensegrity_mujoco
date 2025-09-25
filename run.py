from pathlib import Path
from datetime import datetime

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


if __name__ == '__main__':
    run_single_sim()