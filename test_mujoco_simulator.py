import mujoco
import mujoco.viewer

try:
    # Load your XML model directly
    model = mujoco.MjModel.from_xml_path("mujoco_physics_engine/xml_models/two_3bar_new_platform_config_1.xml")
    data = mujoco.MjData(model)
    
    print("MuJoCo model loaded successfully")
    print("Starting viewer...")
    
    # Use MuJoCo's built-in viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(1000):
            mujoco.mj_step(model, data)
            viewer.sync()
            
            if i % 100 == 0:
                print(f"Step {i}")
                
except Exception as e:
    print(f"MuJoCo visualization test failed: {e}")