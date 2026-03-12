import mujoco
import numpy as np

import mujoco.viewer

# Load Unitree robot model (you'll need the XML file)
model = mujoco.MjModel.from_xml_path("path/to/unitree_model.xml")
data = mujoco.MjData(model)

# Initialize viewer
viewer = mujoco.viewer.launch_passive(model, data)

# Simulation loop
while viewer.is_running():
    # Set control inputs (example: joint positions)
    data.ctrl[:] = np.array([0.0] * model.nu)
    
    # Step simulation
    mujoco.mj_step(model, data)
    
    # Sync viewer
    viewer.sync()

viewer.close()