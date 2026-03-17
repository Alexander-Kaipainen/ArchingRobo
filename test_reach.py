import mujoco
import numpy as np
import time
from importlib.machinery import SourceFileLoader
mod = SourceFileLoader("mujoco_script", "import mujoco.py").load_module()

model = mujoco.MjModel.from_xml_path("unitree_g1/mjcf/scene_barbell.xml")
data = mujoco.MjData(model)
model.opt.timestep = mod.SIMULATE_DT

qpos_idx = mod.build_actuator_qpos_index(model)
qvel_idx = mod.build_actuator_qvel_index(model)
q_squat = mod.build_desired_pose(model, mod.SQUAT_POSE)
dq_des = np.zeros(model.nu)

gains = mod.ACTUATOR_GAINS[:model.nu]
while len(gains) < model.nu:
    gains.append(mod.GAINS["hand"])

data.qpos[qpos_idx] = q_squat

# Try to drop it on the ground and let it settle in squat pose
data.qpos[2] = 0.5  # z height
data.qpos[3] = 1.0  # quat w

mujoco.mj_forward(model, data)

for _ in range(2000):
    q = data.qpos[qpos_idx]
    dq = data.qvel[qvel_idx]
    tau = mod.pd_control(q_squat, q, dq_des, dq, gains)
    tau = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    data.ctrl[:] = tau
    mujoco.mj_step(model, data)

left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
print("Left wrist pos:", data.xpos[left_id])
print("Right wrist pos:", data.xpos[right_id])
print("Left wrist mat:\n", data.xmat[left_id].reshape(3,3))

barbell_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "barbell")
print("Barbell pos:", data.xpos[barbell_id])

pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
print("Pelvis pos:", data.xpos[pelvis_id])
