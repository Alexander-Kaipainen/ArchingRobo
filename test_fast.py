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
q_stand = mod.build_desired_pose(model, mod.STAND_POSE)
q_squat = mod.build_desired_pose(model, mod.SQUAT_POSE)
q_grab = mod.build_desired_pose(model, mod.SQUAT_GRAB_POSE)
q_carry = mod.build_desired_pose(model, mod.STAND_CARRY_POSE)
dq_des = np.zeros(model.nu)

gains = mod.ACTUATOR_GAINS[:model.nu]
while len(gains) < model.nu:
    gains.append(mod.GAINS["hand"])

data.qpos[qpos_idx] = q_stand
data.qpos[2] = 0.78
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)

pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")

print("Initial Pelvis Z:", data.xpos[pelvis_id][2])

steps = int((0.25 * mod.SQUAT_PERIOD) / mod.SIMULATE_DT)
for _ in range(steps):
    q_des = mod.squat_cycle(data.time, q_stand, q_squat, q_grab, q_carry, mod.SQUAT_PERIOD)
    q = data.qpos[qpos_idx]
    dq = data.qvel[qvel_idx]
    tau = mod.pd_control(q_des, q, dq_des, dq, gains)
    tau = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    data.ctrl[:] = tau
    mujoco.mj_step(model, data)

print("Final Pelvis Z:", data.xpos[pelvis_id][2])
print("Final Left hand:", data.xpos[left_id])
print("Final Right hand:", data.xpos[right_id])
