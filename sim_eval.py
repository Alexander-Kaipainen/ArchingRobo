import mujoco
from importlib.machinery import SourceFileLoader
mod = SourceFileLoader("mj", "import mujoco.py").load_module()
import numpy as np

model = mujoco.MjModel.from_xml_path("unitree_g1/mjcf/scene_barbell.xml")
data = mujoco.MjData(model)
model.opt.timestep = mod.SIMULATE_DT

qpos_idx = mod.build_actuator_qpos_index(model)
qvel_idx = mod.build_actuator_qvel_index(model)
q_stand = mod.build_desired_pose(model, mod.STAND_POSE)
q_squat = mod.build_desired_pose(model, mod.SQUAT_POSE)
q_grab = q_squat.copy()
q_carry = mod.build_desired_pose(model, mod.STAND_CARRY_POSE)
dq_des = np.zeros(model.nu)

gains = mod.ACTUATOR_GAINS[:model.nu]
while len(gains) < model.nu: gains.append(mod.GAINS["hand"])

data.qpos[qpos_idx] = q_stand
data.qpos[2] = 0.78
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)

pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

for step in range(int(mod.SQUAT_PERIOD / mod.SIMULATE_DT)):
    phase = (data.time % mod.SQUAT_PERIOD) / mod.SQUAT_PERIOD
    
    q_des = mod.squat_cycle(data.time, q_stand, q_squat, q_grab, q_carry)
    q = data.qpos[qpos_idx]
    dq = data.qvel[qvel_idx]
    tau = mod.pd_control(q_des, q, dq_des, dq, gains)
    tau = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    data.ctrl[:] = tau
    mujoco.mj_step(model, data)
    
    com = data.subtree_com[0]
    left_foot = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")]
    right_foot = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")]
    mid_foot = (left_foot + right_foot) / 2
    
    if data.xpos[pelvis_id][2] < 0.3:
        print(f"Fallen precisely at phase {phase:.3f}")
        print(f"CoM X: {com[0]:.3f}, Mid-foot X: {mid_foot[0]:.3f} -> Diff: {com[0]-mid_foot[0]:.3f}")
        break
else:
    print("Success")
