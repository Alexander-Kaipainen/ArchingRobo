import mujoco
from importlib.machinery import SourceFileLoader
mod = SourceFileLoader("mj", "import mujoco.py").load_module()

model = mujoco.MjModel.from_xml_path("unitree_g1/mjcf/scene_barbell.xml")
data = mujoco.MjData(model)

mod.SQUAT_POSE["left_shoulder_pitch_joint"] = -0.5
mod.SQUAT_POSE["right_shoulder_pitch_joint"] = -0.5
mod.SQUAT_POSE["left_elbow_joint"] = 0.0
mod.SQUAT_POSE["right_elbow_joint"] = 0.0

# Deep hip/knees
mod.SQUAT_POSE["waist_pitch_joint"] = 1.0
mod.SQUAT_POSE["left_hip_pitch_joint"] = -1.5
mod.SQUAT_POSE["right_hip_pitch_joint"] = -1.5
mod.SQUAT_POSE["left_knee_joint"] = 2.6
mod.SQUAT_POSE["right_knee_joint"] = 2.6
mod.SQUAT_POSE["left_ankle_pitch_joint"] = -1.1
mod.SQUAT_POSE["right_ankle_pitch_joint"] = -1.1

qpos_idx = mod.build_actuator_qpos_index(model)
qvel_idx = mod.build_actuator_qvel_index(model)
q_squat = mod.build_desired_pose(model, mod.SQUAT_POSE)
dq_des = __import__('numpy').zeros(model.nu)

gains = mod.ACTUATOR_GAINS[:model.nu]
while len(gains) < model.nu: gains.append(mod.GAINS["hand"])

data.qpos[qpos_idx] = q_squat
data.qpos[2] = 0.78
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)

for _ in range(1000):
    q = data.qpos[qpos_idx]
    dq = data.qvel[qvel_idx]
    tau = mod.pd_control(q_squat, q, dq_des, dq, gains)
    data.ctrl[:] = tau
    mujoco.mj_step(model, data)

left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
print(f"After physics Deep squat wrist is at X={data.xpos[left_id][0]:.3f}, Z={data.xpos[left_id][2]:.3f}, Y={data.xpos[left_id][1]:.3f}")
print(f"Base Z is {data.qpos[2]:.3f} vs Pelvis {data.xpos[1][2]:.3f}")

