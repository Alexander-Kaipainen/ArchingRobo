import mujoco
from importlib.machinery import SourceFileLoader
mod = SourceFileLoader("mj", "import mujoco.py").load_module()

model = mujoco.MjModel.from_xml_path("unitree_g1/mjcf/scene_barbell.xml")
data = mujoco.MjData(model)
mod.SQUAT_POSE["left_shoulder_pitch_joint"] = 0.0
mod.SQUAT_POSE["left_elbow_joint"] = 0.0
mod.SQUAT_POSE["left_shoulder_roll_joint"] = 0.0
q_squat = mod.build_desired_pose(model, mod.SQUAT_POSE)
data.qpos[mod.build_actuator_qpos_index(model)] = q_squat
data.qpos[2] = 0.4
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)
shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link")
wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
print(f"Shoulder Z: {data.xpos[shoulder_id][2]:.3f} X: {data.xpos[shoulder_id][0]:.3f}")
print(f"Wrist Z: {data.xpos[wrist_id][2]:.3f} X: {data.xpos[wrist_id][0]:.3f}")
