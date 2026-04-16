import time
from importlib.machinery import SourceFileLoader
mod = SourceFileLoader("mj", "import mujoco.py").load_module()

import mujoco
import mujoco.viewer
import numpy as np

# ── Load model ────────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(mod.ROBOT_SCENE)
data  = mujoco.MjData(model)
model.opt.timestep = mod.SIM_DT

# ── Build indices / gains ─────────────────────────────────────────────────────
qpos_idx     = mod.build_actuator_qpos_index(model)
qvel_idx     = mod.build_actuator_qvel_index(model)
joint_to_act = mod.build_joint_to_actuator_map(model)
kp, kd       = mod.build_gains(model)
dq_des       = np.zeros(model.nu)

# ── Standing pose ──────────────────────────────────────────────────────────────
q_base = mod.build_desired_pose(model, mod.BASE_POSE)

# Initialise robot in standing position
data.qpos[qpos_idx] = q_base
mod.apply_base_lock(data)
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

# Hold pose for all joints; only elbows will be overridden each step
q_hold = data.qpos[qpos_idx].copy()

left_elbow_act  = joint_to_act.get("left_elbow_joint")
right_elbow_act = joint_to_act.get("right_elbow_joint")
if left_elbow_act is None or right_elbow_act is None:
    raise RuntimeError("Elbow actuators not found in model.")

# ── Viewer ────────────────────────────────────────────────────────────────────
viewer = mujoco.viewer.launch_passive(model, data)
viewer.cam.distance  = mod.CAM_DISTANCE
viewer.cam.elevation = mod.CAM_ELEVATION
viewer.cam.azimuth   = mod.CAM_AZIMUTH
viewer.cam.lookat[:] = mod.CAM_LOOKAT

print("Running: continuous bicep curls. Close the viewer to stop.")

# ── Continuous bicep-curl loop ─────────────────────────────────────────────────
while viewer.is_running():
    frame_t0 = time.perf_counter()

    for _ in range(mod.STEPS_PER_FRAME):
        mod.apply_base_lock(data)
        mujoco.mj_forward(model, data)

        # Smooth sine wave: ELBOW_MIN → ELBOW_MAX → ELBOW_MIN → …
        phase = 2.0 * np.pi * data.time / mod.CURL_PERIOD
        curl  = (0.5 * (mod.ELBOW_MAX + mod.ELBOW_MIN)
                 - 0.5 * (mod.ELBOW_MAX - mod.ELBOW_MIN) * np.cos(phase))

        q_des = q_hold.copy()
        q_des[left_elbow_act]  = curl
        q_des[right_elbow_act] = curl

        q   = data.qpos[qpos_idx]
        dq  = data.qvel[qvel_idx]
        tau = mod.pd_control(q_des, q, dq_des, dq, kp, kd)
        tau = np.clip(tau, model.actuator_ctrlrange[:, 0],
                           model.actuator_ctrlrange[:, 1])
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

    dt = time.perf_counter() - frame_t0
    if dt < mod.VIEWER_DT:
        time.sleep(mod.VIEWER_DT - dt)
    viewer.sync()

viewer.close()
print("Simulation ended.")
