"""Minimal Unitree G1 MuJoCo demo: stable stance + simple bicep curls.

What this script does:
- Keeps legs and torso in a fixed standing pose.
- Runs a slow curl by moving only elbow joints.
- Uses conservative PD gains and actuator clipping for safer behavior.

Run:
- Viewer:   python "import mujoco.py"
- Headless: python "import mujoco.py" --no-viewer
"""

import os
import sys
import time

# macOS viewer requires mjpython; re-exec once when needed.
if sys.platform == "darwin" and os.environ.get("_RUNNING_UNDER_MJPYTHON") != "1":
    import shutil

    os.environ["_RUNNING_UNDER_MJPYTHON"] = "1"
    _mjpython = shutil.which("mjpython") or os.path.join(
        os.path.dirname(sys.executable), "mjpython"
    )
    if _mjpython and os.path.isfile(_mjpython):
        _script = os.path.abspath(sys.argv[0])
        os.execv(_mjpython, [_mjpython, _script] + sys.argv[1:])
    else:
        print("WARNING: mjpython not found; viewer may fail on macOS.")

# This file name shadows the mujoco package; temporarily remove local path.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import mujoco
import mujoco.viewer
import numpy as np

if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_SCENE = os.path.join(SCRIPT_DIR, "unitree_g1", "mjcf", "scene_29dof_with_hand.xml")
HEADLESS = "--no-viewer" in sys.argv

SIM_DT = 0.002
VIEWER_DT = 0.02
STEPS_PER_FRAME = max(1, round(VIEWER_DT / SIM_DT))

CAM_DISTANCE = 2.8
CAM_ELEVATION = -18
CAM_AZIMUTH = 165
CAM_LOOKAT = np.array([0.0, 0.0, 0.85])

# For a very safe demo, keep the floating base fixed so the robot cannot tip.
LOCK_BASE = True
LOCKED_BASE_POS = np.array([0.0, 0.0, 0.78])
LOCKED_BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# Stable full-body standing reference.
BASE_POSE = {
    "left_hip_pitch_joint": -0.34,
    "left_knee_joint": 0.78,
    "left_ankle_pitch_joint": -0.44,
    "right_hip_pitch_joint": -0.34,
    "right_knee_joint": 0.78,
    "right_ankle_pitch_joint": -0.44,
    "waist_pitch_joint": 0.30,
    # Place hands to the side of the body permanently to safely avoid legs
    "left_shoulder_roll_joint": 0.30,
    "right_shoulder_roll_joint": -0.30,
    "left_shoulder_pitch_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
}

CURL_PERIOD = 8.0
# Safe, fixed range that prevents arms from going behind the back.
# 1.10 = arms hanging slightly forward
# 2.30 = fully curled up to chest
ELBOW_MIN = 0.7
ELBOW_MAX = -0.7


def build_actuator_qpos_index(model: mujoco.MjModel) -> np.ndarray:
    idx = np.zeros(model.nu, dtype=int)
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id, 0]
        idx[act_id] = model.jnt_qposadr[joint_id]
    return idx


def build_actuator_qvel_index(model: mujoco.MjModel) -> np.ndarray:
    idx = np.zeros(model.nu, dtype=int)
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id, 0]
        idx[act_id] = model.jnt_dofadr[joint_id]
    return idx


def build_joint_to_actuator_map(model: mujoco.MjModel) -> dict[str, int]:
    mapping = {}
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is not None:
            mapping[name] = act_id
    return mapping


def build_desired_pose(model: mujoco.MjModel, pose: dict[str, float]) -> np.ndarray:
    q_des = np.zeros(model.nu)
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name in pose:
            q_des[act_id] = pose[joint_name]
    return q_des


def gain_for_joint(joint_name: str) -> tuple[float, float]:
    if "hip" in joint_name or "knee" in joint_name:
        return 420.0, 18.0
    if "ankle" in joint_name:
        return 230.0, 11.0
    if "waist" in joint_name:
        return 320.0, 14.0
    if "shoulder" in joint_name:
        return 120.0, 5.0
    if "elbow" in joint_name:
        return 40.0, 2.5
    if "wrist" in joint_name:
        return 45.0, 2.0
    if "hand" in joint_name:
        return 20.0, 1.0
    return 40.0, 2.0


def build_gains(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    kp = np.zeros(model.nu)
    kd = np.zeros(model.nu)
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        kp[act_id], kd[act_id] = gain_for_joint(name)
    return kp, kd


def pd_control(
    q_des: np.ndarray,
    q: np.ndarray,
    dq_des: np.ndarray,
    dq: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return kp * (q_des - q) + kd * (dq_des - dq)


def apply_base_lock(data: mujoco.MjData) -> None:
    if not LOCK_BASE:
        return
    data.qpos[0:3] = LOCKED_BASE_POS
    data.qpos[3:7] = LOCKED_BASE_QUAT
    data.qvel[0:6] = 0.0


def main() -> None:
    if not os.path.exists(ROBOT_SCENE):
        print(f"ERROR: Scene not found: {ROBOT_SCENE}")
        return

    print(f"Loading scene: {ROBOT_SCENE}")
    model = mujoco.MjModel.from_xml_path(ROBOT_SCENE)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    qpos_idx = build_actuator_qpos_index(model)
    qvel_idx = build_actuator_qvel_index(model)
    joint_to_act = build_joint_to_actuator_map(model)
    kp, kd = build_gains(model)
    dq_des = np.zeros(model.nu)

    q_base = build_desired_pose(model, BASE_POSE)

    # Start from a standing configuration to avoid startup collapse.
    data.qpos[qpos_idx] = q_base
    apply_base_lock(data)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Hold every actuator at startup angle; override only elbows below.
    q_hold = data.qpos[qpos_idx].copy()

    left_elbow_act = joint_to_act.get("left_elbow_joint")
    right_elbow_act = joint_to_act.get("right_elbow_joint")
    if left_elbow_act is None or right_elbow_act is None:
        print("ERROR: Elbow actuators not found in model.")
        return

    if HEADLESS:
        print("Headless mode: running 6 seconds of simple bicep curls...")
        for _ in range(int(6.0 / SIM_DT)):
            apply_base_lock(data)
            mujoco.mj_forward(model, data)
            phase = 2.0 * np.pi * data.time / CURL_PERIOD
            # Smooth sine wave from ELBOW_MIN to ELBOW_MAX
            curl = 0.5 * (ELBOW_MAX + ELBOW_MIN) - 0.5 * (ELBOW_MAX - ELBOW_MIN) * np.cos(phase)
            
            q_des = q_hold.copy()
            q_des[left_elbow_act] = curl
            q_des[right_elbow_act] = curl

            q = data.qpos[qpos_idx]
            dq = data.qvel[qvel_idx]
            tau = pd_control(q_des, q, dq_des, dq, kp, kd)
            tau = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
        print("Done.")
        return

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance = CAM_DISTANCE
    viewer.cam.elevation = CAM_ELEVATION
    viewer.cam.azimuth = CAM_AZIMUTH
    viewer.cam.lookat[:] = CAM_LOOKAT

    print("Running: stable stance + elbow curls. Close viewer to stop.")
    while viewer.is_running():
        frame_t0 = time.perf_counter()
        for _ in range(STEPS_PER_FRAME):
            apply_base_lock(data)
            mujoco.mj_forward(model, data)
            phase = 2.0 * np.pi * data.time / CURL_PERIOD
            # Smooth sine wave from ELBOW_MIN to ELBOW_MAX
            curl = 0.5 * (ELBOW_MAX + ELBOW_MIN) - 0.5 * (ELBOW_MAX - ELBOW_MIN) * np.cos(phase)

            q_des = q_hold.copy()
            q_des[left_elbow_act] = curl
            q_des[right_elbow_act] = curl

            q = data.qpos[qpos_idx]
            dq = data.qvel[qvel_idx]
            tau = pd_control(q_des, q, dq_des, dq, kp, kd)
            tau = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

        dt = time.perf_counter() - frame_t0
        if dt < VIEWER_DT:
            time.sleep(VIEWER_DT - dt)
        viewer.sync()

    viewer.close()
    print("Simulation ended.")


if __name__ == "__main__":
    main()