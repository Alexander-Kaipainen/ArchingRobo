"""
Unitree G1 MuJoCo Simulation - Integrated with ARK (Robotics-Ark/ark_unitree_g1)

This script loads the Unitree G1 humanoid robot (29 DOF + hands) in MuJoCo
using the model files from the ark_unitree_g1 repository.

G1 Robot Structure:
  - Legs: 12 DOF (6 per leg: hip pitch/roll/yaw, knee, ankle pitch/roll)
  - Waist: 3 DOF (yaw, roll, pitch)
  - Arms: 14 DOF (7 per arm: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
  - Hands: optional (Dex3-1 dexterous hands)

For full SDK-based control, see:
  ark_unitree_g1/tests/g1_mujoco_sim/unitree_mujoco.py
"""

import sys
import os
import time

# ---------------------------------------------------------------------------
# macOS: mujoco.viewer.launch_passive requires mjpython.
# Re-exec once via the mjpython wrapper if we haven't already.
# We use a sentinel env-var because sys.executable stays as "pythonX.Y" even
# inside mjpython (the wrapper sets argv[0]=sys.executable before execve-ing
# the real binary, so basename(sys.executable) == "mjpython" is never true).
# ---------------------------------------------------------------------------
if sys.platform == "darwin" and os.environ.get("_RUNNING_UNDER_MJPYTHON") != "1":
    import shutil
    os.environ["_RUNNING_UNDER_MJPYTHON"] = "1"
    _mjpython = shutil.which("mjpython") or os.path.join(
        os.path.dirname(sys.executable), "mjpython"
    )
    if _mjpython and os.path.isfile(_mjpython):
        # Pass the absolute script path so mjpython doesn't search cwd
        _script = os.path.abspath(sys.argv[0])
        os.execv(_mjpython, [_mjpython, _script] + sys.argv[1:])
    else:
        print(
            "WARNING: mjpython not found — viewer may fail on macOS.\n"
            "Install with:  pip install mujoco"
        )

# ---------------------------------------------------------------------------
# IMPORTANT: This file is named "import mujoco.py", which shadows the real
# mujoco package. We work around this by removing our directory from sys.path
# temporarily, importing mujoco, then restoring it.
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import mujoco
import mujoco.viewer
import numpy as np

# Restore script dir so relative imports still work if needed
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# ---------------------------------------------------------------------------
# Configuration (mirrors ark_unitree_g1/tests/g1_mujoco_sim/config.py)
# ---------------------------------------------------------------------------
ROBOT = "g1"

# Resolve the path to the Unitree G1 MuJoCo scene XML
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_SCENE = os.path.join(
    SCRIPT_DIR, f"unitree_{ROBOT}", "mjcf", "scene_29dof_with_hand.xml"
)

# Set True to run headless (no viewer window — just load + step test)
# Also activated with:  python "import mujoco.py" --no-viewer
HEADLESS = "--no-viewer" in sys.argv

# Simulation parameters
SIMULATE_DT = 0.002    # 500 Hz physics step (matching ARK default)
VIEWER_DT   = 0.02     # 50 Hz viewer refresh

# Speed scale: 1.0 = real-time, 0.5 = half speed, 0.1 = slow-motion.
# Without this the loop runs only 1 physics step per viewer frame
# (SIMULATE_DT/VIEWER_DT = 0.1×), so the default of 1.0 fixes that.
TIMESCALE = 1.0

# How many physics steps to run per viewer frame so wall-clock matches
# TIMESCALE.  e.g. VIEWER_DT=0.02, SIMULATE_DT=0.002, TIMESCALE=1.0 → 10.
STEPS_PER_FRAME = max(1, round(VIEWER_DT / SIMULATE_DT * TIMESCALE))

# Camera defaults
CAM_DISTANCE  = 3.0
CAM_ELEVATION = -20
CAM_AZIMUTH   = 180
CAM_LOOKAT    = np.array([0.0, 0.0, 0.8])

# ---------------------------------------------------------------------------
# G1 Joint Index Map (29 DOF body, from ark_unitree_g1 motor_indices)
# ---------------------------------------------------------------------------
G1_JOINT_NAMES = [
    # Left leg (6)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# ---------------------------------------------------------------------------
# Motor gain profiles  (Kp = spring stiffness, Kd = damping)
# Rule of thumb: Kd ≈ 2*sqrt(Kp * link_inertia).  Higher Kp = stiffer,
# higher Kd = more damping (less oscillation).  Humanoids need high leg gains.
# ---------------------------------------------------------------------------
GAINS = {
    "leg":   {"kp": 500.0, "kd": 20.0},
    "ankle": {"kp": 200.0, "kd": 10.0},
    "waist": {"kp": 400.0, "kd": 15.0},
    "arm":   {"kp": 150.0, "kd":  6.0},
    "wrist": {"kp":  60.0, "kd":  2.0},
    "hand":  {"kp":  10.0, "kd":  0.5},
}

# Per-actuator gain assignment (order matches <actuator> block in XML)
# fmt: off
ACTUATOR_GAINS = [
    # Left leg
    GAINS["leg"], GAINS["leg"], GAINS["leg"],    # hip pitch/roll/yaw
    GAINS["leg"],                                  # knee
    GAINS["ankle"], GAINS["ankle"],               # ankle pitch/roll
    # Right leg
    GAINS["leg"], GAINS["leg"], GAINS["leg"],
    GAINS["leg"],
    GAINS["ankle"], GAINS["ankle"],
    # Waist
    GAINS["waist"], GAINS["waist"], GAINS["waist"],
    # Left arm
    GAINS["arm"], GAINS["arm"], GAINS["arm"],    # shoulder pitch/roll/yaw
    GAINS["arm"],                                  # elbow
    GAINS["wrist"], GAINS["wrist"], GAINS["wrist"],  # wrist
    # Left hand (7 joints: thumb0/1/2, middle0/1, index0/1)
    GAINS["hand"], GAINS["hand"], GAINS["hand"],
    GAINS["hand"], GAINS["hand"],
    GAINS["hand"], GAINS["hand"],
    # Right arm
    GAINS["arm"], GAINS["arm"], GAINS["arm"],
    GAINS["arm"],
    GAINS["wrist"], GAINS["wrist"], GAINS["wrist"],
    # Right hand
    GAINS["hand"], GAINS["hand"], GAINS["hand"],
    GAINS["hand"], GAINS["hand"],
    GAINS["hand"], GAINS["hand"],
]
# fmt: on

# ---------------------------------------------------------------------------
# Default standing pose — desired joint angles in radians (one per actuator).
# Joints not listed default to 0.0 (straight/neutral).
# Tweak these to change the robot's standing posture.
# ---------------------------------------------------------------------------
# Standing pose: hip_pitch + knee + ankle must sum to ~0 so the torso stays
# upright and the CoM sits over the feet.
#   hip_pitch (negative = lean forward) + knee - ankle ≈ 0
# e.g.  -0.4  +  0.8  - 0.4  = 0  ✓
STAND_POSE = {
    # Left leg
    "left_hip_pitch_joint":   -0.4,
    "left_knee_joint":         0.8,
    "left_ankle_pitch_joint": -0.4,
    # Right leg (mirror)
    "right_hip_pitch_joint":   -0.4,
    "right_knee_joint":         0.8,
    "right_ankle_pitch_joint": -0.4,
    # Arms slightly out to improve lateral stability
    "left_shoulder_roll_joint":   0.2,
    "right_shoulder_roll_joint": -0.2,
}

# Deep squat pose — roughly half-way down, limited by joint ranges
SQUAT_POSE = {
    "left_hip_pitch_joint":   -0.9,
    "left_knee_joint":         1.8,
    "left_ankle_pitch_joint": -0.9,
    "right_hip_pitch_joint":   -0.9,
    "right_knee_joint":         1.8,
    "right_ankle_pitch_joint": -0.9,
    # Arms rise slightly when squatting for balance
    "left_shoulder_pitch_joint":  -0.3,
    "right_shoulder_pitch_joint": -0.3,
    "left_shoulder_roll_joint":   0.2,
    "right_shoulder_roll_joint": -0.2,
}

# Squat cycle period in seconds (half down, half up)
SQUAT_PERIOD = 4.0

# Alias used by the rest of the script
DEFAULT_POSE = STAND_POSE


def squat_cycle(t: float, q_stand: np.ndarray, q_squat: np.ndarray,
                period: float = SQUAT_PERIOD) -> np.ndarray:
    """Smoothly interpolate between standing and squat pose.

    Uses a raised-cosine so acceleration is zero at both ends (no jerk):
        alpha = (1 - cos(2*pi*t/period)) / 2   ∈ [0, 1]
    alpha=0 → standing, alpha=1 → full squat, then back to 0.
    """
    alpha = (1.0 - np.cos(2.0 * np.pi * t / period)) / 2.0
    return (1.0 - alpha) * q_stand + alpha * q_squat


def build_actuator_qpos_index(model: mujoco.MjModel) -> np.ndarray:
    """Return an array mapping actuator index → qpos index.

    MuJoCo lays out qpos as:
      [0:7]   floating base (3 pos + 4 quat)
      [7:]    one slot per 1-DOF joint, in joint-definition order

    Each <motor> actuator references exactly one joint; this function
    resolves that joint's qpos address via model.jnt_qposadr.
    """
    idx = np.zeros(model.nu, dtype=int)
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id, 0]   # joint driven by actuator
        idx[act_id] = model.jnt_qposadr[joint_id]
    return idx


def build_actuator_qvel_index(model: mujoco.MjModel) -> np.ndarray:
    """Return an array mapping actuator index → qvel index (DOF address)."""
    idx = np.zeros(model.nu, dtype=int)
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id, 0]
        idx[act_id] = model.jnt_dofadr[joint_id]
    return idx


def build_desired_pose(model: mujoco.MjModel, pose_dict: dict) -> np.ndarray:
    """Convert a {joint_name: angle_rad} dict into a per-actuator array.

    Actuators not present in pose_dict are set to 0.0 (neutral).
    """
    q_des = np.zeros(model.nu)
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name in pose_dict:
            q_des[act_id] = pose_dict[name]
    return q_des


def pd_control(
    q_des:  np.ndarray,
    q:      np.ndarray,
    dq_des: np.ndarray,
    dq:     np.ndarray,
    gains:  list,
) -> np.ndarray:
    """Compute PD torques for every actuator.

    tau_i = Kp_i * (q_des_i - q_i) + Kd_i * (dq_des_i - dq_i)

    Args:
        q_des:  desired joint positions  [nu]
        q:      current joint positions  [nu]   (sliced from data.qpos)
        dq_des: desired joint velocities [nu]   (usually zeros)
        dq:     current joint velocities [nu]   (sliced from data.qvel)
        gains:  list of {"kp": float, "kd": float} dicts, length nu
    Returns:
        torques array [nu]
    """
    kp = np.array([g["kp"] for g in gains])
    kd = np.array([g["kd"] for g in gains])
    return kp * (q_des - q) + kd * (dq_des - dq)


def print_scene_info(model: mujoco.MjModel):
    """Print useful information about the loaded robot model."""
    print("=" * 60)
    print(f"  Unitree {ROBOT.upper()} — MuJoCo Scene Info")
    print("=" * 60)
    print(f"  Actuators (nu):  {model.nu}")
    print(f"  DoF (nv):        {model.nv}")
    print(f"  Bodies (nbody):  {model.nbody}")
    print(f"  Joints (njnt):   {model.njnt}")
    print(f"  Timestep:        {model.opt.timestep:.4f} s")
    print("-" * 60)

    # List joints
    print("  Joints:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"    [{i:2d}] {name}")
    print("=" * 60)


def main():
    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if not os.path.exists(ROBOT_SCENE):
        print(f"ERROR: Robot scene not found at:\n  {ROBOT_SCENE}")
        print("Make sure the ark_unitree_g1 repo was cloned inside this folder.")
        return

    print(f"Loading Unitree {ROBOT.upper()} from:\n  {ROBOT_SCENE}")
    t0 = time.perf_counter()
    model = mujoco.MjModel.from_xml_path(ROBOT_SCENE)
    data = mujoco.MjData(model)
    print(f"  Model loaded in {time.perf_counter() - t0:.2f}s")

    # Set simulation timestep
    model.opt.timestep = SIMULATE_DT

    # Print scene info
    print_scene_info(model)

    # ------------------------------------------------------------------
    # Build PD controller indices and desired pose
    # ------------------------------------------------------------------
    qpos_idx = build_actuator_qpos_index(model)   # actuator → qpos slot
    qvel_idx = build_actuator_qvel_index(model)   # actuator → qvel slot
    q_des    = build_desired_pose(model, DEFAULT_POSE)  # target angles [nu]
    dq_des   = np.zeros(model.nu)                 # target velocities (zero)

    # Trim/extend gain list to actual actuator count
    gains = ACTUATOR_GAINS[:model.nu]
    while len(gains) < model.nu:
        gains.append(GAINS["hand"])

    # ------------------------------------------------------------------
    # Initialise qpos to the desired standing pose so the robot starts
    # balanced rather than collapsing from the default all-zeros state.
    # ------------------------------------------------------------------
    data.qpos[qpos_idx] = q_des          # set joint angles
    data.qpos[2] = 0.78                  # z height: ~0.78 m clears the floor
    data.qpos[3] = 1.0                   # quaternion w=1 → upright orientation
    data.qpos[4:7] = 0.0                 # quaternion x,y,z = 0
    mujoco.mj_forward(model, data)       # propagate kinematics

    # Quick physics sanity check (10 steps, no viewer)
    print("Running 10 physics steps (sanity check)...")
    for i in range(10):
        mujoco.mj_step(model, data)
    print(f"  OK — sim time = {data.time:.4f}s\n")

    if HEADLESS:
        print("Headless mode — skipping viewer. Model loaded successfully!")
        return

    # Pre-build the two pose arrays for the squat cycle
    q_stand = build_desired_pose(model, STAND_POSE)
    q_squat = build_desired_pose(model, SQUAT_POSE)

    # ------------------------------------------------------------------
    # Install a control callback so the active viewer calls our PD
    # controller on every physics step automatically.
    # ------------------------------------------------------------------
    def controller(m, d):
        q_target = squat_cycle(d.time, q_stand, q_squat)
        q  = d.qpos[qpos_idx]
        dq = d.qvel[qvel_idx]
        tau = pd_control(q_target, q, dq_des, dq, gains)
        tau = np.clip(tau, m.actuator_ctrlrange[:, 0],
                           m.actuator_ctrlrange[:, 1])
        d.ctrl[:] = tau

    mujoco.set_mjcb_control(controller)

    # ------------------------------------------------------------------
    # Launch active (blocking) viewer — more reliable on Windows
    # ------------------------------------------------------------------
    print("\nLaunching MuJoCo viewer — robot will squat and stand on repeat.")
    print("Close the viewer window to stop.\n")
    mujoco.viewer.launch(model, data)
    print("Simulation ended.")


if __name__ == "__main__":
    main()