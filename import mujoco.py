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
    SCRIPT_DIR, f"unitree_{ROBOT}", "mjcf", "scene_barbell.xml"
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
# CoM compensation note:
# A forward waist lean of θ shifts the upper-body CoM forward by
# ~L_upper * sin(θ).  Reducing ankle dorsiflexion by a similar fraction
# (making ankle_pitch less negative) shifts the contact reaction point
# backward, keeping the net CoM over the support polygon.
# Empirically: ankle_offset ≈ +0.05 per 0.1 rad of waist_pitch works well.
STAND_POSE = {
    # Left leg — ankle slightly less negative to compensate forward lean
    "left_hip_pitch_joint":   -0.28,
    "left_knee_joint":         0.62,
    "left_ankle_pitch_joint": -0.24,
    # Right leg (mirror)
    "right_hip_pitch_joint":   -0.28,
    "right_knee_joint":         0.62,
    "right_ankle_pitch_joint": -0.24,
    # Waist pitched forward for stability (keeps CoM over feet during motion)
    "waist_pitch_joint":        0.24,
    # Arms slightly out to improve lateral stability
    "left_shoulder_roll_joint":   0.2,
    "right_shoulder_roll_joint": -0.2,
}

# Deep squat pose — roughly half-way down, limited by joint ranges
SQUAT_POSE = {
    "left_hip_pitch_joint":   -1.5,
    "right_hip_pitch_joint":  -1.5,
    "left_knee_joint":         2.6,
    "right_knee_joint":        2.6,
    "left_ankle_pitch_joint": -1.1,
    "right_ankle_pitch_joint":-1.1,
    # Forward-arched back for setup over bar (reduced to avoid toe overload)
    "waist_pitch_joint":        0.92,
    # Arms lowered/forward and moved outward to clear the thighs
    "left_shoulder_pitch_joint":   -0.45,
    "right_shoulder_pitch_joint":  -0.45,
    "left_shoulder_roll_joint":     0.35,
    "right_shoulder_roll_joint":   -0.35,
    "left_elbow_joint":            0.05,
    "right_elbow_joint":           0.05,
    # Rotate hands 90° so approach is perpendicular to the bar
    "left_wrist_roll_joint":       1.57,
    "right_wrist_roll_joint":      1.57,
}

# The hand closed to grip the barbell
hand_grip = {
    "left_hand_middle_0_joint": 1.0,
    "left_hand_index_0_joint":  1.0,
    "left_hand_thumb_0_joint":  1.0,
    "left_hand_thumb_1_joint":  1.0,
    "right_hand_middle_0_joint": 1.0,
    "right_hand_index_0_joint":  1.0,
    "right_hand_thumb_0_joint":  1.0,
    "right_hand_thumb_1_joint":  1.0,
}

SQUAT_GRAB_POSE = SQUAT_POSE.copy()
SQUAT_GRAB_POSE.update(hand_grip)

# After grab: drop hips a bit more before pressing up with legs
GRAB_SINK_POSE = SQUAT_GRAB_POSE.copy()
GRAB_SINK_POSE.update({
    "left_hip_pitch_joint":    -1.65,
    "right_hip_pitch_joint":   -1.65,
    "left_knee_joint":          2.85,
    "right_knee_joint":         2.85,
    "left_ankle_pitch_joint":  -1.05,
    "right_ankle_pitch_joint": -1.05,
    "waist_pitch_joint":        1.05,
})

# Standing back up but holding the barbell
STAND_CARRY_POSE = STAND_POSE.copy()
STAND_CARRY_POSE.update({
    "waist_pitch_joint":          0.34,
    "left_hip_pitch_joint":      -0.24,
    "right_hip_pitch_joint":     -0.24,
    "left_knee_joint":            0.52,
    "right_knee_joint":           0.52,
    "left_ankle_pitch_joint":    -0.22,
    "right_ankle_pitch_joint":   -0.22,
    "left_shoulder_pitch_joint":   -0.5,
    "right_shoulder_pitch_joint":  -0.5,
    "left_shoulder_roll_joint":    0.0,
    "right_shoulder_roll_joint":   0.0,
    "left_elbow_joint":            1.0,
    "right_elbow_joint":           1.0,
    "left_wrist_roll_joint":       1.57,
    "right_wrist_roll_joint":      1.57,
})
STAND_CARRY_POSE.update(hand_grip)

# Final reach micro-pose blended in only when wrist is very close to bar
FINAL_REACH_POSE = SQUAT_GRAB_POSE.copy()
FINAL_REACH_POSE.update({
    "waist_pitch_joint":            0.88,
    "left_shoulder_pitch_joint":   -0.60,
    "right_shoulder_pitch_joint":  -0.60,
    "left_shoulder_roll_joint":     0.35,
    "right_shoulder_roll_joint":   -0.35,
    "left_elbow_joint":             0.35,
    "right_elbow_joint":            0.35,
})

# Squat cycle period in seconds (half down, half up)
SQUAT_PERIOD = 8.0

# Grab timing inside normalized cycle phase [0, 1)
GRAB_PHASE_ON = 0.30
GRAB_PHASE_OFF = 0.92

# Reach a little before grab and only attach when actually close.
PRE_GRAB_PHASE_ON = 0.16
GRAB_ATTACH_DIST = 0.14

# Small local-frame correction so the bar sits in the hands instead of
# clipping backward into knees during stand-up.
BAR_ATTACH_NUDGE_LOCAL = np.array([0.03, 0.00, 0.06])

# Approximate lateral hand placement on bar, in bar-local frame (meters).
BAR_GRIP_OFFSET_Y = 0.18

# When left wrist gets close to the bar, blend in FINAL_REACH_POSE.
# Distances are in meters.
APPROACH_DIST_START = 0.45
APPROACH_DIST_END = 0.20

# Alias used by the rest of the script
DEFAULT_POSE = STAND_POSE


def squat_cycle(
    t: float,
    q_stand: np.ndarray,
    q_squat: np.ndarray,
    q_grab: np.ndarray,
    q_sink: np.ndarray,
    q_carry: np.ndarray,
    period: float = SQUAT_PERIOD,
) -> np.ndarray:
    """One-shot sequence: stand -> squat -> grab -> sink -> leg press -> drop -> stand."""
    phase = min(max(t / period, 0.0), 1.0)  # clamp to [0, 1], no looping
    
    # 0.0 -> 0.2: Stand to Squat
    if phase < 0.2:
        alpha = (1.0 - np.cos(np.pi * (phase / 0.2))) / 2.0
        return (1.0 - alpha) * q_stand + alpha * q_squat
        
    # 0.2 -> 0.3: Squat to Grab (close hands)
    elif phase < 0.3:
        alpha = (1.0 - np.cos(np.pi * ((phase - 0.2) / 0.1))) / 2.0
        return (1.0 - alpha) * q_squat + alpha * q_grab
        
    # 0.3 -> 0.38: Grab to Sink (pull hips down)
    elif phase < 0.38:
        alpha = (1.0 - np.cos(np.pi * ((phase - 0.3) / 0.08))) / 2.0
        return (1.0 - alpha) * q_grab + alpha * q_sink

    # 0.38 -> 0.46: Hold sink briefly
    elif phase < 0.46:
        return q_sink

    # 0.46 -> 0.62: Sink to Carry (press with legs)
    elif phase < 0.62:
        alpha = (1.0 - np.cos(np.pi * ((phase - 0.46) / 0.16))) / 2.0
        return (1.0 - alpha) * q_sink + alpha * q_carry

    # 0.62 -> 0.74: Stand tall, holding
    elif phase < 0.74:
        return q_carry
        
    # 0.74 -> 0.9: Carry to Grab (squat back down to release)
    elif phase < 0.9:
        alpha = (1.0 - np.cos(np.pi * ((phase - 0.74) / 0.16))) / 2.0
        return (1.0 - alpha) * q_carry + alpha * q_grab
        
    # 0.9 -> 1.0: Release and Stand Up
    else:
        alpha = (1.0 - np.cos(np.pi * ((phase - 0.9) / 0.1))) / 2.0
        return (1.0 - alpha) * q_grab + alpha * q_stand


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


def build_joint_to_actuator_map(model: mujoco.MjModel) -> dict:
    """Return {joint_name: actuator_id} for 1-DOF motorized joints."""
    mapping = {}
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is not None:
            mapping[name] = act_id
    return mapping


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


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    return quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))[1:4]


def normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def proximity_blend(distance: float, start: float, end: float) -> float:
    """Return smooth blend alpha in [0,1] as distance shrinks start->end."""
    if start <= end:
        return 0.0
    x = (start - distance) / (start - end)
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


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

    # ------------------------------------------------------------------
    # Launch passive viewer
    # ------------------------------------------------------------------
    print("Launching MuJoCo viewer (this may take a moment)...")
    viewer = mujoco.viewer.launch_passive(model, data)

    # Configure camera
    viewer.cam.distance  = CAM_DISTANCE
    viewer.cam.elevation = CAM_ELEVATION
    viewer.cam.azimuth   = CAM_AZIMUTH
    viewer.cam.lookat[:] = CAM_LOOKAT

    # Pre-build the pose arrays for the squat cycle
    q_stand = build_desired_pose(model, STAND_POSE)
    q_squat = build_desired_pose(model, SQUAT_POSE)
    q_grab  = build_desired_pose(model, SQUAT_GRAB_POSE)
    q_sink  = build_desired_pose(model, GRAB_SINK_POSE)
    q_reach = build_desired_pose(model, FINAL_REACH_POSE)
    q_carry = build_desired_pose(model, STAND_CARRY_POSE)

    print("\nSimulation running — robot will pick up barbell and stand on repeat.")
    print("Close the viewer window to stop.\n")

    left_wrist_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    right_wrist_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    barbell_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "barbell")
    barbell_joint_id = model.body_jntadr[barbell_body_id] if barbell_body_id >= 0 else -1
    barbell_qposadr = model.jnt_qposadr[barbell_joint_id] if barbell_joint_id >= 0 else -1
    barbell_qveladr = model.jnt_dofadr[barbell_joint_id] if barbell_joint_id >= 0 else -1
    attached = False
    attach_pos_local = np.zeros(3)
    attach_quat_local = np.array([1.0, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    while viewer.is_running():
        step_start = time.perf_counter()

        # Step physics STEPS_PER_FRAME times so simulation keeps pace with
        # wall-clock at the requested TIMESCALE (default = real-time).
        # PD torques are recomputed every physics step (not once per frame)
        # so the controller always sees fresh joint state — prevents shaking.
        for _ in range(STEPS_PER_FRAME):
            # One-shot normalized phase in [0, 1] (no repeat)
            phase = min(max(data.time / SQUAT_PERIOD, 0.0), 1.0)
            
            grip_should_be_active = GRAB_PHASE_ON < phase < GRAB_PHASE_OFF
            wrist_bar_dist = np.inf
            left_dist = np.inf
            right_dist = np.inf

            if (
                left_wrist_body_id >= 0
                and right_wrist_body_id >= 0
                and barbell_body_id >= 0
                and barbell_joint_id >= 0
                and model.jnt_type[barbell_joint_id] == mujoco.mjtJoint.mjJNT_FREE
            ):
                left_wrist_pos = data.xpos[left_wrist_body_id].copy()
                right_wrist_pos = data.xpos[right_wrist_body_id].copy()
                wrist_mid_pos = 0.5 * (left_wrist_pos + right_wrist_pos)
                wrist_quat = normalize_quat(data.xquat[left_wrist_body_id].copy())
                bar_pos = data.xpos[barbell_body_id].copy()
                bar_quat = normalize_quat(data.xquat[barbell_body_id].copy())
                left_grip_pos = bar_pos + quat_rotate(
                    bar_quat, np.array([0.0, BAR_GRIP_OFFSET_Y, 0.0])
                )
                right_grip_pos = bar_pos + quat_rotate(
                    bar_quat, np.array([0.0, -BAR_GRIP_OFFSET_Y, 0.0])
                )
                left_dist = np.linalg.norm(left_grip_pos - left_wrist_pos)
                right_dist = np.linalg.norm(right_grip_pos - right_wrist_pos)
                wrist_bar_dist = 0.5 * (left_dist + right_dist)

                if (
                    grip_should_be_active
                    and left_dist <= GRAB_ATTACH_DIST
                    and right_dist <= GRAB_ATTACH_DIST
                    and not attached
                ):
                    attach_pos_local = quat_rotate(
                        quat_conjugate(wrist_quat), bar_pos - wrist_mid_pos
                    )
                    attach_quat_local = normalize_quat(
                        quat_multiply(quat_conjugate(wrist_quat), bar_quat)
                    )
                    attached = True
                elif not grip_should_be_active:
                    attached = False

                if attached and barbell_qposadr >= 0 and barbell_qveladr >= 0:
                    left_wrist_pos = data.xpos[left_wrist_body_id].copy()
                    right_wrist_pos = data.xpos[right_wrist_body_id].copy()
                    wrist_mid_pos = 0.5 * (left_wrist_pos + right_wrist_pos)
                    wrist_quat = normalize_quat(data.xquat[left_wrist_body_id].copy())

                    target_pos = (
                        wrist_mid_pos
                        + quat_rotate(wrist_quat, attach_pos_local)
                        + quat_rotate(wrist_quat, BAR_ATTACH_NUDGE_LOCAL)
                    )
                    target_quat = normalize_quat(
                        quat_multiply(wrist_quat, attach_quat_local)
                    )

                    data.qpos[barbell_qposadr:barbell_qposadr + 3] = target_pos
                    data.qpos[barbell_qposadr + 3:barbell_qposadr + 7] = target_quat
                    data.qvel[barbell_qveladr:barbell_qveladr + 6] = 0.0
                    mujoco.mj_forward(model, data)

            # Squat cycle: smoothly blend stand ↔ squat based on sim time
            q_des = squat_cycle(data.time, q_stand, q_squat, q_grab, q_sink, q_carry)

            # Pre-grab reach behavior: move hands toward the bar first, then
            # allow grab only when close enough.
            if (
                left_wrist_body_id >= 0
                and right_wrist_body_id >= 0
                and barbell_body_id >= 0
                and PRE_GRAB_PHASE_ON < phase < GRAB_PHASE_OFF
                and not attached
            ):
                dist_alpha = proximity_blend(
                    wrist_bar_dist, APPROACH_DIST_START, APPROACH_DIST_END
                )
                phase_alpha = 1.0 - proximity_blend(
                    phase, GRAB_PHASE_ON, PRE_GRAB_PHASE_ON
                )
                reach_alpha = max(dist_alpha, phase_alpha)
                if phase >= GRAB_PHASE_ON:
                    reach_alpha = max(reach_alpha, 0.95)
                if reach_alpha > 0.0:
                    q_des = (1.0 - reach_alpha) * q_des + reach_alpha * q_reach

            # PD: read current state and compute torques toward q_des
            q  = data.qpos[qpos_idx]
            dq = data.qvel[qvel_idx]
            tau = pd_control(q_des, q, dq_des, dq, gains)

            # Clip to actuator limits defined in the XML
            tau = np.clip(tau, model.actuator_ctrlrange[:, 0],
                               model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

        # Sync viewer at ~50 fps
        elapsed = time.perf_counter() - step_start
        sleep_time = VIEWER_DT - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        viewer.sync()

    viewer.close()
    print("Simulation ended.")


if __name__ == "__main__":
    main()