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
from enum import Enum

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
    "arm":   {"kp": 350.0, "kd": 14.0},
    "wrist": {"kp": 120.0, "kd":  5.0},
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
# Bench press motion definitions
# ---------------------------------------------------------------------------
class BenchPhase(Enum):
    ALIGN = "align"
    REACH = "reach"
    GRASP_CLOSE = "grasp_close"
    SYMMETRY_CAL = "symmetry_cal"
    LOCKIN = "lockin"
    LOAD_TRANSFER = "load_transfer"
    DESCENT = "descent"
    BOTTOM_PAUSE = "bottom_pause"
    ASCENT = "ascent"
    LOCKOUT = "lockout"


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


BENCH_LOCKOUT_POSE = {
    "left_hip_pitch_joint":   -0.20,
    "left_knee_joint":         0.48,
    "left_ankle_pitch_joint": -0.12,
    "right_hip_pitch_joint":   -0.20,
    "right_knee_joint":         0.48,
    "right_ankle_pitch_joint": -0.12,
    "waist_pitch_joint":       -0.10,
    "left_shoulder_pitch_joint":  -1.72,
    "right_shoulder_pitch_joint": -1.72,
    "left_shoulder_roll_joint":    0.55,
    "right_shoulder_roll_joint":  -0.55,
    "left_shoulder_yaw_joint":    -0.18,
    "right_shoulder_yaw_joint":    0.18,
    "left_elbow_joint":            0.05,
    "right_elbow_joint":           0.05,
    "left_wrist_roll_joint":       0.0,
    "right_wrist_roll_joint":      0.0,
    "left_wrist_pitch_joint":      0.0,
    "right_wrist_pitch_joint":     0.0,
    "left_wrist_yaw_joint":        0.0,
    "right_wrist_yaw_joint":       0.0,
}

# Bottom: bar at chest with deep elbow flexion.
BENCH_BOTTOM_POSE = {
    "left_hip_pitch_joint":   -0.20,
    "left_knee_joint":         0.48,
    "left_ankle_pitch_joint": -0.12,
    "right_hip_pitch_joint":   -0.20,
    "right_knee_joint":         0.48,
    "right_ankle_pitch_joint": -0.12,
    "waist_pitch_joint":       -0.10,
    "left_shoulder_pitch_joint":  -1.72,
    "right_shoulder_pitch_joint": -1.72,
    "left_shoulder_roll_joint":    0.55,
    "right_shoulder_roll_joint":  -0.55,
    "left_shoulder_yaw_joint":    -0.18,
    "right_shoulder_yaw_joint":    0.18,
    "left_elbow_joint":            1.50,
    "right_elbow_joint":           1.50,
    "left_wrist_roll_joint":       0.0,
    "right_wrist_roll_joint":      0.0,
    "left_wrist_pitch_joint":      0.0,
    "right_wrist_pitch_joint":     0.0,
    "left_wrist_yaw_joint":        0.0,
    "right_wrist_yaw_joint":       0.0,
}

# Setup and unrack match lockout for clean acquisition.
BENCH_SETUP_POSE = BENCH_LOCKOUT_POSE.copy()
BENCH_UNRACK_POSE = BENCH_LOCKOUT_POSE.copy()

DEFAULT_POSE = BENCH_SETUP_POSE

# Timing (seconds)
ALIGN_TIME = 0.8
REACH_TIME = 1.0
GRASP_CLOSE_TIME = 0.9
SYMMETRY_CAL_TIME = 0.5
LOCKIN_TIME = 0.6
LOAD_TRANSFER_TIME = 0.6
DESCENT_TIME = 2.5
BOTTOM_PAUSE_TIME = 0.8
ASCENT_TIME = 1.5
LOCKOUT_HOLD_TIME = 2.5

# Barbell visual dimensions - loaded with 20 kg plates.
BAR_TOTAL_LENGTH = 1.20
BAR_SHAFT_RADIUS = 0.015
BAR_PLATE_RADIUS = 0.12
BAR_PLATE_THICKNESS = 0.028
COLLAR_RADIUS = 0.030
COLLAR_THICKNESS = 0.018
HAND_SPREAD_BIAS = 0.10

GRIP_OPEN = {
    "left_hand_thumb_0_joint": -0.20,
    "left_hand_thumb_1_joint": -0.20,
    "left_hand_thumb_2_joint": 0.05,
    "left_hand_middle_0_joint": -0.20,
    "left_hand_middle_1_joint": 0.05,
    "left_hand_index_0_joint": -0.20,
    "left_hand_index_1_joint": 0.05,
    "right_hand_thumb_0_joint": -0.20,
    "right_hand_thumb_1_joint": -0.20,
    "right_hand_thumb_2_joint": 0.05,
    "right_hand_middle_0_joint": -0.20,
    "right_hand_middle_1_joint": 0.05,
    "right_hand_index_0_joint": -0.20,
    "right_hand_index_1_joint": 0.05,
}

GRIP_CLOSED = {
    "left_hand_thumb_0_joint": 0.35,
    "left_hand_thumb_1_joint": 0.60,
    "left_hand_thumb_2_joint": 1.00,
    "left_hand_middle_0_joint": -1.05,
    "left_hand_middle_1_joint": 1.35,
    "left_hand_index_0_joint": -1.00,
    "left_hand_index_1_joint": 1.30,
    "right_hand_thumb_0_joint": 0.35,
    "right_hand_thumb_1_joint": 0.60,
    "right_hand_thumb_2_joint": 1.00,
    "right_hand_middle_0_joint": -1.05,
    "right_hand_middle_1_joint": 1.35,
    "right_hand_index_0_joint": -1.00,
    "right_hand_index_1_joint": 1.30,
}


def blend_pose(q_a: np.ndarray, q_b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * q_a + alpha * q_b


class BenchPressController:
    def __init__(self, model: mujoco.MjModel):
        self.phase = BenchPhase.LOCKOUT
        self.phase_time = 0.0
        self.prev_bar_center = None

        self.setup = build_desired_pose(model, BENCH_SETUP_POSE)
        self.unrack = build_desired_pose(model, BENCH_UNRACK_POSE)
        self.bottom = build_desired_pose(model, BENCH_BOTTOM_POSE)
        self.lockout = build_desired_pose(model, BENCH_LOCKOUT_POSE)

        self.joint_to_act = {}
        for act_id in range(model.nu):
            joint_id = model.actuator_trnid[act_id, 0]
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if jname:
                self.joint_to_act[jname] = act_id

        self.left_wrist_body = self._find_body_exact(model, "left_wrist_yaw_link")
        self.right_wrist_body = self._find_body_exact(model, "right_wrist_yaw_link")
        if self.left_wrist_body < 0:
            self.left_wrist_body = self._find_body(model, ["left", "wrist"])
        if self.right_wrist_body < 0:
            self.right_wrist_body = self._find_body(model, ["right", "wrist"])
        self.torso_body = self._find_body(model, ["torso"])
        # Local offset from wrist_yaw frame toward true grip line on palm.
        self.grip_offset_local = np.array([0.105, 0.0015, -0.008], dtype=np.float64)

    def _find_body_exact(self, model: mujoco.MjModel, exact_name: str) -> int:
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name == exact_name:
                return body_id
        return -1

    def _find_body(self, model: mujoco.MjModel, tokens: list) -> int:
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name:
                continue
            n = name.lower()
            if all(tok in n for tok in tokens):
                return body_id
        return -1

    def _joint_q(self, model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
        act_id = self.joint_to_act.get(joint_name)
        if act_id is None:
            return 0.0
        joint_id = model.actuator_trnid[act_id, 0]
        qpos_adr = model.jnt_qposadr[joint_id]
        return float(data.qpos[qpos_adr])

    def _grip_points(self, data: mujoco.MjData):
        if self.left_wrist_body < 0 or self.right_wrist_body < 0:
            return None, None

        l_rot = data.xmat[self.left_wrist_body].reshape(3, 3)
        r_rot = data.xmat[self.right_wrist_body].reshape(3, 3)
        l_pos = data.xpos[self.left_wrist_body] + l_rot @ self.grip_offset_local
        r_pos = data.xpos[self.right_wrist_body] + r_rot @ self.grip_offset_local
        return l_pos, r_pos

    def _phase_target(self) -> np.ndarray:
        if self.phase == BenchPhase.LOCKOUT:
            return self.lockout.copy()
        if self.phase == BenchPhase.DESCENT:
            a = smoothstep(self.phase_time / DESCENT_TIME)
            return blend_pose(self.lockout, self.bottom, a)
        if self.phase == BenchPhase.BOTTOM_PAUSE:
            return self.bottom.copy()
        if self.phase == BenchPhase.ASCENT:
            a = smoothstep(self.phase_time / ASCENT_TIME)
            return blend_pose(self.bottom, self.lockout, a)
        return self.lockout.copy()

    def _advance_phase(self, dt: float):
        self.phase_time += dt
        if self.phase == BenchPhase.LOCKOUT and self.phase_time >= LOCKOUT_HOLD_TIME:
            self.phase = BenchPhase.DESCENT
            self.phase_time = 0.0
        elif self.phase == BenchPhase.DESCENT and self.phase_time >= DESCENT_TIME:
            self.phase = BenchPhase.BOTTOM_PAUSE
            self.phase_time = 0.0
        elif self.phase == BenchPhase.BOTTOM_PAUSE and self.phase_time >= BOTTOM_PAUSE_TIME:
            self.phase = BenchPhase.ASCENT
            self.phase_time = 0.0
        elif self.phase == BenchPhase.ASCENT and self.phase_time >= ASCENT_TIME:
            self.phase = BenchPhase.LOCKOUT
            self.phase_time = 0.0
        elif self.phase == BenchPhase.LOCKOUT and self.phase_time >= LOCKOUT_HOLD_TIME:
            self.phase = BenchPhase.DESCENT
            self.phase_time = 0.0

    def target(self, model: mujoco.MjModel, data: mujoco.MjData, dt: float) -> np.ndarray:
        q_des = self._phase_target()

        # Lock lower body and torso.
        for jn in (
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_knee_joint", "right_knee_joint",
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            "left_ankle_roll_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        ):
            aid = self.joint_to_act.get(jn)
            if aid is not None:
                q_des[aid] = self.setup[aid]

        # Shoulder pitch follows a short, chest-centric arc.
        for jn in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"):
            aid = self.joint_to_act.get(jn)
            if aid is not None:
                q_des[aid] = np.clip(q_des[aid], -1.80, -1.60)

        # Shoulder roll opens/closes through the rep to lower/raise the bar.
        for jn, lo, hi in (
            ("left_shoulder_roll_joint", 0.50, 1.00),
            ("right_shoulder_roll_joint", -1.00, -0.50),
        ):
            aid = self.joint_to_act.get(jn)
            if aid is not None:
                q_des[aid] = np.clip(q_des[aid], lo, hi)

        # Keep shoulder yaw fixed to avoid forearm twist and drift.
        yaw_lock = {
            "left_shoulder_yaw_joint": -0.18,
            "right_shoulder_yaw_joint": 0.18,
        }
        for jn, val in yaw_lock.items():
            aid = self.joint_to_act.get(jn)
            if aid is not None:
                q_des[aid] = val

        # Mirror right side from left for bilateral symmetry.
        mirror_pairs = (
            ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", 1.0),
            ("left_shoulder_roll_joint", "right_shoulder_roll_joint", -1.0),
            ("left_elbow_joint", "right_elbow_joint", 1.0),
        )
        for l_name, r_name, sign in mirror_pairs:
            l_id = self.joint_to_act.get(l_name)
            r_id = self.joint_to_act.get(r_name)
            if l_id is not None and r_id is not None:
                q_des[r_id] = sign * q_des[l_id]

        # Wrists neutral.
        for jn in (
            "left_wrist_roll_joint", "right_wrist_roll_joint",
            "left_wrist_pitch_joint", "right_wrist_pitch_joint",
            "left_wrist_yaw_joint", "right_wrist_yaw_joint",
        ):
            aid = self.joint_to_act.get(jn)
            if aid is not None:
                q_des[aid] = 0.0

        # Elbow safety clamp.
        for jn in ("left_elbow_joint", "right_elbow_joint"):
            aid = self.joint_to_act.get(jn)
            if aid is not None:
                q_des[aid] = np.clip(q_des[aid], 0.0, 1.70)

        # Re-enforce mirrored elbow after clamping.
        l_id = self.joint_to_act.get("left_elbow_joint")
        r_id = self.joint_to_act.get("right_elbow_joint")
        if l_id is not None and r_id is not None:
            q_des[r_id] = q_des[l_id]

        # Grip always closed (setup/grasp phases are skipped).

        for jn, open_val in GRIP_OPEN.items():
            aid = self.joint_to_act.get(jn)
            if aid is None:
                continue
            q_des[aid] = GRIP_CLOSED.get(jn, open_val)

        self._advance_phase(dt)
        return q_des


def draw_barbell(viewer, data: mujoco.MjData, left_body: int, right_body: int):
    if left_body < 0 or right_body < 0:
        return

    grip_offset_local = np.array([0.105, 0.0015, -0.008], dtype=np.float64)
    left_rot = data.xmat[left_body].reshape(3, 3)
    right_rot = data.xmat[right_body].reshape(3, 3)
    left_p = data.xpos[left_body].copy() + left_rot @ grip_offset_local
    right_p = data.xpos[right_body].copy() + right_rot @ grip_offset_local

    center = 0.5 * (left_p + right_p)
    lr = right_p - left_p
    lr[2] = 0.0
    lr_norm = np.linalg.norm(lr)
    axis = lr / lr_norm if lr_norm > 1e-6 else np.array([0.0, 1.0, 0.0])

    half_len = 0.5 * BAR_TOTAL_LENGTH
    left_end = center - axis * half_len
    right_end = center + axis * half_len
    left_end[2] = center[2]
    right_end[2] = center[2]

    scn = viewer.user_scn
    scn.ngeom = 0

    def add_geom() -> int:
        idx = scn.ngeom
        scn.ngeom += 1
        return idx

    # Bar shaft (silver).
    g = add_geom()
    mujoco.mjv_initGeom(
        scn.geoms[g],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.array([0.82, 0.82, 0.85, 1.0]),
    )
    mujoco.mjv_connector(
        scn.geoms[g],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        BAR_SHAFT_RADIUS,
        left_end.astype(np.float64),
        right_end.astype(np.float64),
    )

    def draw_plate(center_pos: np.ndarray, radius: float, thickness: float, color: list):
        p0 = center_pos - axis * (thickness * 0.5)
        p1 = center_pos + axis * (thickness * 0.5)
        g = add_geom()
        mujoco.mjv_initGeom(
            scn.geoms[g],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).flatten(),
            np.array([*color, 1.0]),
        )
        mujoco.mjv_connector(
            scn.geoms[g],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            p0.astype(np.float64),
            p1.astype(np.float64),
        )

    # 20 kg plates.
    plate_inset = 0.05
    for end, direction in ((left_end, +1), (right_end, -1)):
        plate_center = end + axis * direction * plate_inset
        draw_plate(plate_center, BAR_PLATE_RADIUS, BAR_PLATE_THICKNESS, [0.18, 0.18, 0.18])

    # Collars.
    collar_inset = plate_inset + BAR_PLATE_THICKNESS + 0.012
    for end, direction in ((left_end, +1), (right_end, -1)):
        collar_center = end + axis * direction * collar_inset
        draw_plate(collar_center, COLLAR_RADIUS, COLLAR_THICKNESS, [0.75, 0.75, 0.78])


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
    # Initialise qpos to a lying bench-press start pose on the static bench.
    # ------------------------------------------------------------------
    data.qpos[qpos_idx] = q_des          # set joint angles
    data.qpos[0] = -0.15                 # align torso centered over bench
    data.qpos[1] = 0.0
    data.qpos[2] = 0.74                  # back resting on bench surface
    # Rotate base to lie on back (supine): ~-90 deg about y-axis.
    data.qpos[3] = 0.70710678            # quaternion w
    data.qpos[4] = 0.0                   # quaternion x
    data.qpos[5] = -0.70710678           # quaternion y
    data.qpos[6] = 0.0                   # quaternion z
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

    # Bench press phase controller
    bench = BenchPressController(model)
    last_phase = None

    print("\nSimulation running — robot starts supine on the static bench.")
    print("Only the barbell is rendered dynamically; bench/rack come from scene XML.")
    print("Close the viewer window to stop.\n")

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
            # Bench press state machine target with live feedback corrections.
            q_des = bench.target(model, data, model.opt.timestep)

            # PD: read current state and compute torques toward q_des
            q  = data.qpos[qpos_idx]
            dq = data.qvel[qvel_idx]
            tau = pd_control(q_des, q, dq_des, dq, gains)

            # Clip to actuator limits defined in the XML
            tau = np.clip(tau, model.actuator_ctrlrange[:, 0],
                               model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            # Pin floating base to bench pose to prevent torso lift-off.
            data.qpos[0] = -0.15
            data.qpos[1] = 0.0
            data.qpos[2] = 0.74
            data.qpos[3] = 0.70710678
            data.qpos[4] = 0.0
            data.qpos[5] = -0.70710678
            data.qpos[6] = 0.0
            data.qvel[0:6] = 0.0

        if bench.phase != last_phase:
            print(f"Phase: {bench.phase.value}")
            last_phase = bench.phase

        # Spawn/update visual barbell.
        draw_barbell(viewer, data, bench.left_wrist_body, bench.right_wrist_body)

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