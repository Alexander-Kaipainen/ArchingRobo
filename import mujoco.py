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
# Motor gain profiles (from ark_unitree_g1 UnitreeSdk2Bridge)
# ---------------------------------------------------------------------------
GAINS = {
    "high_torque": {"kp": 300.0, "kd": 3.0},
    "low_torque":  {"kp":  80.0, "kd": 3.0},
    "wrist":       {"kp":  40.0, "kd": 1.5},
}


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

    print("\nSimulation running — close the viewer window to stop.\n")
    print("Tip: For full ARK + Unitree SDK control, run:")
    print("  cd ark_unitree_g1/tests/g1_mujoco_sim")
    print("  python unitree_mujoco.py\n")

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    num_motors = model.nu

    while viewer.is_running():
        step_start = time.perf_counter()

        # --- Control: zero torques (robot stands under gravity) ---
        # Replace this section with your own controller logic.
        # Example: apply small position targets to keep the robot standing
        data.ctrl[:] = np.zeros(num_motors)

        # Step physics
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