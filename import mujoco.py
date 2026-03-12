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

# Animation mode: enables bench press sequence
ENABLE_ANIMATION = "--animate" in sys.argv or not HEADLESS

# Simulation parameters
SIMULATE_DT = 0.002    # 500 Hz physics step (matching ARK default)
VIEWER_DT   = 0.02     # 50 Hz viewer refresh

# Camera defaults
CAM_DISTANCE  = 4.0
CAM_ELEVATION = -25
CAM_AZIMUTH   = 135
CAM_LOOKAT    = np.array([0.5, 0.0, 0.6])

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


class BenchPressAnimation:
    """
    Bench press animation — robot starts ALREADY LYING on the bench.

    Actuator indices (ctrl[i]):
      0-5   left leg    6-11  right leg    12-14 waist
      15    L shoulder pitch   16 L shoulder roll   17 L shoulder yaw   18 L elbow
      19-21 L wrist roll/pitch/yaw    22-28 L hand fingers
      29    R shoulder pitch   30 R shoulder roll   31 R shoulder yaw   32 R elbow
      33-35 R wrist roll/pitch/yaw    36-42 R hand fingers

    Phases:
       0-2s   Settle on bench
       2-4s   Reach arms up toward barbell
       4-6s   Lower bar to chest
       6-8s   Press bar up
       8-10s  Lower bar again
      10-12s  Press and hold lockout
    """

    def __init__(self, duration=12.0):
        self.duration = duration
        self.elapsed = 0.0
        self.active = True

    def _leg_targets(self):
        """Constant leg targets for lying pose."""
        return {
            0:  1.0,   6:  1.0,    # hip pitch
            3:  0.8,   9:  0.8,    # knee
            4: -0.4,  10: -0.4,    # ankle pitch
        }

    def get_target_qpos(self, model, t):
        n = model.nu
        q = np.zeros(n)

        # Legs always hold the lying pose
        for idx, val in self._leg_targets().items():
            if idx < n:
                q[idx] = val

        # Waist: keep torso straight at all times
        q[12] = 0.0;  q[13] = 0.0;  q[14] = 0.0

        if t < 2.0:
            # Settle — arms at sides
            q[15] = -0.3;  q[16] =  0.3;  q[18] = 0.4   # left arm
            q[29] = -0.3;  q[30] = -0.3;  q[32] = 0.4   # right arm

        elif t < 4.0:
            # Reach up toward barbell
            a = (t - 2.0) / 2.0
            q[15] = -0.3 + 1.8 * a;   q[29] = -0.3 + 1.8 * a
            q[16] =  0.3 * (1 - a);   q[30] = -0.3 * (1 - a)
            q[18] =  0.4 * (1 - a);   q[32] =  0.4 * (1 - a)

        elif t < 6.0:
            # Lower bar to chest (bend elbows, lower shoulders)
            a = (t - 4.0) / 2.0
            q[15] = 1.5 - 1.0 * a;    q[29] = 1.5 - 1.0 * a
            q[18] = 1.2 * a;          q[32] = 1.2 * a
            q[16] = 0.2 * a;          q[30] = -0.2 * a

        elif t < 8.0:
            # PRESS UP
            a = (t - 6.0) / 2.0
            q[15] = 0.5 + 1.0 * a;    q[29] = 0.5 + 1.0 * a
            q[18] = 1.2 * (1 - a);    q[32] = 1.2 * (1 - a)
            q[16] = 0.2 * (1 - a);    q[30] = -0.2 * (1 - a)

        elif t < 10.0:
            # Lower again
            a = (t - 8.0) / 2.0
            q[15] = 1.5 - 1.0 * a;    q[29] = 1.5 - 1.0 * a
            q[18] = 1.2 * a;          q[32] = 1.2 * a
            q[16] = 0.2 * a;          q[30] = -0.2 * a

        else:
            # Press and hold lockout
            q[15] = 1.5;  q[29] = 1.5
            q[18] = 0.0;  q[32] = 0.0

        self.elapsed = t
        self.active = t < self.duration
        return q

    def is_active(self):
        return self.active


def bench_press_controller(model: mujoco.MjModel, data: mujoco.MjData,
                           animation: BenchPressAnimation, t: float):
    """
    PD controller tracking bench-press animation keyframes.
    Also manages the barbell grip constraint (weld).
    """
    # --- Activate / deactivate barbell grip (both hands) ---
    # At t≈3.5s (arms reaching bar), activate the grip welds
    grip_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grip_bar_l")
    grip_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grip_bar_r")
    for grip_id in [grip_l_id, grip_r_id]:
        if grip_id >= 0:
            if t >= 3.5:
                data.eq_active[grip_id] = 1  # grip ON
            else:
                data.eq_active[grip_id] = 0  # grip OFF

    # --- Get target joint angles ---
    if animation and animation.is_active():
        q_target = animation.get_target_qpos(model, t)
    else:
        q_target = animation.get_target_qpos(model, animation.duration) if animation else np.zeros(model.nu)

    q_current = data.qpos[7:]
    v_current = data.qvel[6:]

    for i in range(model.nu):
        q_err = q_target[i] - q_current[i]
        v_err = -v_current[i]
        if i < 12:      # legs
            kp, kd = 300.0, 8.0
        elif i < 15:    # waist
            kp, kd = 250.0, 6.0
        else:           # arms + hands
            kp, kd = 150.0, 4.0
        data.ctrl[i] = kp * q_err + kd * v_err

    data.ctrl[:] = np.clip(data.ctrl, -200, 200)


def standing_controller(model: mujoco.MjModel, data: mujoco.MjData):
    """
    Simple PD controller to keep the robot standing upright.
    
    The default standing posture has joints near zero (extended/neutral).
    We apply P control toward these targets with reasonable gains.
    """
    # Joint gains (simplified: use same for most joints)
    kp = 200.0  # Position gain
    kd = 5.0    # Damping gain
    
    # Target positions (mostly zeros for standing — extended legs, neutral torso)
    q_target = np.zeros(model.nq - 1)  # Exclude floating base (first 7 DOF)
    
    # Read current joint positions and velocities (skip floating base)
    q_current = data.qpos[7:]  # Skip [x, y, z, qw, qx, qy, qz]
    v_current = data.qvel[6:]  # Skip floating base linear/angular vel
    
    # Compute PD torques for each actuator
    for i in range(model.nu):
        # Position error
        q_err = q_target[i] - q_current[i]
        
        # Velocity (setpoint is zero)
        v_err = -v_current[i]
        
        # PD law: tau = kp * q_err + kd * v_err
        data.ctrl[i] = kp * q_err + kd * v_err
    
    # Clip torques to reasonable limits to avoid instability
    data.ctrl[:] = np.clip(data.ctrl, -100, 100)


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


def setup_initial_pose(model, data):
    """
    Place the robot lying on the bench and the barbell on the J-hooks.
    Floating base qpos: [x, y, z, qw, qx, qy, qz]
    """
    # --- Robot: lie on the bench, face up ---
    data.qpos[0] = 0.65   # x: shift toward head-end so torso is on the pad
    data.qpos[1] = 0.0    # y: centered
    data.qpos[2] = 0.55   # z: pelvis above pad (pad top ≈ 0.455 in world)
    # Quaternion: lying on back = rotate -90 deg around Y
    data.qpos[3] = 0.7071   # qw
    data.qpos[4] = 0.0      # qx
    data.qpos[5] = -0.7071  # qy
    data.qpos[6] = 0.0      # qz

    # Use named joint lookup for reliable qpos indexing
    def set_joint(name, value):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            adr = model.jnt_qposadr[jid]
            data.qpos[adr] = value

    # Legs: hips bent so feet reach ground beside bench
    set_joint("left_hip_pitch_joint",   1.0)
    set_joint("right_hip_pitch_joint",  1.0)
    set_joint("left_knee_joint",        0.8)
    set_joint("right_knee_joint",       0.8)
    set_joint("left_ankle_pitch_joint", -0.4)
    set_joint("right_ankle_pitch_joint",-0.4)
    # Arms: at sides, elbows slightly bent
    set_joint("left_shoulder_pitch_joint",  -0.3)
    set_joint("left_shoulder_roll_joint",    0.3)
    set_joint("left_elbow_joint",            0.4)
    set_joint("right_shoulder_pitch_joint", -0.3)
    set_joint("right_shoulder_roll_joint",  -0.3)   # mirror
    set_joint("right_elbow_joint",           0.4)

    # --- Barbell: rest on J-hooks ---
    bb_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "barbell_free")
    bb_qadr = model.jnt_qposadr[bb_jnt_id]
    data.qpos[bb_qadr + 0] = 0.35   # x
    data.qpos[bb_qadr + 1] = 0.0    # y
    data.qpos[bb_qadr + 2] = 0.979  # z (hook top 0.965 + bar radius 0.014)
    data.qpos[bb_qadr + 3] = 1.0    # qw
    data.qpos[bb_qadr + 4] = 0.0    # qx
    data.qpos[bb_qadr + 5] = 0.0    # qy
    data.qpos[bb_qadr + 6] = 0.0    # qz

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    print("  Initial pose: robot lying on bench, barbell on J-hooks")
    print(f"    Pelvis: {data.body('pelvis').xpos}")
    print(f"    Barbell: {data.body('barbell').xpos}")


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

    # Set up initial pose: robot on bench, barbell on hooks
    setup_initial_pose(model, data)

    # Initialize animation if enabled
    animation = None
    if ENABLE_ANIMATION:
        animation = BenchPressAnimation(duration=12.0)
        print("  Animation: ENABLED (bench press sequence)")
    else:
        print("  Animation: disabled")

    # Print scene info
    print_scene_info(model)
    print()

    if HEADLESS:
        print("Headless mode — skipping viewer. Model loaded successfully!")
        return

    # ------------------------------------------------------------------
    # Attempt to launch passive viewer (with error handling)
    # ------------------------------------------------------------------
    print("Attempting to launch MuJoCo viewer...")
    viewer = None
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
        print("✓ Viewer launched successfully")
    except Exception as e:
        print(f"✗ Viewer launch failed: {e}")
        print("\nFalling back to non-visual simulation loop...")
        print("  (Simulation will run in the background)")
        print("  Run with --no-viewer to skip this message\n")

    if viewer is None:
        # ------------------------------------------------------------------
        # Fallback: Run simulation without viewer (useful for CI/headless)
        # ------------------------------------------------------------------
        num_motors = model.nu
        sim_duration = 10.0  # Run for 10 seconds
        steps_per_sec = int(1.0 / SIMULATE_DT)
        target_steps = int(sim_duration * steps_per_sec)

        print(f"Running {target_steps} simulation steps (~{sim_duration}s)...")
        for step in range(target_steps):
            t = data.time
            
            if animation and ENABLE_ANIMATION:
                bench_press_controller(model, data, animation, t)
            else:
                standing_controller(model, data)
            
            # Step physics
            mujoco.mj_step(model, data)

            # Status update every 1000 steps
            if (step + 1) % steps_per_sec == 0:
                elapsed_sim = data.time
                print(f"  [{step+1:5d}/{target_steps}] sim_time={elapsed_sim:.2f}s, "
                      f"robot_z={data.body('pelvis').xpos[2]:.3f}m")

        print(f"\n✓ Simulation complete. Final sim time: {data.time:.2f}s")
        print("Tip: For viewer support, ensure:")
        print("  1. You have a display (X11, Wayland, or Windows GUI)")
        print("  2. Graphics drivers are installed")
        print("  3. Try: pip install --upgrade mujoco\n")
        return

    # ------------------------------------------------------------------
    # Simulation loop with viewer
    # ------------------------------------------------------------------
    # Configure camera
    viewer.cam.distance  = CAM_DISTANCE
    viewer.cam.elevation = CAM_ELEVATION
    viewer.cam.azimuth   = CAM_AZIMUTH
    viewer.cam.lookat[:] = CAM_LOOKAT

    print("\nSimulation running — close the viewer window to stop.\n")
    if ENABLE_ANIMATION:
        print("Bench Press Animation:")
        print("   0-2s   Settle on bench")
        print("   2-4s   Reach up to barbell")
        print("   4-6s   Lower bar to chest")
        print("   6-8s   Press up!")
        print("   8-10s  Lower again")
        print("  10-12s  Press and hold lockout\n")

    num_motors = model.nu

    # Number of physics sub-steps per viewer frame so sim runs in real time.
    # E.g. VIEWER_DT=0.02, SIMULATE_DT=0.002 → 10 physics steps per render.
    n_substeps = max(1, int(round(VIEWER_DT / SIMULATE_DT)))
    print(f"Physics sub-steps per frame: {n_substeps}")

    while viewer.is_running():
        step_start = time.perf_counter()

        # Run multiple physics steps per render frame
        for _ in range(n_substeps):
            # Apply controller
            t = data.time
            if animation and ENABLE_ANIMATION:
                bench_press_controller(model, data, animation, t)
            else:
                standing_controller(model, data)

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