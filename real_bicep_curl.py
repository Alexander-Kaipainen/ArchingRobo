import numpy as np
import time
import sys
import os
import argparse

# Ensure we can find the unitree_g1 package
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from unitree_g1.robot_arm import G1_29_ArmController, G1_29_JointIndex, G1_29_JointArmIndex

def main():
    parser = argparse.ArgumentParser(description="Unitree G1 Professional Bicep Curl Controller")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode (uses 'lo' interface and Domain 1)")
    parser.add_argument("--interface", type=str, default="enp129s0", help="Network interface for real robot (default: enp129s0)")
    parser.add_argument("--domain", type=int, default=None, help="DDS Domain ID (defaults to 1 for sim, 0 for real)")
    args = parser.parse_args()

    # Determine connection parameters
    if args.sim:
        interface = "lo"
        domain_id = args.domain if args.domain is not None else 1
        print(">>> RUNNING IN SIMULATION MODE (lo, Domain 1) <<<")
    else:
        interface = args.interface
        domain_id = args.domain if args.domain is not None else 0
        print(f">>> RUNNING ON REAL ROBOT ({interface}, Domain {domain_id}) <<<")
    
    try:
        # Initialize the controller
        controller = G1_29_ArmController(network_interface=interface, domain_id=domain_id)
        controller.speed_instant_max()
    except TimeoutError as e:
        if args.sim:
            print(f"ERROR: {e}")
            print("SIM MODE REQUIRES THE MUJOCO BRIDGE TO BE RUNNING FIRST:")
            print("  python3.12 tests/g1_mujoco_sim/unitree_mujoco.py")
        else:
            print(f"ERROR: Failed to initialize controller: {e}")
        return
    except Exception as e:
        print(f"ERROR: Failed to initialize controller: {e}")
        return

    # --- HARDWARE TUNING ---
    main_arm_joints = [
        G1_29_JointIndex.kLeftShoulderPitch, G1_29_JointIndex.kLeftShoulderRoll,
        G1_29_JointIndex.kLeftShoulderYaw, G1_29_JointIndex.kLeftElbow,
        G1_29_JointIndex.kRightShoulderPitch, G1_29_JointIndex.kRightShoulderRoll,
        G1_29_JointIndex.kRightShoulderYaw, G1_29_JointIndex.kRightElbow
    ]
    
    # Tuned for professional responsiveness
    TUNED_KP = 145.0
    TUNED_KD = 6.5
    
    for j_idx in main_arm_joints:
        controller.msg.motor_cmd[j_idx].kp = TUNED_KP
        controller.msg.motor_cmd[j_idx].kd = TUNED_KD
    controller.ctrl_dual_arm(controller.get_current_dual_arm_q(), np.zeros(14))

    print(f"\nDirect Control Tuned: KP={TUNED_KP}, KD={TUNED_KD}")
    
    # 1. MOVE TO PREP (SYNCHRONIZED)
    prep_q = np.zeros(14)
    prep_q[0] = -0.4   # Left Shoulder Pitch Forward
    prep_q[1] = 0.5    # Left Shoulder Roll Out
    prep_q[7] = -0.4   # Right Shoulder Pitch Forward
    prep_q[8] = -0.5   # Right Shoulder Roll Out
    
    start_q = controller.get_current_dual_arm_q()

    print("\nMoving to safe prep posture...")
    duration = 3.0
    steps = int(duration * 250)
    for i in range(steps):
        t0 = time.perf_counter()
        alpha = (i + 1) / steps
        cmd_q = (1 - alpha) * start_q + alpha * prep_q
        controller.ctrl_dual_arm(cmd_q, np.zeros(14))
        dt = time.perf_counter() - t0
        if dt < 0.004:
            time.sleep(0.004 - dt)

    # 2. CONTINUOUS BICEP CURL (FILTERED + RAMPED)
    CURL_PERIOD = 5.5
    ELBOW_EXTENDED = 0.0
    ELBOW_FLEXED = 1.4
    RAMP_UP_TIME = 2.5  # Seconds to reach full amplitude
    
    # LPF Parameters (20Hz Cutoff at 250Hz sampling)
    ALPHA = 0.45 # Smoothing factor
    filtered_q = prep_q.copy()
    filtered_dq = np.zeros(14)

    print("\nStarting Professional Bicep Curls.")
    print(">>> REMINDER: Press L2+R2 on remote to enter Debug Mode if you haven't yet! <<<")
    print(">>> SAFETY: Press Ctrl+C to return to home and stop. <<<")
    
    try:
        start_time = time.time()
        while True:
            t0 = time.perf_counter()
            elapsed = time.time() - start_time
            
            # Amplitude Ramping
            ramp = min(1.0, elapsed / RAMP_UP_TIME)
            
            # Current Target (Raw)
            omega = (2.0 * np.pi) / CURL_PERIOD
            phase = omega * elapsed
            amp = ((ELBOW_FLEXED - ELBOW_EXTENDED) / 2.0) * ramp
            mid_val = (ELBOW_FLEXED + ELBOW_EXTENDED) / 2.0
            
            target_q_raw = prep_q.copy()
            target_dq_raw = np.zeros(14)
            
            # Elbow sine wave
            q_sine = mid_val - amp * np.cos(phase)
            dq_sine = amp * omega * np.sin(phase)
            
            target_q_raw[3] = q_sine     # Left Elbow
            target_dq_raw[3] = dq_sine
            target_q_raw[10] = q_sine    # Right Elbow
            target_dq_raw[10] = dq_sine
            
            # Update Filtered Values (LPF)
            filtered_q = ALPHA * target_q_raw + (1.0 - ALPHA) * filtered_q
            filtered_dq = ALPHA * target_dq_raw + (1.0 - ALPHA) * filtered_dq
            
            controller.ctrl_dual_arm(filtered_q, np.zeros(14))
            
            # Precise 250Hz sync
            dt = time.perf_counter() - t0
            if dt < 0.004:
                time.sleep(0.004 - dt)
            
            if int(elapsed * 2) % 20 == 0:
                print(f"Direct Command (Smoothed)... (Angle: {filtered_q[3]:.2f}, Ramp: {ramp*100:.0f}%)")

    except KeyboardInterrupt:
        print("\nInterrupt received. Stopping...")
    finally:
        print("Cleaning up...")
        controller._running = True
        controller.ctrl_dual_arm_go_home()
        controller.shutdown()
        print("Controller stopped.")

if __name__ == "__main__":
    main()
