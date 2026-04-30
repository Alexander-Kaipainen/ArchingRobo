import numpy as np
import time
import sys
import os
import argparse
import threading

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)


# Fix /tmp/cdds.LOG conflict for multiple DDS clients
try:
    import unitree_sdk2py.core.channel as dds_channel
    trace_config = getattr(dds_channel, "ChannelConfigHasInterface", "")
    if "/tmp/cdds.LOG" in trace_config:
        trace_path = f"/tmp/cdds.{os.getpid()}.LOG"
        dds_channel.ChannelConfigHasInterface = trace_config.replace("/tmp/cdds.LOG", trace_path)
except Exception:
    pass

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

class DDSFullBodyController:
    def __init__(self, network_interface="lo", domain_id=1):
        self.NUM_MOTORS = 35
        self.running = True
        
        ChannelFactoryInitialize(domain_id, network_interface)
        
        self.current_q = np.zeros(self.NUM_MOTORS)
        self.current_dq = np.zeros(self.NUM_MOTORS)
        
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        
        self.state_ready = threading.Event()
        
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._on_lowstate, 10)
        
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        
        if not self.state_ready.wait(timeout=10.0):
            raise TimeoutError("Failed to receive rt/lowstate within 10 seconds. Check connection.")
            
        print("DDS Full Body Controller Initialized!")

    def _on_lowstate(self, msg):
        limit = min(self.NUM_MOTORS, len(msg.motor_state))
        for idx in range(limit):
            self.current_q[idx] = msg.motor_state[idx].q
            self.current_dq[idx] = msg.motor_state[idx].dq
        self.msg.mode_machine = msg.mode_machine
        self.state_ready.set()
        
    def write_cmd(self, target_q, target_dq, kps, kds):
        for idx in range(self.NUM_MOTORS):
            self.msg.motor_cmd[idx].mode = 0x01
            self.msg.motor_cmd[idx].q = float(target_q[idx])
            self.msg.motor_cmd[idx].dq = float(target_dq[idx])
            self.msg.motor_cmd[idx].tau = 0.0
            self.msg.motor_cmd[idx].kp = float(kps[idx])
            self.msg.motor_cmd[idx].kd = float(kds[idx])
        
        self.msg.crc = self.crc.Crc(self.msg)
        self.lowcmd_publisher.Write(self.msg)

JOINT_NAME_MAP = {
    "left_hip_pitch_joint": 0, "left_hip_roll_joint": 1, "left_hip_yaw_joint": 2,
    "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6, "right_hip_roll_joint": 7, "right_hip_yaw_joint": 8,
    "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14,
    "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16, "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19, "left_wrist_pitch_joint": 20, "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23, "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26, "right_wrist_pitch_joint": 27, "right_wrist_yaw_joint": 28
}

GAINS = {
    "leg": {"kp": 500.0, "kd": 20.0},
    "ankle": {"kp": 200.0, "kd": 10.0},
    "waist": {"kp": 400.0, "kd": 15.0},
    "arm": {"kp": 220.0, "kd": 8.0},
    "wrist": {"kp": 90.0, "kd": 3.0},
    "hand": {"kp": 10.0, "kd": 0.5},
}

ACTUATOR_GAINS = [
    GAINS["leg"], GAINS["leg"], GAINS["leg"],
    GAINS["leg"],
    GAINS["ankle"], GAINS["ankle"],
    GAINS["leg"], GAINS["leg"], GAINS["leg"],
    GAINS["leg"],
    GAINS["ankle"], GAINS["ankle"],
    GAINS["waist"], GAINS["waist"], GAINS["waist"],
    GAINS["arm"], GAINS["arm"], GAINS["arm"],
    GAINS["arm"],
    GAINS["wrist"], GAINS["wrist"], GAINS["wrist"],
    GAINS["arm"], GAINS["arm"], GAINS["arm"],
    GAINS["arm"],
    GAINS["wrist"], GAINS["wrist"], GAINS["wrist"],
]

STAND_POSE = {
    "left_hip_pitch_joint": -0.40,
    "left_knee_joint": 0.80,
    "left_ankle_pitch_joint": -0.40,
    "right_hip_pitch_joint": -0.40,
    "right_knee_joint": 0.80,
    "right_ankle_pitch_joint": -0.40,
    "left_shoulder_pitch_joint": 0.00,
    "left_shoulder_roll_joint": 0.08,
    "left_shoulder_yaw_joint": 0.00,
    "left_elbow_joint": 0.18,
    "left_wrist_roll_joint": 0.00,
    "left_wrist_pitch_joint": 0.00,
    "left_wrist_yaw_joint": 0.00,
    "right_shoulder_pitch_joint": 0.00,
    "right_shoulder_roll_joint": -0.08,
    "right_shoulder_yaw_joint": 0.00,
    "right_elbow_joint": 0.18,
    "right_wrist_roll_joint": 0.00,
    "right_wrist_pitch_joint": 0.00,
    "right_wrist_yaw_joint": 0.00,
}

LATERAL_RAISE_POSE = {
    "left_hip_pitch_joint": -0.40,
    "left_knee_joint": 0.80,
    "left_ankle_pitch_joint": -0.40,
    "right_hip_pitch_joint": -0.40,
    "right_knee_joint": 0.80,
    "right_ankle_pitch_joint": -0.40,
    "waist_pitch_joint": 0.00,
    "left_shoulder_pitch_joint": 0.05,
    "left_shoulder_roll_joint": 1.55,
    "left_shoulder_yaw_joint": 0.00,
    "left_elbow_joint": 0.55,
    "left_wrist_roll_joint": 0.00,
    "left_wrist_pitch_joint": 0.00,
    "left_wrist_yaw_joint": 0.00,
    "right_shoulder_pitch_joint": -0.05,
    "right_shoulder_roll_joint": -1.55,
    "right_shoulder_yaw_joint": 0.00,
    "right_elbow_joint": 0.55,
    "right_wrist_roll_joint": 0.00,
    "right_wrist_pitch_joint": 0.00,
    "right_wrist_yaw_joint": 0.00,
}

START_HOLD_S = 1.0
RAISE_S = 3.5
TOP_HOLD_S = 0.8
LOWER_S = 3.5
REST_HOLD_S = 0.8

def smoothstep(x: float) -> float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def pose_schedule(t: float, q_stand: np.ndarray, q_raise: np.ndarray) -> np.ndarray:
    if t < START_HOLD_S:
        return q_stand

    cycle_t = t - START_HOLD_S
    cycle_len = RAISE_S + TOP_HOLD_S + LOWER_S + REST_HOLD_S
    phase = cycle_t % cycle_len

    if phase < RAISE_S:
        alpha = smoothstep(phase / RAISE_S)
    elif phase < RAISE_S + TOP_HOLD_S:
        alpha = 1.0
    elif phase < RAISE_S + TOP_HOLD_S + LOWER_S:
        down_t = phase - (RAISE_S + TOP_HOLD_S)
        alpha = 1.0 - smoothstep(down_t / LOWER_S)
    else:
        alpha = 0.0

    return (1.0 - alpha) * q_stand + alpha * q_raise

def build_dds_pose(target_dict, current_q):
    ret = current_q.copy()
    for name, angle in target_dict.items():
        if name in JOINT_NAME_MAP:
            ret[JOINT_NAME_MAP[name]] = angle
    return ret

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode (lo, Domain 1)")
    parser.add_argument("--interface", type=str, default="enp129s0")
    parser.add_argument("--domain", type=int, default=None)
    args = parser.parse_args()

    if args.sim:
        interface = "lo"
        domain_id = args.domain if args.domain is not None else 1
        print(">>> RUNNING IN SIMULATION MODE (lo, Domain 1) <<<")
    else:
        interface = args.interface
        domain_id = args.domain if args.domain is not None else 0
        print(f">>> RUNNING ON REAL ROBOT ({interface}, Domain {domain_id}) <<<")
    
    try:
        controller = DDSFullBodyController(network_interface=interface, domain_id=domain_id)
    except Exception as e:
        print("Initialization failed.", e)
        return

    # Prepare arrays
    kps = np.zeros(35)
    kds = np.zeros(35)
    for i in range(min(35, len(ACTUATOR_GAINS))):
        kps[i] = ACTUATOR_GAINS[i]["kp"]
        kds[i] = ACTUATOR_GAINS[i]["kd"]
    # Fill remaining with hand gains/defaults to be safe
    for i in range(len(ACTUATOR_GAINS), 35):
        kps[i] = GAINS["hand"]["kp"]
        kds[i] = GAINS["hand"]["kd"]

    start_q = controller.current_q.copy()
    q_stand = build_dds_pose(STAND_POSE, start_q)
    q_raise = build_dds_pose(LATERAL_RAISE_POSE, start_q)
    
    print("\nMoving to initial STAND posture smoothly...")
    duration = 3.0
    steps = int(duration * 250)
    for i in range(steps):
        t0 = time.perf_counter()
        alpha = smoothstep((i + 1) / steps)
        cmd_q = (1 - alpha) * start_q + alpha * q_stand
        cmd_dq = np.zeros(35)
        controller.write_cmd(cmd_q, cmd_dq, kps, kds)
        dt = time.perf_counter() - t0
        if dt < 0.004:
            time.sleep(0.004 - dt)

    print("\nStarting Lateral Raise cycle loop. Press Ctrl+C to stop.")
    
    try:
        start_time = time.time()
        while True:
            t0 = time.perf_counter()
            elapsed = time.time() - start_time
            
            target_q = pose_schedule(elapsed, q_stand, q_raise)
            target_dq = np.zeros(35) 
            
            controller.write_cmd(target_q, target_dq, kps, kds)
            
            dt = time.perf_counter() - t0
            if dt < 0.004:
                time.sleep(0.004 - dt)
                
    except KeyboardInterrupt:
        print("\nStopping...")
        
if __name__ == "__main__":
    main()
