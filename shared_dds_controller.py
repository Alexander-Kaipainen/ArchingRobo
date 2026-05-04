import numpy as np
import time
import threading
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

CONTROL_DT = 1.0 / 250.0
SAFETY_GAIN_SCALE = 0.70
MAX_KP = 350.0
MAX_KD = 15.0

HAND_INDICES = list(range(29, 35))

MAX_DQ_PER_SEC = np.full(35, 1.0, dtype=float)
MAX_DQ_PER_SEC[:12] = 2.0
MAX_DQ_PER_SEC[12:15] = 1.5
MAX_DQ_PER_SEC[15:29] = 1.2
MAX_DQ_PER_SEC[19:22] = 1.0
MAX_DQ_PER_SEC[26:29] = 1.0
MAX_DQ_PER_SEC[29:] = 0.5

JOINT_LIMITS_MIN = np.full(35, np.nan, dtype=float)
JOINT_LIMITS_MAX = np.full(35, np.nan, dtype=float)
JOINT_LIMITS_MIN[15:22] = [-1.4, -1.6, -1.4, -1.6, -1.6, -1.2, -1.6]
JOINT_LIMITS_MAX[15:22] = [1.4, 1.6, 1.4, 1.6, 1.6, 1.2, 1.6]
JOINT_LIMITS_MIN[22:29] = [-1.4, -1.6, -1.4, -1.6, -1.6, -1.2, -1.6]
JOINT_LIMITS_MAX[22:29] = [1.4, 1.6, 1.4, 1.6, 1.6, 1.2, 1.6]

def clamp_joint_limits(q_cmd: np.ndarray) -> np.ndarray:
    q_safe = q_cmd.copy()
    for idx in range(35):
        q_min = JOINT_LIMITS_MIN[idx]
        q_max = JOINT_LIMITS_MAX[idx]
        if not np.isnan(q_min) and q_safe[idx] < q_min:
            q_safe[idx] = q_min
        if not np.isnan(q_max) and q_safe[idx] > q_max:
            q_safe[idx] = q_max
    return q_safe

class DDSFullBodyController:
    def __init__(self, network_interface="lo", domain_id=1):
        self.NUM_MOTORS = 35
        self.running = True
        
        ChannelFactoryInitialize(domain_id, network_interface)
        
        self.current_q = np.zeros(self.NUM_MOTORS)
        self.current_dq = np.zeros(self.NUM_MOTORS)
        self.latest_state_time = 0.0
        
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._on_lowstate, 10)
        
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        
        self.state_ready = threading.Event()
        
        if not self.state_ready.wait(timeout=10.0):
            raise TimeoutError("Failed to receive rt/lowstate within 10 seconds. Check connection.")
            
        print("DDS Full Body Controller Initialized!")

    def _on_lowstate(self, msg):
        limit = min(self.NUM_MOTORS, len(msg.motor_state))
        for idx in range(limit):
            self.current_q[idx] = msg.motor_state[idx].q
            self.current_dq[idx] = msg.motor_state[idx].dq
        self.msg.mode_machine = msg.mode_machine
        self.latest_state_time = time.time()
        self.state_ready.set()
        
    def write_cmd(self, target_q, target_dq, kps, kds):
        if self.latest_state_time and (time.time() - self.latest_state_time) > 0.5:
            return

        q_safe = clamp_joint_limits(target_q)
        max_delta = MAX_DQ_PER_SEC * CONTROL_DT
        delta = np.clip(q_safe - self.current_q, -max_delta, max_delta)
        q_safe = self.current_q + delta

        kp_safe = np.clip(kps, 0.0, MAX_KP)
        kd_safe = np.clip(kds, 0.0, MAX_KD)
        kp_safe[HAND_INDICES] = 0.0
        kd_safe[HAND_INDICES] = 0.0
        q_safe[HAND_INDICES] = self.current_q[HAND_INDICES]

        for idx in range(self.NUM_MOTORS):
            self.msg.motor_cmd[idx].mode = 0x01
            self.msg.motor_cmd[idx].q = float(q_safe[idx])
            self.msg.motor_cmd[idx].dq = float(target_dq[idx])
            self.msg.motor_cmd[idx].tau = 0.0
            self.msg.motor_cmd[idx].kp = float(kp_safe[idx])
            self.msg.motor_cmd[idx].kd = float(kd_safe[idx])
        
        self.msg.crc = self.crc.Crc(self.msg)
        self.lowcmd_publisher.Write(self.msg)
