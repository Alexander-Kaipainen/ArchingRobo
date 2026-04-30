import numpy as np
import time
import threading
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
