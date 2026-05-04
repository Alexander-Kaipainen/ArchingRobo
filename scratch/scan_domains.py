import time
import sys
import os
import numpy as np

# Add the current dir to path to find unitree modules
sys.path.append(os.getcwd())

def test_domain(domain_id, interface="enp129s0"):
    print(f"--- Scanning Domain {domain_id} on {interface} ---")
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        
        ChannelFactoryInitialize(domain_id, interface)
        
        received = [False]
        def cb(msg):
            received[0] = True
            print(f"  [SUCCESS] Received LowState on Domain {domain_id}!")

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(cb, 10)
        
        for _ in range(50): # 5 seconds
            if received[0]:
                return True
            time.sleep(0.1)
            
        print(f"  [TIMEOUT] No data on Domain {domain_id}")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

if __name__ == "__main__":
    for d in [0, 1, 2]:
        if test_domain(d):
            print(f"\n>>> FOUND ROBOT ON DOMAIN {d} <<<")
            sys.exit(0)
    print("\nRobot not found on Domains 0, 1, or 2.")
