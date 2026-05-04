import time
import sys
import os

# Force CycloneDDS to use ONLY the robot interface
os.environ["CYCLONEDDS_URI"] = """
<CycloneDDS>
    <Domain>
        <General>
            <NetworkInterfaceAddress>enp129s0</NetworkInterfaceAddress>
            <AllowMulticast>true</AllowMulticast>
        </General>
    </Domain>
</CycloneDDS>
"""

sys.path.append(os.getcwd())

def test_connection():
    print("--- Attempting Forced Connection on enp129s0 (Domain 0) ---")
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        
        ChannelFactoryInitialize(0, "enp129s0")
        
        received = [False]
        def cb(msg):
            received[0] = True
            print("  [!!!] SUCCESS! Robot detected!")

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(cb, 10)
        
        for i in range(100):
            if received[0]: return True
            if i % 10 == 0: print(f"  Waiting... ({i//10}s)")
            time.sleep(0.1)
            
        print("  [FAILED] Still no data. Trying Domain 1...")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

if __name__ == "__main__":
    test_connection()
