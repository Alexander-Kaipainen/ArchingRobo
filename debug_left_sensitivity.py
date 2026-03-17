import os
os.environ['_RUNNING_UNDER_MJPYTHON'] = '1'

import numpy as np
import mujoco
from importlib.machinery import SourceFileLoader
mod = SourceFileLoader('mjmod', '/Users/davidloc/Github/ArchingRobo/import mujoco.py').load_module()

model = mujoco.MjModel.from_xml_path('/Users/davidloc/Github/ArchingRobo/unitree_g1/mjcf/scene_barbell.xml')
data = mujoco.MjData(model)

qpos_idx = mod.build_actuator_qpos_index(model)
base = mod.build_desired_pose(model, mod.SQUAT_POSE)

def set_joint(q_des, name, val):
    for act_id in range(model.nu):
        j = model.actuator_trnid[act_id, 0]
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n == name:
            q_des[act_id] = val
            return

left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_wrist_roll_link')
bar_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'barbell')

print('sp el -> lx ly lz dl')
for sp in [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2]:
    for el in [-0.4, -0.2, 0.0, 0.2, 0.4]:
        q = base.copy()
        set_joint(q, 'left_shoulder_pitch_joint', sp)
        set_joint(q, 'left_elbow_joint', el)
        data.qpos[qpos_idx] = q
        data.qpos[2] = 0.78
        data.qpos[3] = 1.0
        mujoco.mj_forward(model, data)
        lpos = data.xpos[left_id].copy()
        bpos = data.xpos[bar_id].copy()
        dl = np.linalg.norm(lpos - bpos)
        print(f'{sp: .2f} {el: .2f} -> {lpos[0]: .3f} {lpos[1]: .3f} {lpos[2]: .3f} {dl: .3f}')
