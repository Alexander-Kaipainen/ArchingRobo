import os
os.environ['_RUNNING_UNDER_MJPYTHON'] = '1'

import numpy as np
import mujoco
from importlib.machinery import SourceFileLoader

mod = SourceFileLoader('mjmod', '/Users/davidloc/Github/ArchingRobo/import mujoco.py').load_module()

model = mujoco.MjModel.from_xml_path('/Users/davidloc/Github/ArchingRobo/unitree_g1/mjcf/scene_barbell.xml')
data = mujoco.MjData(model)
model.opt.timestep = mod.SIMULATE_DT

qpos_idx = mod.build_actuator_qpos_index(model)
qvel_idx = mod.build_actuator_qvel_index(model)
dq_des = np.zeros(model.nu)

gains = mod.ACTUATOR_GAINS[:model.nu]
while len(gains) < model.nu:
    gains.append(mod.GAINS['hand'])

q_stand = mod.build_desired_pose(model, mod.STAND_POSE)
q_squat = mod.build_desired_pose(model, mod.SQUAT_POSE)
q_grab = mod.build_desired_pose(model, mod.SQUAT_GRAB_POSE)
q_sink = mod.build_desired_pose(model, mod.GRAB_SINK_POSE)
q_reach = mod.build_desired_pose(model, mod.FINAL_REACH_POSE)
q_carry = mod.build_desired_pose(model, mod.STAND_CARRY_POSE)
joint_to_act = mod.build_joint_to_actuator_map(model)

l_sh_pitch_id = joint_to_act.get('left_shoulder_pitch_joint')
l_sh_roll_id = joint_to_act.get('left_shoulder_roll_joint')
l_elbow_id = joint_to_act.get('left_elbow_joint')
r_sh_pitch_id = joint_to_act.get('right_shoulder_pitch_joint')
r_sh_roll_id = joint_to_act.get('right_shoulder_roll_joint')
r_elbow_id = joint_to_act.get('right_elbow_joint')

data.qpos[qpos_idx] = q_stand
data.qpos[2] = 0.78
data.qpos[3] = 1.0
mujoco.mj_forward(model, data)

left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_wrist_roll_link')
right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist_roll_link')
bar_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'barbell')

min_left = (1e9, 0.0)
min_right = (1e9, 0.0)

print('phase dl dr dm lx rx ly ry lz rz bz')
steps = int(mod.SQUAT_PERIOD / mod.SIMULATE_DT)
for i in range(steps):
    phase = min(max(data.time / mod.SQUAT_PERIOD, 0.0), 1.0)

    lpos = data.xpos[left_id].copy()
    rpos = data.xpos[right_id].copy()
    bpos = data.xpos[bar_id].copy()

    bquat = mod.normalize_quat(data.xquat[bar_id].copy())
    left_grip = bpos + mod.quat_rotate(bquat, np.array([0.0, mod.BAR_GRIP_OFFSET_Y, 0.0]))
    right_grip = bpos + mod.quat_rotate(bquat, np.array([0.0, -mod.BAR_GRIP_OFFSET_Y, 0.0]))

    dl = float(np.linalg.norm(left_grip - lpos))
    dr = float(np.linalg.norm(right_grip - rpos))
    dm = 0.5 * (dl + dr)

    if dl < min_left[0]:
        min_left = (dl, phase)
    if dr < min_right[0]:
        min_right = (dr, phase)

    if i % 120 == 0 and 0.12 <= phase <= 0.40:
        print(f"{phase:.3f} {dl:.3f} {dr:.3f} {dm:.3f} {lpos[0]:.3f} {rpos[0]:.3f} {lpos[1]:.3f} {rpos[1]:.3f} {lpos[2]:.3f} {rpos[2]:.3f} {bpos[2]:.3f}")

    q_des = mod.squat_cycle(data.time, q_stand, q_squat, q_grab, q_sink, q_carry)

    if left_id >= 0 and right_id >= 0 and bar_id >= 0 and mod.PRE_GRAB_PHASE_ON < phase < mod.GRAB_PHASE_OFF:
        dist_alpha = mod.proximity_blend(dm, mod.APPROACH_DIST_START, mod.APPROACH_DIST_END)
        phase_alpha = 1.0 - mod.proximity_blend(phase, mod.GRAB_PHASE_ON, mod.PRE_GRAB_PHASE_ON)
        reach_alpha = max(dist_alpha, phase_alpha)
        if phase >= mod.GRAB_PHASE_ON:
            reach_alpha = max(reach_alpha, 0.95)
        if reach_alpha > 0.0:
            q_des = (1.0 - reach_alpha) * q_des + reach_alpha * q_reach

        lr_error = dl - dr
        corr = min(abs(lr_error), 0.15)
        if lr_error > 0.0 and None not in (l_sh_pitch_id, l_sh_roll_id, l_elbow_id):
            q_des[l_sh_pitch_id] -= 0.8 * corr
            q_des[l_sh_roll_id] -= 1.0 * corr
            q_des[l_elbow_id] += 0.4 * corr

    q = data.qpos[qpos_idx]
    dq = data.qvel[qvel_idx]
    tau = mod.pd_control(q_des, q, dq_des, dq, gains)
    tau = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    data.ctrl[:] = tau
    mujoco.mj_step(model, data)

print('min_left', min_left)
print('min_right', min_right)
