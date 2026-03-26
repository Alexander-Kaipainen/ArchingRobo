import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(SCRIPT_DIR, "unitree_g1", "mjcf", "scene_bench_press_one_rep.xml")
HEADLESS = "--no-viewer" in sys.argv

VIEW_DT = 0.02


def smoothstep(x: float) -> float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=float)


def get_joint_qpos_map(model: mujoco.MjModel) -> dict[str, int]:
    out = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name:
            continue
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            out[name] = model.jnt_qposadr[j]
    return out


def set_free_joint_pose(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, pos_xyz: np.ndarray, quat_wxyz: np.ndarray):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return
    qadr = model.jnt_qposadr[jid]
    data.qpos[qadr:qadr + 3] = pos_xyz
    data.qpos[qadr + 3:qadr + 7] = quat_wxyz


def apply_joint_pose(data: mujoco.MjData, qmap: dict[str, int], pose: dict[str, float]):
    for jname, val in pose.items():
        qadr = qmap.get(jname)
        if qadr is not None:
            data.qpos[qadr] = val


def interp_pose(a: dict[str, float], b: dict[str, float], t: float) -> dict[str, float]:
    keys = set(a.keys()) | set(b.keys())
    out = {}
    for key in keys:
        av = a.get(key, 0.0)
        bv = b.get(key, 0.0)
        out[key] = av + t * (bv - av)
    return out


def main():
    if not os.path.exists(SCENE_PATH):
        raise FileNotFoundError(f"Scene not found: {SCENE_PATH}")

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    qmap = get_joint_qpos_map(model)

    left_hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    right_hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")

    # Spawn fixed lying pose (kinematic, no balancing)
    robot_base_pos = np.array([-0.79, 0.0, 0.58], dtype=float)
    robot_base_quat = quat_from_rpy(3.14159265, -1.5708, 3.14159265)
    bar_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    bar_rack = np.array([-1.25, 0.0, 0.945], dtype=float)

    # Lower body fixed to avoid twitching
    lower_body = {
        "left_hip_pitch_joint": -0.35,
        "right_hip_pitch_joint": -0.35,
        "left_knee_joint": 0.95,
        "right_knee_joint": 0.95,
        "left_ankle_pitch_joint": -0.45,
        "right_ankle_pitch_joint": -0.45,
        "waist_pitch_joint": -0.10,
    }

    arms_rack = {
        "left_shoulder_pitch_joint": -1.18,
        "right_shoulder_pitch_joint": -1.18,
        "left_shoulder_roll_joint": 0.27,
        "right_shoulder_roll_joint": -0.27,
        "left_shoulder_yaw_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.06,
        "right_elbow_joint": 0.06,
        "left_wrist_pitch_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
    }

    arm_ik_joint_names = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_pitch_joint",
    ]
    arm_ik_scales = np.array([0.28, 0.22, 0.18, 1.00, 0.55, 0.28, 0.22, 0.18, 1.00, 0.55], dtype=float)

    arm_ik_qadr = []
    arm_ik_dadr = []
    arm_ik_min = []
    arm_ik_max = []
    for name in arm_ik_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        arm_ik_qadr.append(model.jnt_qposadr[jid])
        arm_ik_dadr.append(model.jnt_dofadr[jid])
        arm_ik_min.append(model.jnt_range[jid][0])
        arm_ik_max.append(model.jnt_range[jid][1])
    arm_ik_qadr = np.array(arm_ik_qadr, dtype=int)
    arm_ik_dadr = np.array(arm_ik_dadr, dtype=int)
    arm_ik_min = np.array(arm_ik_min, dtype=float)
    arm_ik_max = np.array(arm_ik_max, dtype=float)

    hands_open = {
        "left_hand_thumb_0_joint": 0.0,
        "left_hand_thumb_1_joint": 0.0,
        "left_hand_thumb_2_joint": 0.0,
        "left_hand_index_0_joint": 0.0,
        "left_hand_index_1_joint": 0.0,
        "left_hand_middle_0_joint": 0.0,
        "left_hand_middle_1_joint": 0.0,
        "right_hand_thumb_0_joint": 0.0,
        "right_hand_thumb_1_joint": 0.0,
        "right_hand_thumb_2_joint": 0.0,
        "right_hand_index_0_joint": 0.0,
        "right_hand_index_1_joint": 0.0,
        "right_hand_middle_0_joint": 0.0,
        "right_hand_middle_1_joint": 0.0,
    }

    hands_closed = {
        "left_hand_thumb_0_joint": 0.35,
        "left_hand_thumb_1_joint": 0.65,
        "left_hand_thumb_2_joint": 0.65,
        "left_hand_index_0_joint": 0.9,
        "left_hand_index_1_joint": 0.9,
        "left_hand_middle_0_joint": 0.9,
        "left_hand_middle_1_joint": 0.9,
        "right_hand_thumb_0_joint": 0.35,
        "right_hand_thumb_1_joint": 0.65,
        "right_hand_thumb_2_joint": 0.65,
        "right_hand_index_0_joint": 0.9,
        "right_hand_index_1_joint": 0.9,
        "right_hand_middle_0_joint": 0.9,
        "right_hand_middle_1_joint": 0.9,
    }

    pose_open_rack = {**lower_body, **arms_rack, **hands_open}
    pose_closed_rack = {**lower_body, **arms_rack, **hands_closed}
    pose_grabbed = {**lower_body, **hands_closed}

    def attached_bar_pos() -> np.ndarray:
        if left_hand_body < 0 or right_hand_body < 0:
            return bar_rack.copy()
        lp = data.xpos[left_hand_body].copy()
        rp = data.xpos[right_hand_body].copy()
        center = 0.5 * (lp + rp)
        return center + np.array([0.0, 0.0, 0.02], dtype=float)

    def solve_arm_ik(left_target: np.ndarray, right_target: np.ndarray):
        if left_hand_body < 0 or right_hand_body < 0:
            return
        if arm_ik_dadr.size == 0:
            return

        jacp_l = np.zeros((3, model.nv), dtype=float)
        jacp_r = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        damping = 0.04

        for _ in range(16):
            mujoco.mj_forward(model, data)
            lp = data.xpos[left_hand_body].copy()
            rp = data.xpos[right_hand_body].copy()
            err = np.concatenate([left_target - lp, right_target - rp], axis=0)
            if np.linalg.norm(err) < 2e-3:
                break

            jacp_l.fill(0.0)
            jacp_r.fill(0.0)
            jacr.fill(0.0)
            mujoco.mj_jacBody(model, data, jacp_l, jacr, left_hand_body)
            jacr.fill(0.0)
            mujoco.mj_jacBody(model, data, jacp_r, jacr, right_hand_body)

            j_full = np.vstack([jacp_l[:, arm_ik_dadr], jacp_r[:, arm_ik_dadr]])
            j_weighted = j_full * arm_ik_scales[np.newaxis, :]

            lhs = j_weighted @ j_weighted.T + (damping * damping) * np.eye(6, dtype=float)
            rhs = 0.85 * err
            step_weighted = j_weighted.T @ np.linalg.solve(lhs, rhs)
            step = step_weighted * arm_ik_scales
            step = np.clip(step, -0.08, 0.08)

            data.qpos[arm_ik_qadr] = np.clip(data.qpos[arm_ik_qadr] + step, arm_ik_min, arm_ik_max)

        mujoco.mj_forward(model, data)

    def joint_anchor_world(joint_name: str) -> np.ndarray | None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return None
        bid = model.jnt_bodyid[jid]
        rot = data.xmat[bid].reshape(3, 3)
        return data.xpos[bid].copy() + rot @ model.jnt_pos[jid]

    set_free_joint_pose(model, data, "floating_base_joint", robot_base_pos, robot_base_quat)
    apply_joint_pose(data, qmap, pose_closed_rack)
    mujoco.mj_forward(model, data)
    rack_lp = data.xpos[left_hand_body].copy() if left_hand_body >= 0 else np.array([0.0, 0.12, 0.0])
    rack_rp = data.xpos[right_hand_body].copy() if right_hand_body >= 0 else np.array([0.0, -0.12, 0.0])

    left_shoulder_anchor = joint_anchor_world("left_shoulder_pitch_joint")
    right_shoulder_anchor = joint_anchor_world("right_shoulder_pitch_joint")
    if left_shoulder_anchor is not None and right_shoulder_anchor is not None:
        shoulder_width = abs(left_shoulder_anchor[1] - right_shoulder_anchor[1])
    else:
        shoulder_width = abs(rack_lp[1] - rack_rp[1])

    grip_width = 1.5 * shoulder_width
    grip_half_width = 0.5 * grip_width

    bar_unrack = np.array([-1.20, 0.0, 0.86], dtype=float)
    bar_bottom = np.array([-1.10, 0.0, 0.70], dtype=float)

    # One clean rep with explicit chest-touch bar path
    keyframes = [
        (0.0, pose_open_rack, 0.0, None),
        (0.8, pose_closed_rack, 0.0, None),
        (1.8, pose_grabbed, 1.0, bar_unrack),
        (3.8, pose_grabbed, 1.0, bar_bottom),
        (4.6, pose_grabbed, 1.0, bar_bottom),
        (6.4, pose_grabbed, 1.0, bar_unrack),
        (7.2, pose_grabbed, 1.0, bar_rack),
        (7.8, pose_open_rack, 0.0, None),
    ]
    total_time = keyframes[-1][0]

    def apply_state(pose: dict[str, float], grip: float, bar_target: np.ndarray | None):
        set_free_joint_pose(model, data, "floating_base_joint", robot_base_pos, robot_base_quat)
        apply_joint_pose(data, qmap, pose)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        if bar_target is not None and grip > 1e-4:
            left_target = bar_target + np.array([0.0, grip_half_width, -0.02], dtype=float)
            right_target = bar_target + np.array([0.0, -grip_half_width, -0.02], dtype=float)
            solve_arm_ik(left_target, right_target)

        grip = float(np.clip(grip, 0.0, 1.0))
        if bar_target is None:
            attached = attached_bar_pos()
            bar_pos = (1.0 - grip) * bar_rack + grip * attached
        else:
            bar_pos = (1.0 - grip) * bar_rack + grip * bar_target

        set_free_joint_pose(model, data, "barbell_free", bar_pos, bar_quat)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    # Initial settle
    apply_state(pose_open_rack, 0.0, None)

    if HEADLESS:
        start = time.perf_counter()
        while True:
            t = time.perf_counter() - start
            if t > total_time:
                apply_state(pose_open_rack, 0.0, None)
                break

            i = 0
            while i + 1 < len(keyframes) and t > keyframes[i + 1][0]:
                i += 1
            t0, p0, g0, b0 = keyframes[i]
            t1, p1, g1, b1 = keyframes[i + 1]
            a = smoothstep((t - t0) / max(1e-6, (t1 - t0)))
            pose = interp_pose(p0, p1, a)
            grip = g0 + a * (g1 - g0)

            if b0 is None and b1 is None:
                bar_target = None
            elif b0 is None:
                bar_target = b1
            elif b1 is None:
                bar_target = b0
            else:
                bar_target = (1.0 - a) * b0 + a * b1

            apply_state(pose, grip, bar_target)

        print("One rep complete.")
        return

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance = 4.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 145
    viewer.cam.lookat[:] = np.array([-0.90, 0.0, 0.75])

    start = time.perf_counter()
    done_printed = False

    while viewer.is_running():
        t = time.perf_counter() - start
        if t > total_time:
            pose = pose_open_rack
            grip = 0.0
            bar_target = None
            if not done_printed:
                print("One rep complete.")
                done_printed = True
        else:
            i = 0
            while i + 1 < len(keyframes) and t > keyframes[i + 1][0]:
                i += 1
            t0, p0, g0, b0 = keyframes[i]
            t1, p1, g1, b1 = keyframes[i + 1]
            a = smoothstep((t - t0) / max(1e-6, (t1 - t0)))
            pose = interp_pose(p0, p1, a)
            grip = g0 + a * (g1 - g0)

            if b0 is None and b1 is None:
                bar_target = None
            elif b0 is None:
                bar_target = b1
            elif b1 is None:
                bar_target = b0
            else:
                bar_target = (1.0 - a) * b0 + a * b1

        apply_state(pose, grip, bar_target)
        viewer.sync()
        time.sleep(VIEW_DT)

    viewer.close()


if __name__ == "__main__":
    main()