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


def interp_key_poses(keys: list[tuple[float, dict[str, float]]], alpha: float) -> dict[str, float]:
    if not keys:
        return {}
    if alpha <= keys[0][0]:
        return dict(keys[0][1])
    if alpha >= keys[-1][0]:
        return dict(keys[-1][1])

    for i in range(len(keys) - 1):
        t0, p0 = keys[i]
        t1, p1 = keys[i + 1]
        if alpha <= t1:
            a = smoothstep((alpha - t0) / max(1e-6, (t1 - t0)))
            return interp_pose(p0, p1, a)
    return dict(keys[-1][1])


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
        "left_wrist_roll_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
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

    def mirrored_hand_pose(raw: dict[str, float]) -> dict[str, float]:
        pose = {}
        for side in ("left", "right"):
            for name, val in raw.items():
                pose[f"{side}_hand_{name}_joint"] = val
        return pose

    wrist_joint_names = [
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
    ]

    hand_joint_names = []
    for side in ("left", "right"):
        for name in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"):
            hand_joint_names.append(f"{side}_hand_{name}_joint")

    tracked_hand_and_wrist_joints = set(wrist_joint_names + hand_joint_names)
    joint_ranges: dict[str, tuple[float, float]] = {}
    for jname in tracked_hand_and_wrist_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        if model.jnt_limited[jid]:
            joint_ranges[jname] = (float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1]))

    def normalize_joint_target(jname: str, desired: float) -> float:
        limits = joint_ranges.get(jname)
        if limits is None:
            return float(desired)
        lo, hi = limits
        direct = float(np.clip(desired, lo, hi))
        flipped = float(np.clip(-desired, lo, hi))
        direct_err = abs(abs(direct) - abs(desired))
        flipped_err = abs(abs(flipped) - abs(desired))
        if flipped_err + 1e-12 < direct_err:
            return flipped
        return direct

    def normalize_pose_targets(raw_pose: dict[str, float]) -> dict[str, float]:
        out = {}
        for jname, val in raw_pose.items():
            out[jname] = normalize_joint_target(jname, val)
        return out

    def make_hand_wrist_pose(finger_raw: dict[str, float], wrist_raw: dict[str, float]) -> dict[str, float]:
        raw = {}
        raw.update(mirrored_hand_pose(finger_raw))
        raw.update(wrist_raw)
        return normalize_pose_targets(raw)

    fingers_open = {
        "thumb_0": -0.20,
        "thumb_1": 0.05,
        "thumb_2": 0.05,
        "index_0": 0.05,
        "index_1": 0.00,
        "middle_0": 0.05,
        "middle_1": 0.00,
    }
    fingers_drape = {
        "thumb_0": -0.12,
        "thumb_1": 0.06,
        "thumb_2": 0.06,
        "index_0": -0.18,
        "index_1": 0.18,
        "middle_0": -0.18,
        "middle_1": 0.18,
    }
    fingers_mid_curl = {
        "thumb_0": -0.06,
        "thumb_1": 0.10,
        "thumb_2": 0.12,
        "index_0": -0.50,
        "index_1": 0.55,
        "middle_0": -0.50,
        "middle_1": 0.55,
    }
    fingers_wrap = {
        "thumb_0": 0.05,
        "thumb_1": 0.25,
        "thumb_2": 0.30,
        "index_0": -1.10,
        "index_1": 1.40,
        "middle_0": -1.15,
        "middle_1": 1.45,
    }
    fingers_thumb_lock = {
        "thumb_0": 0.35,
        "thumb_1": 0.70,
        "thumb_2": 1.10,
        "index_0": -1.10,
        "index_1": 1.40,
        "middle_0": -1.15,
        "middle_1": 1.45,
    }

    wrist_neutral = {
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }
    wrist_hover = {
        "left_wrist_pitch_joint": 0.10,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_wrist_pitch_joint": 0.10,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }
    wrist_drape = {
        "left_wrist_pitch_joint": 0.10,
        "left_wrist_roll_joint": 0.087,
        "left_wrist_yaw_joint": 0.0,
        "right_wrist_pitch_joint": 0.10,
        "right_wrist_roll_joint": -0.087,
        "right_wrist_yaw_joint": 0.0,
    }

    hands_open_rack = make_hand_wrist_pose(fingers_open, wrist_neutral)
    hands_hover = make_hand_wrist_pose(fingers_open, wrist_hover)
    hands_drape = make_hand_wrist_pose(fingers_drape, wrist_drape)
    hands_mid_curl = make_hand_wrist_pose(fingers_mid_curl, wrist_drape)
    hands_wrap = make_hand_wrist_pose(fingers_wrap, wrist_drape)
    hands_thumb_lock = make_hand_wrist_pose(fingers_thumb_lock, wrist_drape)
    hands_wrist_stack = make_hand_wrist_pose(fingers_thumb_lock, wrist_neutral)
    hands_squeeze = make_hand_wrist_pose(fingers_thumb_lock, wrist_neutral)

    pose_rack = {**lower_body, **arms_rack}
    pose_grabbed = {**lower_body, **arms_rack}

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
    apply_joint_pose(data, qmap, pose_rack)
    mujoco.mj_forward(model, data)
    rack_lp = data.xpos[left_hand_body].copy() if left_hand_body >= 0 else np.array([0.0, 0.12, 0.0])
    rack_rp = data.xpos[right_hand_body].copy() if right_hand_body >= 0 else np.array([0.0, -0.12, 0.0])

    left_shoulder_anchor = joint_anchor_world("left_shoulder_pitch_joint")
    right_shoulder_anchor = joint_anchor_world("right_shoulder_pitch_joint")
    if left_shoulder_anchor is not None and right_shoulder_anchor is not None:
        shoulder_width = abs(left_shoulder_anchor[1] - right_shoulder_anchor[1])
    else:
        shoulder_width = abs(rack_lp[1] - rack_rp[1])

    grip_width = 1.65 * shoulder_width
    grip_half_width = 0.5 * grip_width

    bar_unrack = np.array([-1.20, 0.0, 0.86], dtype=float)
    bar_bottom = np.array([-1.10, 0.0, 0.70], dtype=float)

    # One clean rep with explicit chest-touch bar path and sequential grip choreography.
    keyframes = [
        (0.0, pose_rack, hands_open_rack, 0.0, None),
        (0.8, pose_rack, hands_hover, 0.0, None),
        (1.8, pose_grabbed, hands_squeeze, 1.0, bar_unrack),
        (3.8, pose_grabbed, hands_squeeze, 1.0, bar_bottom),
        (4.6, pose_grabbed, hands_squeeze, 1.0, bar_bottom),
        (6.4, pose_grabbed, hands_squeeze, 1.0, bar_unrack),
        (7.2, pose_grabbed, hands_squeeze, 1.0, bar_rack),
        (7.8, pose_rack, hands_open_rack, 0.0, None),
    ]

    grip_micro_keys = [
        (0.00, hands_hover),
        (0.30, hands_drape),
        (0.52, hands_mid_curl),
        (0.62, hands_wrap),
        (0.72, hands_wrap),
        (0.82, hands_thumb_lock),
        (0.92, hands_wrist_stack),
        (1.00, hands_squeeze),
    ]
    grip_sequence_segment_index = 1
    total_time = keyframes[-1][0]

    hand_z_micro_keys = [
        (0.00, 0.04),
        (0.30, 0.032),
        (0.52, 0.010),
        (0.62, 0.006),
        (0.82, 0.002),
        (0.92, 0.000),
        (1.00, 0.000),
    ]
    grip_attach_micro_keys = [
        (0.00, 0.0),
        (0.82, 0.0),
        (0.92, 0.35),
        (1.00, 1.0),
    ]
    bar_lift_micro_keys = [
        (0.00, 0.0),
        (0.82, 0.0),
        (0.90, 0.25),
        (1.00, 1.0),
    ]

    def interp_scalar_keys(keys: list[tuple[float, float]], alpha: float) -> float:
        if not keys:
            return 0.0
        if alpha <= keys[0][0]:
            return float(keys[0][1])
        if alpha >= keys[-1][0]:
            return float(keys[-1][1])

        for i in range(len(keys) - 1):
            t0, v0 = keys[i]
            t1, v1 = keys[i + 1]
            if alpha <= t1:
                a = smoothstep((alpha - t0) / max(1e-6, (t1 - t0)))
                return float(v0 + a * (v1 - v0))
        return float(keys[-1][1])

    def sample_state(t: float) -> tuple[dict[str, float], dict[str, float], float, np.ndarray | None, np.ndarray | None, float]:
        if t > total_time:
            return pose_rack, hands_open_rack, 0.0, None, None, -0.02

        i = 0
        while i + 1 < len(keyframes) and t > keyframes[i + 1][0]:
            i += 1

        t0, p0, h0, g0, b0 = keyframes[i]
        t1, p1, h1, g1, b1 = keyframes[i + 1]
        a = smoothstep((t - t0) / max(1e-6, (t1 - t0)))

        pose = interp_pose(p0, p1, a)
        grip = g0 + a * (g1 - g0)

        if i == grip_sequence_segment_index:
            hand_pose = interp_key_poses(grip_micro_keys, a)
            grip = interp_scalar_keys(grip_attach_micro_keys, a)
            bar_lift = interp_scalar_keys(bar_lift_micro_keys, a)
            bar_target = (1.0 - bar_lift) * bar_rack + bar_lift * bar_unrack
            ik_anchor = bar_target
            hand_z_offset = interp_scalar_keys(hand_z_micro_keys, a)
        else:
            hand_pose = interp_pose(h0, h1, a)
            ik_anchor = None
            hand_z_offset = -0.02

        if i != grip_sequence_segment_index:
            if b0 is None and b1 is None:
                bar_target = None
            elif b0 is None:
                bar_target = (1.0 - a) * bar_rack + a * b1
            elif b1 is None:
                bar_target = (1.0 - a) * b0 + a * bar_rack
            else:
                bar_target = (1.0 - a) * b0 + a * b1

        return pose, hand_pose, grip, bar_target, ik_anchor, hand_z_offset

    def apply_state(
        pose: dict[str, float],
        hand_pose: dict[str, float],
        grip: float,
        bar_target: np.ndarray | None,
        ik_anchor: np.ndarray | None,
        hand_z_offset: float,
    ):
        set_free_joint_pose(model, data, "floating_base_joint", robot_base_pos, robot_base_quat)
        apply_joint_pose(data, qmap, pose)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        if bar_target is not None and grip > 1e-4:
            anchor = ik_anchor if ik_anchor is not None else bar_target
            left_target = anchor + np.array([0.0, grip_half_width, hand_z_offset], dtype=float)
            right_target = anchor + np.array([0.0, -grip_half_width, hand_z_offset], dtype=float)
            solve_arm_ik(left_target, right_target)

        apply_joint_pose(data, qmap, hand_pose)
        mujoco.mj_forward(model, data)

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
    apply_state(pose_rack, hands_open_rack, 0.0, None, None, -0.02)

    if HEADLESS:
        start = time.perf_counter()
        while True:
            t = time.perf_counter() - start
            if t > total_time:
                apply_state(pose_rack, hands_open_rack, 0.0, None, None, -0.02)
                break

            pose, hand_pose, grip, bar_target, ik_anchor, hand_z_offset = sample_state(t)
            apply_state(pose, hand_pose, grip, bar_target, ik_anchor, hand_z_offset)

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
            pose = pose_rack
            hand_pose = hands_open_rack
            grip = 0.0
            bar_target = None
            ik_anchor = None
            hand_z_offset = -0.02
            if not done_printed:
                print("One rep complete.")
                done_printed = True
        else:
            pose, hand_pose, grip, bar_target, ik_anchor, hand_z_offset = sample_state(t)

        apply_state(pose, hand_pose, grip, bar_target, ik_anchor, hand_z_offset)
        viewer.sync()
        time.sleep(VIEW_DT)

    viewer.close()


if __name__ == "__main__":
    main()