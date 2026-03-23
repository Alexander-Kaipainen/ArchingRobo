import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(SCRIPT_DIR, "unitree_g1", "mjcf", "scene_29dof_with_hand.xml")
HEADLESS = "--no-viewer" in sys.argv

VIEW_DT = 0.02


def smoothstep(x: float) -> float:
	x = np.clip(x, 0.0, 1.0)
	return x * x * (3.0 - 2.0 * x)


def get_joint_qpos_map(model: mujoco.MjModel) -> dict[str, int]:
	out = {}
	for jid in range(model.njnt):
		name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
		if not name:
			continue
		if model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
			out[name] = model.jnt_qposadr[jid]
	return out


def set_free_joint_pose(
	model: mujoco.MjModel,
	data: mujoco.MjData,
	joint_name: str,
	pos_xyz: np.ndarray,
	quat_wxyz: np.ndarray,
) -> None:
	jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
	if jid < 0:
		return
	qadr = model.jnt_qposadr[jid]
	data.qpos[qadr : qadr + 3] = pos_xyz
	data.qpos[qadr + 3 : qadr + 7] = quat_wxyz


def apply_joint_pose(data: mujoco.MjData, qmap: dict[str, int], pose: dict[str, float]) -> None:
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


def hand_pose(closed: bool) -> dict[str, float]:
	if not closed:
		return {
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
	return {
		"left_hand_thumb_0_joint": 0.35,
		"left_hand_thumb_1_joint": 0.65,
		"left_hand_thumb_2_joint": 0.65,
		"left_hand_index_0_joint": 0.90,
		"left_hand_index_1_joint": 0.90,
		"left_hand_middle_0_joint": 0.90,
		"left_hand_middle_1_joint": 0.90,
		"right_hand_thumb_0_joint": 0.35,
		"right_hand_thumb_1_joint": 0.65,
		"right_hand_thumb_2_joint": 0.65,
		"right_hand_index_0_joint": 0.90,
		"right_hand_index_1_joint": 0.90,
		"right_hand_middle_0_joint": 0.90,
		"right_hand_middle_1_joint": 0.90,
	}


def overhead_extension_pose(elbow_flexion: float, closed_grip: bool) -> dict[str, float]:
	"""
	elbow_flexion in radians:
	  ~1.75 rad -> bottom (forearm behind head)
	  ~0.10 rad -> top (near full extension, not hard lockout)
	"""
	elbow_bottom = 1.75
	elbow_top = 0.10
	ext_progress = np.clip((elbow_bottom - elbow_flexion) / (elbow_bottom - elbow_top), 0.0, 1.0)

	# As elbows flex (load shifts back), dorsiflex ankles slightly and keep knees soft.
	sway_back = 1.0 - ext_progress

	base = {
		# Lower-body stability base (small, continuous compliance).
		"left_hip_pitch_joint": -0.08 - 0.02 * sway_back,
		"right_hip_pitch_joint": -0.08 - 0.02 * sway_back,
		"left_knee_joint": 0.12 + 0.03 * sway_back,
		"right_knee_joint": 0.12 + 0.03 * sway_back,
		"left_ankle_pitch_joint": 0.05 + 0.03 * sway_back,
		"right_ankle_pitch_joint": 0.05 + 0.03 * sway_back,
		"left_ankle_roll_joint": 0.0,
		"right_ankle_roll_joint": 0.0,
		"waist_yaw_joint": 0.0,
		"waist_roll_joint": 0.0,
		"waist_pitch_joint": -0.02,
		# Shoulder setup: full overhead pitch + slight adduction + external rotation.
		"left_shoulder_pitch_joint": -2.75,
		"right_shoulder_pitch_joint": -2.75,
		"left_shoulder_roll_joint": 0.18,
		"right_shoulder_roll_joint": -0.18,
		"left_shoulder_yaw_joint": -0.45,
		"right_shoulder_yaw_joint": 0.45,
		# Prime mover for the rep.
		"left_elbow_joint": elbow_flexion,
		"right_elbow_joint": elbow_flexion,
		# Wrist mostly neutral with slight extension for grip stability.
		"left_wrist_pitch_joint": 0.17,
		"right_wrist_pitch_joint": 0.17,
		"left_wrist_roll_joint": 0.0,
		"right_wrist_roll_joint": 0.0,
		"left_wrist_yaw_joint": 0.0,
		"right_wrist_yaw_joint": 0.0,
	}
	return {**base, **hand_pose(closed_grip)}


def arms_down_pose() -> dict[str, float]:
	return {
		"left_hip_pitch_joint": -0.06,
		"right_hip_pitch_joint": -0.06,
		"left_knee_joint": 0.10,
		"right_knee_joint": 0.10,
		"left_ankle_pitch_joint": 0.04,
		"right_ankle_pitch_joint": 0.04,
		"waist_yaw_joint": 0.0,
		"waist_roll_joint": 0.0,
		"waist_pitch_joint": -0.02,
		"left_shoulder_pitch_joint": -0.30,
		"right_shoulder_pitch_joint": -0.30,
		"left_shoulder_roll_joint": 0.10,
		"right_shoulder_roll_joint": -0.10,
		"left_shoulder_yaw_joint": 0.0,
		"right_shoulder_yaw_joint": 0.0,
		"left_elbow_joint": 0.25,
		"right_elbow_joint": 0.25,
		"left_wrist_pitch_joint": 0.0,
		"right_wrist_pitch_joint": 0.0,
		"left_wrist_roll_joint": 0.0,
		"right_wrist_roll_joint": 0.0,
		"left_wrist_yaw_joint": 0.0,
		"right_wrist_yaw_joint": 0.0,
		**hand_pose(False),
	}


def main() -> None:
	if not os.path.exists(SCENE_PATH):
		raise FileNotFoundError(f"Scene not found: {SCENE_PATH}")

	model = mujoco.MjModel.from_xml_path(SCENE_PATH)
	data = mujoco.MjData(model)
	qmap = get_joint_qpos_map(model)

	robot_base_pos = np.array([0.0, 0.0, 0.793], dtype=float)
	robot_base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

	pose_start = arms_down_pose()
	pose_setup = overhead_extension_pose(elbow_flexion=1.75, closed_grip=True)
	pose_top = overhead_extension_pose(elbow_flexion=0.10, closed_grip=True)
	pose_bottom = overhead_extension_pose(elbow_flexion=1.75, closed_grip=True)

	# Setup -> concentric (press) -> slower eccentric (lower) -> finish.
	keyframes = [
		(0.0, pose_start),
		(1.3, pose_setup),
		(2.5, pose_top),
		(4.5, pose_bottom),
		(5.2, pose_setup),
		(6.0, pose_start),
	]
	total_time = keyframes[-1][0]

	def apply_state(pose: dict[str, float]) -> None:
		set_free_joint_pose(model, data, "floating_base_joint", robot_base_pos, robot_base_quat)
		apply_joint_pose(data, qmap, pose)
		data.qvel[:] = 0.0
		mujoco.mj_forward(model, data)

	apply_state(pose_start)

	if HEADLESS:
		start = time.perf_counter()
		while True:
			t = time.perf_counter() - start
			if t > total_time:
				apply_state(pose_start)
				break

			i = 0
			while i + 1 < len(keyframes) and t > keyframes[i + 1][0]:
				i += 1

			t0, p0 = keyframes[i]
			t1, p1 = keyframes[i + 1]
			a = smoothstep((t - t0) / max(1e-6, (t1 - t0)))
			apply_state(interp_pose(p0, p1, a))

		print("One overhead tricep-extension rep complete.")
		return

	viewer = mujoco.viewer.launch_passive(model, data)
	viewer.cam.distance = 3.6
	viewer.cam.elevation = -15
	viewer.cam.azimuth = 140
	viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.95])

	start = time.perf_counter()
	done_printed = False

	while viewer.is_running():
		t = time.perf_counter() - start
		if t > total_time:
			pose = pose_start
			if not done_printed:
				print("One overhead tricep-extension rep complete.")
				done_printed = True
		else:
			i = 0
			while i + 1 < len(keyframes) and t > keyframes[i + 1][0]:
				i += 1

			t0, p0 = keyframes[i]
			t1, p1 = keyframes[i + 1]
			a = smoothstep((t - t0) / max(1e-6, (t1 - t0)))
			pose = interp_pose(p0, p1, a)

		apply_state(pose)
		viewer.sync()
		time.sleep(VIEW_DT)

	viewer.close()


if __name__ == "__main__":
	main()
