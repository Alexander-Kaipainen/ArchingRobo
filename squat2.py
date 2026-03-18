import os
import time

import mujoco
import mujoco.viewer
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_SCENE = os.path.join(SCRIPT_DIR, "unitree_g1", "mjcf", "scene_29dof_with_hand.xml")

SIM_DT = 0.002


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
	GAINS["hand"], GAINS["hand"], GAINS["hand"], GAINS["hand"], GAINS["hand"], GAINS["hand"], GAINS["hand"],
	GAINS["arm"], GAINS["arm"], GAINS["arm"],
	GAINS["arm"],
	GAINS["wrist"], GAINS["wrist"], GAINS["wrist"],
	GAINS["hand"], GAINS["hand"], GAINS["hand"], GAINS["hand"], GAINS["hand"], GAINS["hand"], GAINS["hand"],
]


STAND_POSE = {
	"left_hip_pitch_joint": -0.40,
	"left_knee_joint": 0.80,
	"left_ankle_pitch_joint": -0.40,
	"right_hip_pitch_joint": -0.40,
	"right_knee_joint": 0.80,
	"right_ankle_pitch_joint": -0.40,
	"left_shoulder_roll_joint": 0.20,
	"right_shoulder_roll_joint": -0.20,
}


PUSHUP_READY_POSE = {
	"left_hip_pitch_joint": -1.35,
	"left_knee_joint": 2.25,
	"left_ankle_pitch_joint": -0.95,
	"right_hip_pitch_joint": -1.35,
	"right_knee_joint": 2.25,
	"right_ankle_pitch_joint": -0.95,
	"waist_pitch_joint": 0.52,
	"left_shoulder_pitch_joint": 1.15,
	"right_shoulder_pitch_joint": 1.15,
	"left_shoulder_roll_joint": 0.10,
	"right_shoulder_roll_joint": -0.10,
	"left_elbow_joint": -1.55,
	"right_elbow_joint": -1.55,
	"left_wrist_pitch_joint": 0.45,
	"right_wrist_pitch_joint": 0.45,
}


START_HOLD_S = 1.2
DESCENT_S = 5.0
BOTTOM_HOLD_S = 0.7
ASCENT_S = 4.0
TOP_HOLD_S = 0.7


def build_actuator_qpos_index(model: mujoco.MjModel) -> np.ndarray:
	idx = np.zeros(model.nu, dtype=int)
	for act_id in range(model.nu):
		joint_id = model.actuator_trnid[act_id, 0]
		idx[act_id] = model.jnt_qposadr[joint_id]
	return idx


def build_actuator_qvel_index(model: mujoco.MjModel) -> np.ndarray:
	idx = np.zeros(model.nu, dtype=int)
	for act_id in range(model.nu):
		joint_id = model.actuator_trnid[act_id, 0]
		idx[act_id] = model.jnt_dofadr[joint_id]
	return idx


def build_desired_pose(model: mujoco.MjModel, pose: dict) -> np.ndarray:
	q_des = np.zeros(model.nu)
	for act_id in range(model.nu):
		joint_id = model.actuator_trnid[act_id, 0]
		name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
		if name in pose:
			q_des[act_id] = pose[name]
	return q_des


def smoothstep(x: float) -> float:
	x = np.clip(x, 0.0, 1.0)
	return x * x * (3.0 - 2.0 * x)


def pose_schedule(t: float, q_stand: np.ndarray, q_pushup: np.ndarray) -> np.ndarray:
	if t < START_HOLD_S:
		return q_stand

	cycle_t = t - START_HOLD_S
	cycle_len = DESCENT_S + BOTTOM_HOLD_S + ASCENT_S + TOP_HOLD_S
	phase = cycle_t % cycle_len

	if phase < DESCENT_S:
		alpha = smoothstep(phase / DESCENT_S)
	elif phase < DESCENT_S + BOTTOM_HOLD_S:
		alpha = 1.0
	elif phase < DESCENT_S + BOTTOM_HOLD_S + ASCENT_S:
		rise_t = phase - (DESCENT_S + BOTTOM_HOLD_S)
		alpha = 1.0 - smoothstep(rise_t / ASCENT_S)
	else:
		alpha = 0.0

	return (1.0 - alpha) * q_stand + alpha * q_pushup


def pd_control(q_des, q, dq_des, dq, gains):
	kp = np.array([g["kp"] for g in gains])
	kd = np.array([g["kd"] for g in gains])
	return kp * (q_des - q) + kd * (dq_des - dq)


def main():
	if not os.path.exists(ROBOT_SCENE):
		print(f"Scene XML not found: {ROBOT_SCENE}")
		return

	print(f"Loading scene: {ROBOT_SCENE}")
	t0 = time.perf_counter()
	model = mujoco.MjModel.from_xml_path(ROBOT_SCENE)
	data = mujoco.MjData(model)
	print(f"Loaded in {time.perf_counter() - t0:.2f}s")

	model.opt.timestep = SIM_DT

	qpos_idx = build_actuator_qpos_index(model)
	qvel_idx = build_actuator_qvel_index(model)
	q_stand = build_desired_pose(model, STAND_POSE)
	q_pushup = build_desired_pose(model, PUSHUP_READY_POSE)
	dq_des = np.zeros(model.nu)

	gains = ACTUATOR_GAINS[:model.nu]
	while len(gains) < model.nu:
		gains.append(GAINS["hand"])

	data.qpos[qpos_idx] = q_stand
	data.qpos[2] = 0.78
	data.qpos[3] = 1.0
	data.qpos[4:7] = 0.0
	mujoco.mj_forward(model, data)

	def controller(m, d):
		q_target = pose_schedule(d.time, q_stand, q_pushup)
		q = d.qpos[qpos_idx]
		dq = d.qvel[qvel_idx]
		tau = pd_control(q_target, q, dq_des, dq, gains)
		tau = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
		d.ctrl[:] = tau

	mujoco.set_mjcb_control(controller)

	print("Starting viewer: repeating stand -> lower -> rise loop.")
	mujoco.viewer.launch(model, data)


if __name__ == "__main__":
	main()
