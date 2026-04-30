import re

with open("brunken_voice.py", "r") as f:
    text = f.read()

# 1. Add mappings at the top after RIGHT_ARM_INDICES
if "JOINT_NAME_MAP" not in text:
    constants = """
JOINT_NAME_MAP = {
    "left_hip_pitch_joint": 0, "left_hip_roll_joint": 1, "left_hip_yaw_joint": 2,
    "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6, "right_hip_roll_joint": 7, "right_hip_yaw_joint": 8,
    "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14,
    "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16, "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19, "left_wrist_pitch_joint": 20, "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23, "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26, "right_wrist_pitch_joint": 27, "right_wrist_yaw_joint": 28
}

def build_dds_pose(target_dict, current_q):
    ret = current_q.copy()
    for name, angle in target_dict.items():
        if name in JOINT_NAME_MAP:
            ret[JOINT_NAME_MAP[name]] = angle
    return ret

def smoothstep(x: float) -> float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)
"""
    text = text.replace("NUM_MOTORS = 35", "NUM_MOTORS = 35\n" + constants)

# 2. Add squat and lateral raise methods to LowLevelSaluteController
methods = """
    def _publish_target_kp_kd(self, target_kp: np.ndarray, target_kd: np.ndarray, duration: float) -> None:
        duration = max(0.0, duration)
        steps = max(1, int(duration * 250))
        with self.state_lock:
            start_kp = self._target_kp.copy()
            start_kd = self._target_kd.copy()
        for step in range(steps):
            alpha = (step + 1) / steps
            with self.state_lock:
                self._target_kp[:] = (1.0 - alpha) * start_kp + alpha * target_kp
                self._target_kd[:] = (1.0 - alpha) * start_kd + alpha * target_kd
            time.sleep(0.004)

    def start_squat(self) -> None:
        self.stop_animations()
        self._animation_stop_event.clear()
        self._animation_thread = threading.Thread(target=self._squat_loop, daemon=True)
        self._animation_thread.start()

    def start_lateral_raise(self) -> None:
        self.stop_animations()
        self._animation_stop_event.clear()
        self._animation_thread = threading.Thread(target=self._lateral_raise_loop, daemon=True)
        self._animation_thread.start()

    def _squat_loop(self) -> None:
        if not self.wait_for_state(timeout=3.0): return
        
        # High gains for full body
        squat_kp = np.full(NUM_MOTORS, 80.0)
        squat_kd = np.full(NUM_MOTORS, 3.0)
        
        # Legs and waist get powerful gains
        for idx in [0,1,2,3,6,7,8,9]: # legs
            squat_kp[idx] = 300.0 # hardware clamped from 500 for safety
            squat_kd[idx] = 12.0
        for idx in [4,5,10,11]:   # ankles
            squat_kp[idx] = 200.0
            squat_kd[idx] = 10.0
        for idx in [12,13,14]:    # waist
            squat_kp[idx] = 250.0
            squat_kd[idx] = 12.0
        for idx in [15,16,17,18,22,23,24,25]: # arms
            squat_kp[idx] = 150.0
            squat_kd[idx] = 8.0

        STAND_POSE = {
            "left_hip_pitch_joint": -0.40, "left_knee_joint": 0.80, "left_ankle_pitch_joint": -0.40,
            "right_hip_pitch_joint": -0.40, "right_knee_joint": 0.80, "right_ankle_pitch_joint": -0.40,
            "left_shoulder_roll_joint": 0.20, "right_shoulder_roll_joint": -0.20,
        }
        PUSHUP_READY_POSE = {
            "left_hip_pitch_joint": -1.35, "left_knee_joint": 2.25, "left_ankle_pitch_joint": -0.95,
            "right_hip_pitch_joint": -1.35, "right_knee_joint": 2.25, "right_ankle_pitch_joint": -0.95,
            "waist_pitch_joint": 0.52, "left_shoulder_pitch_joint": 1.15, "right_shoulder_pitch_joint": 1.15,
            "left_shoulder_roll_joint": 0.10, "right_shoulder_roll_joint": -0.10,
            "left_elbow_joint": -1.55, "right_elbow_joint": -1.55, "left_wrist_pitch_joint": 0.45,
            "right_wrist_pitch_joint": 0.45,
        }
        
        with self.state_lock:
            start_q = self.current_q.copy()
            rest_kp = self._target_kp.copy()
            rest_kd = self._target_kd.copy()
        
        q_stand = build_dds_pose(STAND_POSE, start_q)
        q_pushup = build_dds_pose(PUSHUP_READY_POSE, start_q)
        
        # Smoothly transition kp/kd and go to STAND
        self._publish_target_kp_kd(squat_kp, squat_kd, 2.0)
        self._publish_target(q_stand, 2.0, hold=True)
        
        START_HOLD_S = 1.2
        DESCENT_S = 5.0
        BOTTOM_HOLD_S = 0.7
        ASCENT_S = 4.0
        TOP_HOLD_S = 0.7
        cycle_len = DESCENT_S + BOTTOM_HOLD_S + ASCENT_S + TOP_HOLD_S

        start_time = time.time()
        while not self._animation_stop_event.is_set():
            t = time.time() - start_time
            if t < START_HOLD_S:
                target_q = q_stand
            else:
                phase = (t - START_HOLD_S) % cycle_len
                if phase < DESCENT_S:
                    alpha = smoothstep(phase / DESCENT_S)
                elif phase < DESCENT_S + BOTTOM_HOLD_S:
                    alpha = 1.0
                elif phase < DESCENT_S + BOTTOM_HOLD_S + ASCENT_S:
                    rise_t = phase - (DESCENT_S + BOTTOM_HOLD_S)
                    alpha = 1.0 - smoothstep(rise_t / ASCENT_S)
                else:
                    alpha = 0.0
                target_q = (1.0 - alpha) * q_stand + alpha * q_pushup
                
            with self.state_lock:
                self._target_q[:] = target_q
            time.sleep(0.004)
        
        # Restore resting stance
        self._publish_target(start_q, 2.0, hold=True)
        self._publish_target_kp_kd(rest_kp, rest_kd, 1.0)
        
    def _lateral_raise_loop(self) -> None:
        if not self.wait_for_state(timeout=3.0): return
        
        raise_kp = np.full(NUM_MOTORS, 80.0)
        raise_kd = np.full(NUM_MOTORS, 3.0)
        for idx in [0,1,2,3,6,7,8,9,4,5,10,11]: 
            raise_kp[idx] = 150.0  # Solid legs
            raise_kd[idx] = 8.0
        for idx in [15,16,17,18,22,23,24,25]:
            raise_kp[idx] = 180.0  # Strong arms
            raise_kd[idx] = 8.0
            
        STAND_POSE = {
            "left_hip_pitch_joint": -0.40, "left_knee_joint": 0.80, "left_ankle_pitch_joint": -0.40,
            "right_hip_pitch_joint": -0.40, "right_knee_joint": 0.80, "right_ankle_pitch_joint": -0.40,
            "left_shoulder_pitch_joint": 0.00, "left_shoulder_roll_joint": 0.08, "left_shoulder_yaw_joint": 0.00,
            "left_elbow_joint": 0.18, "left_wrist_roll_joint": 0.00, "left_wrist_pitch_joint": 0.00, "left_wrist_yaw_joint": 0.00,
            "right_shoulder_pitch_joint": 0.00, "right_shoulder_roll_joint": -0.08, "right_shoulder_yaw_joint": 0.00,
            "right_elbow_joint": 0.18, "right_wrist_roll_joint": 0.00, "right_wrist_pitch_joint": 0.00, "right_wrist_yaw_joint": 0.00,
        }
        LATERAL_RAISE_POSE = {
            "left_hip_pitch_joint": -0.40, "left_knee_joint": 0.80, "left_ankle_pitch_joint": -0.40,
            "right_hip_pitch_joint": -0.40, "right_knee_joint": 0.80, "right_ankle_pitch_joint": -0.40,
            "waist_pitch_joint": 0.00, "left_shoulder_pitch_joint": 0.05, "left_shoulder_roll_joint": 1.55, "left_shoulder_yaw_joint": 0.00,
            "left_elbow_joint": 0.55, "left_wrist_roll_joint": 0.00, "left_wrist_pitch_joint": 0.00, "left_wrist_yaw_joint": 0.00,
            "right_shoulder_pitch_joint": -0.05, "right_shoulder_roll_joint": -1.55, "right_shoulder_yaw_joint": 0.00,
            "right_elbow_joint": 0.55, "right_wrist_roll_joint": 0.00, "right_wrist_pitch_joint": 0.00, "right_wrist_yaw_joint": 0.00,
        }

        with self.state_lock:
            start_q = self.current_q.copy()
            rest_kp = self._target_kp.copy()
            rest_kd = self._target_kd.copy()
            
        q_stand = build_dds_pose(STAND_POSE, start_q)
        q_raise = build_dds_pose(LATERAL_RAISE_POSE, start_q)
        
        self._publish_target_kp_kd(raise_kp, raise_kd, 2.0)
        self._publish_target(q_stand, 2.0, hold=True)
        
        START_HOLD_S = 1.0
        RAISE_S = 3.5
        TOP_HOLD_S = 0.8
        LOWER_S = 3.5
        REST_HOLD_S = 0.8
        cycle_len = RAISE_S + TOP_HOLD_S + LOWER_S + REST_HOLD_S

        start_time = time.time()
        while not self._animation_stop_event.is_set():
            t = time.time() - start_time
            if t < START_HOLD_S:
                target_q = q_stand
            else:
                phase = (t - START_HOLD_S) % cycle_len
                if phase < RAISE_S:
                    alpha = smoothstep(phase / RAISE_S)
                elif phase < RAISE_S + TOP_HOLD_S:
                    alpha = 1.0
                elif phase < RAISE_S + TOP_HOLD_S + LOWER_S:
                    down_t = phase - (RAISE_S + TOP_HOLD_S)
                    alpha = 1.0 - smoothstep(down_t / LOWER_S)
                else:
                    alpha = 0.0
                target_q = (1.0 - alpha) * q_stand + alpha * q_raise
                
            with self.state_lock:
                self._target_q[:] = target_q
            time.sleep(0.004)

        # Restore resting stance
        self._publish_target(start_q, 2.0, hold=True)
        self._publish_target_kp_kd(rest_kp, rest_kd, 1.0)
"""
if "start_squat" not in text:
    # insert before shutdown
    text = text.replace("    def shutdown(self) -> None:", methods + "\n    def shutdown(self) -> None:")

# 3. Add handle_command logic
handle_squat = """        if any(fuzzy_contains_phrase(text, phrase) for phrase in ("squat", "do a squat", "start squat")):
            if self.salute_controller is not None:
                self.salute_controller.start_squat()
            else:
                self.log("[warn] squat unavailable in audio-only mode")
            return
        if any(fuzzy_contains_phrase(text, phrase) for phrase in ("lateral raise", "shoulder raise", "fly", "lateral")):
            if self.salute_controller is not None:
                self.salute_controller.start_lateral_raise()
            else:
                self.log("[warn] lateral raise unavailable in audio-only mode")
            return"""
            
if "start_squat" not in text: # It means we haven't patched commands
    pass # Wait, we check text with regex

if "start_squat" not in text and "start_lateral_raise" not in text:
    pass

text = re.sub(
    r'        if any\(fuzzy_contains_phrase\(text, phrase\) for phrase in \("sit down", "sit", "squat"\)\):\n            self._call_loco\("Sit"\)\n            return',
    r'        if any(fuzzy_contains_phrase(text, phrase) for phrase in ("sit down", "sit")):\n            self._call_loco("Sit")\n            return\n' + handle_squat,
    text
)

with open("brunken_voice.py", "w") as f:
    f.write(text)
