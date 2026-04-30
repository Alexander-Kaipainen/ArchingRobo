import re

with open("brunken_voice.py", "r") as f:
    text = f.read()

# Replace Kp/Kd hardcoding in __init__
text = text.replace(
    "self._target_q = np.zeros(NUM_MOTORS, dtype=float)",
    "self._target_q = np.zeros(NUM_MOTORS, dtype=float)\n        self._target_kp = np.full(NUM_MOTORS, 20.0, dtype=float)\n        self._target_kd = np.full(NUM_MOTORS, 3.0, dtype=float)"
)

# _prime_message
text = text.replace(
    "self.msg.motor_cmd[idx].kp = 20.0\n            self.msg.motor_cmd[idx].kd = 3.0",
    "self.msg.motor_cmd[idx].kp = float(self._target_kp[idx])\n            self.msg.motor_cmd[idx].kd = float(self._target_kd[idx])"
)

# _publish_loop
text = text.replace(
    "            with self.state_lock:\n                q_cmd = self._target_q.copy()\n            self._write_full_body_command(q_cmd)",
    "            with self.state_lock:\n                q_cmd = self._target_q.copy()\n                kp_cmd = self._target_kp.copy()\n                kd_cmd = self._target_kd.copy()\n            self._write_full_body_command(q_cmd, kp_cmd, kd_cmd)"
)

# _write_full_body_command signature and kp/kd mapping
repl_write = """    def _write_full_body_command(self, q_cmd: np.ndarray, kp_cmd: np.ndarray, kd_cmd: np.ndarray) -> None:
        for idx in range(NUM_MOTORS):
            self.msg.motor_cmd[idx].mode = 0x01
            self.msg.motor_cmd[idx].q = float(q_cmd[idx])
            self.msg.motor_cmd[idx].dq = 0.0
            self.msg.motor_cmd[idx].tau = 0.0
            
            # Apply dynamic gains
            self.msg.motor_cmd[idx].kp = float(kp_cmd[idx])
            self.msg.motor_cmd[idx].kd = float(kd_cmd[idx])"""

text = re.sub(
    r"    def _write_full_body_command\(self, q_cmd: np.ndarray\) -> None:.*?            else:\n                self.msg.motor_cmd\[idx\].kp = 20.0\n                self.msg.motor_cmd\[idx\].kd = 3.0",
    repl_write,
    text,
    flags=re.DOTALL
)

with open("brunken_voice.py", "w") as f:
    f.write(text)

