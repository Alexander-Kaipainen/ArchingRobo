import re

with open("brunken_voice.py", "r") as f:
    text = f.read()

# Fix __init__ values for kp and kd
correct_init = """        self._target_q = np.zeros(NUM_MOTORS, dtype=float)
        self._target_kp = np.full(NUM_MOTORS, 20.0, dtype=float)
        self._target_kd = np.full(NUM_MOTORS, 3.0, dtype=float)
        # Restore right arm high gains
        for idx in range(NUM_MOTORS):
            if idx in RIGHT_ARM_INDICES:
                if idx == 25:
                    self._target_kp[idx] = 140.0
                    self._target_kd[idx] = 5.0
                elif idx >= 26:
                    self._target_kp[idx] = 70.0
                    self._target_kd[idx] = 3.0
                else:
                    self._target_kp[idx] = 120.0
                    self._target_kd[idx] = 4.0"""

text = text.replace(
    "self._target_q = np.zeros(NUM_MOTORS, dtype=float)\n        self._target_kp = np.full(NUM_MOTORS, 20.0, dtype=float)\n        self._target_kd = np.full(NUM_MOTORS, 3.0, dtype=float)",
    correct_init
)

with open("brunken_voice.py", "w") as f:
    f.write(text)
