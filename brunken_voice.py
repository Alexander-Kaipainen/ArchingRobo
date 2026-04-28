#!/usr/bin/env python3
"""Local voice command system for the Unitree G1.

This script keeps everything on the Linux PC:
- tiny.en wake-word loop on 2-second chunks
- base.en command capture on 4-second chunks
- local TTS with pyttsx3 / espeak-ng
- low-level right-arm salute through Unitree SDK2 topics
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    import whisper


def load_sounddevice():
    try:
        import sounddevice as sounddevice_module
    except (ModuleNotFoundError, OSError) as error:
        raise RuntimeError(
            "sounddevice/PortAudio is unavailable; install sounddevice and PortAudio or use the arecord fallback"
        ) from error
    return sounddevice_module


def load_unitree_modules() -> dict[str, Any]:
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        import unitree_sdk2py.core.channel as dds_channel
        import os

        trace_config = getattr(dds_channel, "ChannelConfigHasInterface", "")
        if "/tmp/cdds.LOG" in trace_config:
            trace_path = f"/tmp/cdds.{os.getpid()}.LOG"
            dds_channel.ChannelConfigHasInterface = trace_config.replace("/tmp/cdds.LOG", trace_path)

        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Unitree SDK dependencies are missing in this interpreter. Use the robot's working Python env for control mode."
        ) from error

    return {
        "ChannelFactoryInitialize": ChannelFactoryInitialize,
        "ChannelPublisher": ChannelPublisher,
        "ChannelSubscriber": ChannelSubscriber,
        "LocoClient": LocoClient,
        "LowState": LowState_,
        "LowCmdMsgType": LowCmd_,
        "LowCmdFactory": unitree_hg_msg_dds__LowCmd_,
        "HandCmdMsgType": HandCmd_,
        "HandCmdFactory": unitree_hg_msg_dds__HandCmd_,
        "CRC": CRC,
    }


def load_whisper_module():
    try:
        import whisper as whisper_module

        return whisper_module
    except ModuleNotFoundError:
        pass

    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    venv_site = os.path.join(os.getcwd(), ".venv", "lib", py_ver, "site-packages")
    if os.path.isdir(venv_site) and venv_site not in sys.path:
        sys.path.insert(0, venv_site)
        try:
            import whisper as whisper_module

            print(f"[audio] loaded whisper from {venv_site}")
            return whisper_module
        except ModuleNotFoundError:
            pass

    raise RuntimeError(
        "openai-whisper is required for wake-word and command transcription; install it in this interpreter or create .venv and install there"
    )


def resample_to_whisper_rate(samples: np.ndarray, capture_rate: float, whisper_rate: int) -> np.ndarray:
    if int(capture_rate) == whisper_rate:
        return samples

    try:
        from scipy.signal import resample_poly

        gcd = np.gcd(int(capture_rate), whisper_rate)
        up = whisper_rate // gcd
        down = int(capture_rate) // gcd
        return resample_poly(samples, up, down).astype(np.float32, copy=False)
    except ModuleNotFoundError:
        # Fallback path when scipy is not installed; keeps basic functionality.
        src_len = len(samples)
        if src_len <= 1:
            return samples.astype(np.float32, copy=False)
        duration = src_len / float(capture_rate)
        dst_len = max(1, int(round(duration * whisper_rate)))
        src_x = np.linspace(0.0, duration, src_len, endpoint=False)
        dst_x = np.linspace(0.0, duration, dst_len, endpoint=False)
        return np.interp(dst_x, src_x, samples).astype(np.float32, copy=False)


RIGHT_ARM_INDICES = [22, 23, 24, 25, 26, 27, 28]
NUM_MOTORS = 29  # 29 body motors, ignoring the 14 hand motors


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fuzzy_contains_phrase(text: str, phrase: str, threshold: float = 0.78) -> bool:
    text_n = normalize_text(text)
    phrase_n = normalize_text(phrase)
    if not text_n or not phrase_n:
        return False
    if phrase_n in text_n:
        return True

    text_tokens = text_n.split()
    phrase_tokens = phrase_n.split()
    if len(text_tokens) < len(phrase_tokens):
        return False

    span = len(phrase_tokens)
    for i in range(len(text_tokens) - span + 1):
        window = " ".join(text_tokens[i : i + span])
        score = difflib.SequenceMatcher(None, window, phrase_n).ratio()
        if score >= threshold:
            return True
    return False


def looks_like_repeated_hallucination(text: str) -> bool:
    tokens = normalize_text(text).split()
    if len(tokens) < 12:
        return False

    # Detect low-diversity repeated output like "bra bra bra ..."
    unique = len(set(tokens))
    diversity = unique / max(1, len(tokens))
    most_common_count = max(tokens.count(t) for t in set(tokens)) if tokens else 0
    if diversity < 0.18 or most_common_count >= int(0.45 * len(tokens)):
        return True

    # Detect short repeating n-grams.
    for n in (1, 2, 3):
        spans = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not spans:
            continue
        top = max(spans.count(s) for s in set(spans))
        if top >= max(6, int(0.35 * len(spans))):
            return True

    return False


def contains_wake_word(text: str, wake_word: str = "cheese") -> bool:
    cleaned = normalize_text(text)
    if not cleaned:
        return False
    if wake_word in cleaned.split():
        return True
    if wake_word in cleaned:
        return True
    for token in cleaned.split():
        if difflib.SequenceMatcher(None, token, wake_word).ratio() >= 0.82:
            return True
    return False


_tts_engine = None

def say(text: str) -> None:
    global _tts_engine
    try:
        import pyttsx3
    except ModuleNotFoundError as error:
        raise RuntimeError("pyttsx3 is required for TTS; install it with `pip install pyttsx3`") from error

    if _tts_engine is None:
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty("rate", 180)
        _tts_engine.setProperty("volume", 1.0)
        
    _tts_engine.say(text)
    _tts_engine.runAndWait()


@dataclass
class AudioConfig:
    whisper_rate: int = 16000
    capture_rate: int = 48000
    wake_seconds: float = 2.0
    command_seconds: float = 4.0
    arecord_device: str = "auto"


def list_input_devices() -> None:
    print("[audio] input devices:")
    try:
        sd = load_sounddevice()
        devices = sd.query_devices()
        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                print(
                    f"  [{index}] {device['name']} "
                    f"(inputs={device['max_input_channels']}, default_sr={device['default_samplerate']})"
                )
        return
    except RuntimeError:
        pass

    # Fallback for machines without PortAudio bindings.
    proc = subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=False)
    text = (proc.stdout + "\n" + proc.stderr).strip()
    if text:
        print(text)
        print("[audio] arecord fallback defaults try: default -> pulse -> pipewire -> plughw")
    else:
        print("  no input devices found")


def resolve_input_device(requested_device: Optional[str]) -> tuple[int, float]:
    sd = load_sounddevice()
    devices = sd.query_devices()
    input_devices = [
        (index, device)
        for index, device in enumerate(devices)
        if device.get("max_input_channels", 0) > 0
    ]
    if not input_devices:
        raise RuntimeError("No input-capable audio devices were found")

    if requested_device is None or requested_device == "auto":
        default_input = sd.default.device[0] if sd.default.device else None
        if default_input is not None and default_input >= 0:
            device = devices[default_input]
            if device.get("max_input_channels", 0) > 0:
                return default_input, float(device.get("default_samplerate") or 48000.0)

        index, device = input_devices[0]
        return index, float(device.get("default_samplerate") or 48000.0)

    try:
        index = int(requested_device)
        device = devices[index]
    except ValueError:
        matches = [
            (index, device)
            for index, device in input_devices
            if requested_device.lower() in device["name"].lower()
        ]
        if not matches:
            raise RuntimeError(f'No microphone matched "{requested_device}"')
        index, device = matches[0]

    if device.get("max_input_channels", 0) <= 0:
        raise RuntimeError(f'Device "{device["name"]}" is not an input device')

    return index, float(device.get("default_samplerate") or 48000.0)


class LowLevelSaluteController:
    def __init__(self, network_interface: str, domain_id: int):
        self.network_interface = network_interface
        self.domain_id = domain_id
        sdk = load_unitree_modules()
        self.LowState = sdk["LowState"]
        self.ChannelSubscriber = sdk["ChannelSubscriber"]
        self.ChannelPublisher = sdk["ChannelPublisher"]

        sdk["ChannelFactoryInitialize"](domain_id, network_interface)

        self.state_lock = threading.Lock()
        self.current_q = np.zeros(NUM_MOTORS, dtype=float)
        self.current_dq = np.zeros(NUM_MOTORS, dtype=float)
        self.latest_state_time = 0.0
        self._state_ready = threading.Event()

        self.crc = sdk["CRC"]()
        self.msg = sdk["LowCmdFactory"]()
        self.msg.mode_pr = 0

        self.lowstate_subscriber = self.ChannelSubscriber("rt/lowstate", self.LowState)
        self.lowstate_subscriber.Init(self._on_lowstate, 10)

        self.lowcmd_publisher = self.ChannelPublisher("rt/lowcmd", sdk["LowCmdMsgType"])
        self.lowcmd_publisher.Init()
        
        self._animation_thread = None
        self._animation_stop_event = threading.Event()
        self._running = True
        self._target_q = np.zeros(NUM_MOTORS, dtype=float)
        
        self.wait_for_state(timeout=2.0)
        with self.state_lock:
            self._target_q[:] = self.current_q[:]
            self._prime_message()

        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publish_thread.start()

    def _on_lowstate(self, msg) -> None:
        with self.state_lock:
            if not self._state_ready.is_set():
                print(f"[salute_controller] First lowstate received! Body motors: {len(msg.motor_state)}")
            limit = min(NUM_MOTORS, len(msg.motor_state))
            for idx in range(limit):
                self.current_q[idx] = msg.motor_state[idx].q
                self.current_dq[idx] = msg.motor_state[idx].dq
            self.msg.mode_machine = msg.mode_machine
            self.latest_state_time = time.time()
            self._state_ready.set()

    def wait_for_state(self, timeout: float = 10.0) -> bool:
        return self._state_ready.wait(timeout=timeout)

    def _prime_message(self) -> None:
        for idx in range(NUM_MOTORS):
            self.msg.motor_cmd[idx].mode = 0x01
            self.msg.motor_cmd[idx].kp = 20.0
            self.msg.motor_cmd[idx].kd = 3.0
            self.msg.motor_cmd[idx].q = float(self._target_q[idx])
            self.msg.motor_cmd[idx].dq = 0.0
            self.msg.motor_cmd[idx].tau = 0.0

    def _publish_loop(self) -> None:
        while self._running:
            with self.state_lock:
                q_cmd = self._target_q.copy()
            self._write_full_body_command(q_cmd)
            time.sleep(0.004)

    def _publish_target(self, target_q: np.ndarray, duration: float, hold: bool = False) -> None:
        duration = max(0.0, duration)
        steps = max(1, int(duration * 250))

        with self.state_lock:
            start_q = self._target_q.copy()

        for step in range(steps):
            alpha = (step + 1) / steps
            with self.state_lock:
                self._target_q[:] = (1.0 - alpha) * start_q + alpha * target_q
            time.sleep(0.004)

        if hold:
            with self.state_lock:
                self._target_q[:] = target_q

    def _write_full_body_command(self, q_cmd: np.ndarray) -> None:
        for idx in range(NUM_MOTORS):
            self.msg.motor_cmd[idx].mode = 0x01
            self.msg.motor_cmd[idx].q = float(q_cmd[idx])
            self.msg.motor_cmd[idx].dq = 0.0
            self.msg.motor_cmd[idx].tau = 0.0

            if idx in RIGHT_ARM_INDICES:
                if idx == 25:
                    self.msg.motor_cmd[idx].kp = 140.0
                    self.msg.motor_cmd[idx].kd = 5.0
                elif idx >= 26:
                    self.msg.motor_cmd[idx].kp = 70.0
                    self.msg.motor_cmd[idx].kd = 3.0
                else:
                    self.msg.motor_cmd[idx].kp = 120.0
                    self.msg.motor_cmd[idx].kd = 4.0
            else:
                self.msg.motor_cmd[idx].kp = 20.0
                self.msg.motor_cmd[idx].kd = 3.0

        self.msg.crc = self.crc.Crc(self.msg)
        self.lowcmd_publisher.Write(self.msg)

    def salute(self) -> None:
        if not self.wait_for_state(timeout=3.0):
            print("[salute] lowstate not ready; skipping salute")
            return

        with self.state_lock:
            self._target_q[:] = self.current_q[:]
            rest_q = self.current_q.copy()

        salute_q = rest_q.copy()
        # Proper Right Arm Salute (Military Style)
        # Ultra-conservative clearance: -1.0 elbow and 0.0 yaw.
        salute_q[22] = -1.00  # Shoulder Pitch
        salute_q[23] = -1.50  # Shoulder Roll: Move arm wide
        salute_q[24] = 0.00   # Shoulder Yaw: No inward rotation
        salute_q[25] = -0.45  # ELBOW: Conservative 90-degree bend
        salute_q[26] = 0.00
        salute_q[27] = -0.40  # Wrist Pitch
        salute_q[28] = 0.00

        print("[state] executing salute (ultra-wide clearance)")
        try:
            self._publish_target(salute_q, duration=0.8, hold=True)
            time.sleep(1.5)
        finally:
            self._publish_target(rest_q, duration=1.2, hold=True)

    def shutdown(self) -> None:
        self.stop_animations()
        self._running = False
        if hasattr(self, 'publish_thread'):
            self.publish_thread.join(timeout=0.1)

    def start_bicep_curl(self) -> None:
        """Start a continuous bicep curl animation in a background thread."""
        self.stop_animations()
        self._animation_stop_event.clear()
        self._animation_thread = threading.Thread(target=self._bicep_curl_loop, daemon=True)
        self._animation_thread.start()

    def stop_animations(self) -> None:
        """Stop any active background animations (like bicep curls)."""
        self._animation_stop_event.set()
        if self._animation_thread is not None:
            self._animation_thread.join(timeout=0.1)
            self._animation_thread = None

    def _bicep_curl_loop(self) -> None:
        if not self.wait_for_state(timeout=3.0):
            return

        with self.state_lock:
            self._target_q[:] = self.current_q[:]
            rest_q = self.current_q.copy()

        # 1. Move to Prep Posture (Shoulders forward and out + Elbows at 'Bottom')
        prep_q = rest_q.copy()
        prep_q[15] = -0.4  # Left Shoulder Pitch Forward
        prep_q[16] = 0.5   # Left Shoulder Roll Out
        prep_q[22] = -0.4  # Right Shoulder Pitch Forward
        prep_q[23] = -0.5  # Right Shoulder Roll Out
        prep_q[18] = 0.0   # Left Elbow Extended
        prep_q[25] = 0.0   # Right Elbow Extended
        
        self._publish_target(prep_q, duration=2.5, hold=True)
        time.sleep(0.5)

        # 2. Continuous Curl Loop
        CURL_PERIOD = 5.0
        ELBOW_EXTENDED = 0.0
        ELBOW_FLEXED = 1.4
        
        start_time = time.time()
        while not self._animation_stop_event.is_set():
            elapsed = time.time() - start_time
            omega = (2.0 * np.pi) / CURL_PERIOD
            phase = omega * elapsed
            
            # Sine wave between extended and flexed
            amp = (ELBOW_FLEXED - ELBOW_EXTENDED) / 2.0
            mid = (ELBOW_FLEXED + ELBOW_EXTENDED) / 2.0
            val = mid - amp * np.cos(phase)
            
            with self.state_lock:
                self._target_q[18] = val  # Left Elbow
                self._target_q[25] = val  # Right Elbow
            time.sleep(0.01)

        # 3. Return to rest
        self._publish_target(rest_q, duration=1.2, hold=True)

    def move_to_dual_arm_pose(self, q14_target: np.ndarray, duration: float = 1.2, hold: bool = True) -> None:
        if not self.wait_for_state(timeout=2.0):
            print("[arms] lowstate not ready; cannot move to target pose")
            return

        with self.state_lock:
            full_target = self.current_q.copy()

        if q14_target.shape[0] != 14:
            print("[arms] invalid pose length; expected 14")
            return

        for i, body_idx in enumerate(range(15, 29)):
            full_target[body_idx] = float(q14_target[i])

        self._publish_target(full_target, duration=duration, hold=hold)


class DexHandController:
    LEFT_TOPIC = "rt/dex3/left/cmd"
    RIGHT_TOPIC = "rt/dex3/right/cmd"

    def __init__(self):
        sdk = load_unitree_modules()
        self.ChannelPublisher = sdk["ChannelPublisher"]
        self.HandCmdMsgType = sdk["HandCmdMsgType"]
        self.HandCmdFactory = sdk["HandCmdFactory"]

        self.left_pub = self.ChannelPublisher(self.LEFT_TOPIC, self.HandCmdMsgType)
        self.left_pub.Init()
        self.right_pub = self.ChannelPublisher(self.RIGHT_TOPIC, self.HandCmdMsgType)
        self.right_pub.Init()

        self.left_msg = self.HandCmdFactory()
        self.right_msg = self.HandCmdFactory()
        self._init_msgs()

    @staticmethod
    def _ris_mode(motor_id: int, status: int = 0x01, timeout: int = 0) -> int:
        mode = 0
        mode |= (motor_id & 0x0F)
        mode |= (status & 0x07) << 4
        mode |= (timeout & 0x01) << 7
        return mode

    def _init_msgs(self) -> None:
        for i in range(7):
            self.left_msg.motor_cmd[i].mode = self._ris_mode(i, status=0x01)
            self.right_msg.motor_cmd[i].mode = self._ris_mode(i, status=0x01)

            self.left_msg.motor_cmd[i].q = 0.0
            self.right_msg.motor_cmd[i].q = 0.0

            self.left_msg.motor_cmd[i].dq = 0.0
            self.right_msg.motor_cmd[i].dq = 0.0

            self.left_msg.motor_cmd[i].tau = 0.0
            self.right_msg.motor_cmd[i].tau = 0.0

            self.left_msg.motor_cmd[i].kp = 30.0
            self.right_msg.motor_cmd[i].kp = 30.0
            self.left_msg.motor_cmd[i].kd = 1.5
            self.right_msg.motor_cmd[i].kd = 1.5

    def set_hands(self, left_q: np.ndarray, right_q: np.ndarray, repeat: int = 8, dt: float = 0.03) -> None:
        for i in range(7):
            self.left_msg.motor_cmd[i].q = float(left_q[i])
            self.right_msg.motor_cmd[i].q = float(right_q[i])

        for _ in range(max(1, repeat)):
            self.left_pub.Write(self.left_msg)
            self.right_pub.Write(self.right_msg)
            time.sleep(dt)

    def open_hands(self) -> None:
        # For these grippers, 0.0 is often 'neutral' or 'mid-way'. 
        # Negative values usually open them further in MuJoCo.
        q_open = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1], dtype=float)
        self.set_hands(q_open, q_open)

    def close_hands_for_bar(self) -> None:
        # Increase these values if they aren't closing enough.
        # Positive values (flexion) should close the fingers.
        q_close_left = np.array([1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3], dtype=float)
        q_close_right = np.array([1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3], dtype=float)
        self.set_hands(q_close_left, q_close_right)


class VoiceCommandSystem:
    def __init__(
        self,
        interface: str,
        domain_id: int,
        wake_word: str = "cheese",
        wake_model_name: str = "tiny",
        command_model_name: str = "base",
        input_device: str = "auto",
        arecord_device: str = "auto",
        language: str = "en",
        mic_debug: bool = False,
        mic_debug_save: bool = False,
        switch_seconds: float = 20.0,
        audio_only: bool = False,
    ):
        self.audio = AudioConfig()
        self.wake_word = wake_word
        self.running = True
        self.console_lock = threading.Lock()
        self.language = language.lower().strip() if language else "auto"
        self.mic_debug = mic_debug
        self.mic_debug_save = mic_debug_save
        self.audio_backend = "sounddevice"
        self.audio_only = audio_only
        self.sd = None
        self._quiet_chunks = 0
        self._nonsense_chunks = 0
        self._arecord_candidates: list[str] = []
        self._arecord_index = 0
        self._chunks_since_switch = 0
        self._force_switch_chunks = max(1, int(round(max(1.0, switch_seconds) / self.audio.wake_seconds)))
        self._mic_debug_dir = pathlib.Path("/tmp")

        try:
            self.sd = load_sounddevice()
        except RuntimeError:
            self.audio_backend = "arecord"
            if not shutil.which("arecord"):
                raise RuntimeError(
                    "No audio backend found. Install sounddevice+PortAudio or make sure `arecord` is available."
                )

        if self.audio_backend == "sounddevice":
            self.input_device_index, capture_rate = resolve_input_device(input_device)
            self.audio.capture_rate = capture_rate
            self.sd.default.device = (self.input_device_index, None)
            self.log(
                f"[audio] using input device {self.input_device_index}: "
                f"{self.sd.query_devices(self.input_device_index)['name']}"
            )
        else:
            self.input_device_index = -1
            self.audio.capture_rate = 48000
            self._arecord_candidates = self._discover_arecord_devices(arecord_device)
            self._arecord_index = 0
            self.audio.arecord_device = self._arecord_candidates[self._arecord_index]
            self.log(f"[audio] using arecord fallback backend on {self.audio.arecord_device}")
            if self.mic_debug:
                self.log(f"[audio][debug] arecord candidates: {self._arecord_candidates}")

        self.whisper = load_whisper_module()
        self.wake_model = self.whisper.load_model(wake_model_name)
        self.command_model = self.whisper.load_model(command_model_name)

        self.salute_controller = None
        self.hand_controller = None
        self.loco = None
        if not self.audio_only:
            self.salute_controller = LowLevelSaluteController(interface, domain_id)
            try:
                self.hand_controller = DexHandController()
            except Exception as error:
                self.log(f"[warn] hand controller unavailable: {error}")
                self.hand_controller = None
            self.loco = self._init_loco_client()
        else:
            self.log("[state] audio-only mode: skipping DDS and robot controllers")

    def _init_loco_client(self) -> Optional["LocoClient"]:
        try:
            sdk = load_unitree_modules()
            client = sdk["LocoClient"]()
            client.Init()
            client.Start()
            return client
        except Exception as error:
            self.log(f"[warn] loco client unavailable: {error}")
            return None

    def log(self, message: str) -> None:
        with self.console_lock:
            print(message, flush=True)

    def _discover_arecord_devices(self, requested_device: str) -> list[str]:
        requested = (requested_device or "auto").strip()
        rotate_endpoints = {"auto", "default", "pulse", "pipewire"}
        if requested not in rotate_endpoints:
            return [requested]

        proc = subprocess.run(["arecord", "-L"], capture_output=True, text=True, check=False)
        devices: list[str] = []
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                name = line.strip()
                if not name:
                    continue
                if name.startswith(("default", "pulse", "pipewire", "plughw:")):
                    devices.append(name)

        preferred = [
            requested if requested in {"default", "pulse", "pipewire"} else "default",
            "default",
            "pulse",
            "pipewire",
            "plughw:CARD=PCH,DEV=2",
            "plughw:CARD=PCH,DEV=0",
        ]
        ordered: list[str] = []

        def add_unique(name: str) -> None:
            if name and name not in ordered:
                ordered.append(name)

        for name in preferred:
            if name in devices:
                add_unique(name)
        for name in devices:
            add_unique(name)

        if not ordered:
            ordered = ["default", "pulse", "pipewire", "plughw:CARD=PCH,DEV=2", "plughw:CARD=PCH,DEV=0"]
        return ordered

    def _switch_arecord_device(self, reason: Optional[str] = None) -> None:
        if len(self._arecord_candidates) <= 1:
            return
        current = self.audio.arecord_device
        for _ in range(len(self._arecord_candidates)):
            self._arecord_index = (self._arecord_index + 1) % len(self._arecord_candidates)
            candidate = self._arecord_candidates[self._arecord_index]
            if candidate != current:
                self.audio.arecord_device = candidate
                self._chunks_since_switch = 0
                if reason:
                    self.log(f"[audio] switched arecord device to {self.audio.arecord_device} ({reason})")
                else:
                    self.log(f"[audio] switched arecord device to {self.audio.arecord_device}")
                return

    def _save_wake_chunk_debug(self, chunk: np.ndarray, transcript: str) -> None:
        if not self.mic_debug_save or len(chunk) == 0:
            return
        source = self.audio.arecord_device if self.audio_backend == "arecord" else str(self.input_device_index)
        safe_source = re.sub(r"[^a-zA-Z0-9_.-]", "_", source)
        stamp_ms = int(time.time() * 1000)
        snippet = normalize_text(transcript)[:24] or "silence"
        safe_snippet = re.sub(r"[^a-zA-Z0-9_.-]", "_", snippet)
        wav_path = self._mic_debug_dir / f"cheese_wake_{stamp_ms}_{safe_source}_{safe_snippet}.wav"
        pcm = np.clip(chunk, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16, copy=False)
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(self.audio.capture_rate))
            wav_file.writeframes(pcm16.tobytes())
        self.log(f"[audio][debug] saved wake chunk: {wav_path}")

    def record_audio(self, seconds: float) -> np.ndarray:
        if self.audio_backend == "sounddevice":
            audio = self.sd.rec(
                int(seconds * self.audio.capture_rate),
                samplerate=self.audio.capture_rate,
                channels=1,
                dtype="float32",
            )
            self.sd.wait()
            samples = audio[:, 0]
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32, copy=False)
            return samples

        wav_path = "/tmp/cheese_capture.wav"
        last_error = None
        attempts = [
            (self.audio.arecord_device, int(self.audio.capture_rate), 1),
            (self.audio.arecord_device, int(self.audio.capture_rate), 2),
            (self.audio.arecord_device, 16000, 1),
            (self.audio.arecord_device, 48000, 2),
        ]
        for dev, rate, chans in attempts:
            cmd = [
                "arecord",
                "-q",
                "-D",
                dev,
                "-d",
                str(max(1, int(round(seconds)))),
                "-f",
                "S16_LE",
                "-r",
                str(rate),
                "-c",
                str(chans),
                wav_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                self.audio.capture_rate = float(rate)
                break
            last_error = (proc.stderr or proc.stdout).strip()
        else:
            raise RuntimeError(f"arecord failed: {last_error}")

        with wave.open(wav_path, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            channels = wav_file.getnchannels()
            raw = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                raw = raw.reshape(-1, channels)
                samples = raw.mean(axis=1)
            else:
                samples = raw
        try:
            os.remove(wav_path)
        except OSError:
            pass
        return samples

    def prepare_audio_for_whisper(self, samples: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        if peak > 1e-6 and peak < 0.20:
            gain = min(20.0, 0.50 / peak)
            samples = np.clip(samples * gain, -1.0, 1.0)
        return resample_to_whisper_rate(samples, self.audio.capture_rate, self.audio.whisper_rate)

    def transcribe(self, model: whisper.Whisper, samples: np.ndarray) -> str:
        prepared = self.prepare_audio_for_whisper(samples)
        transcribe_kwargs = {
            "fp16": False,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "verbose": False,
            "no_speech_threshold": 0.60,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
        }
        if self.language != "auto":
            transcribe_kwargs["language"] = self.language
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = model.transcribe(
                prepared,
                **transcribe_kwargs,
            )
        return normalize_text(result.get("text", ""))

    def wake_loop(self) -> None:
        self.log("[state] waiting")
        while self.running:
            chunk = self.record_audio(self.audio.wake_seconds)
            if self.audio_backend == "arecord":
                self._chunks_since_switch += 1
            level = float(np.mean(np.abs(chunk))) if len(chunk) else 0.0
            peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
            if self.mic_debug:
                src = self.audio.arecord_device if self.audio_backend == "arecord" else str(self.input_device_index)
                self.log(f"[audio][debug] source={src} mean={level:.5f} peak={peak:.5f}")
            if level < 0.005:
                self._quiet_chunks += 1
                self.log(f"[audio] low level mean={level:.4f} peak={peak:.4f}")
            else:
                self._quiet_chunks = 0

            if self.audio_backend == "arecord" and self._quiet_chunks >= 3:
                self._quiet_chunks = 0
                self._switch_arecord_device(reason="low-level")

            text = self.transcribe(self.wake_model, chunk)
            if self.mic_debug and text:
                self.log(f"[audio][debug] raw wake transcript: {text}")
            self._save_wake_chunk_debug(chunk, text)
            if looks_like_repeated_hallucination(text):
                self._nonsense_chunks += 1
                self.log("[audio] repeated/gibberish transcript detected; treating as silence")
                text = ""
            else:
                self._nonsense_chunks = 0

            if self.audio_backend == "arecord" and self._nonsense_chunks >= 2:
                self._nonsense_chunks = 0
                self._switch_arecord_device(reason="gibberish")

            if self.audio_backend == "arecord" and self._chunks_since_switch >= self._force_switch_chunks:
                self._switch_arecord_device(reason=f"timer>{self._force_switch_chunks * self.audio.wake_seconds:.0f}s")

            self.log(f"[wake] {text or '<silence>'}")

            if not contains_wake_word(text, self.wake_word):
                # Safety check: if they just shout 'stop' without the wake word, still handle it!
                if any(fuzzy_contains_phrase(text, p) for p in ("stop", "halt", "freeze")):
                    self.log("[state] emergency stop detected (no wake word)")
                    self.handle_command("stop")
                continue

            self.log("[state] wake detected")
            
            # Start confirmation voice and motion in the background
            threading.Thread(target=say, args=("Yes sir",), daemon=True).start()
            if self.salute_controller is not None:
                threading.Thread(target=self.salute_controller.salute, daemon=True).start()

            # IMMEDIATELY start listening for the command
            self.log("[state] listening for command...")
            command_audio = self.record_audio(self.audio.command_seconds)
            command_text = self.transcribe(self.command_model, command_audio)
            self.log(f"[state] recognized command: \"{command_text or '<silence>'}\"")

            self.handle_command(command_text)
            self.log("[state] waiting")

    def _call_loco(self, method_name: str, *args) -> bool:
        if self.loco is None:
            self.log(f"[warn] loco unavailable: {method_name}")
            return False
        method = getattr(self.loco, method_name, None)
        if method is None:
            self.log(f"[warn] loco method missing: {method_name}")
            return False
        try:
            method(*args)
            return True
        except Exception as error:
            self.log(f"[warn] loco command failed ({method_name}): {error}")
            return False

    def handle_command(self, command_text: str) -> None:
        text = normalize_text(command_text)
        if not text:
            self.log("[state] no command recognized")
            return

        self.log("[state] executing")

        if any(fuzzy_contains_phrase(text, phrase) for phrase in ("walk forward", "go forward", "move forward", "forward")):
            self._call_loco("Move", 0.25, 0.0, 0.0, True)
            return
        if any(fuzzy_contains_phrase(text, phrase) for phrase in ("walk back", "walk backward", "go back", "move back", "backward", "reverse")):
            self._call_loco("Move", -0.25, 0.0, 0.0, True)
            return
        if any(fuzzy_contains_phrase(text, phrase) for phrase in ("turn left", "rotate left", "yaw left")):
            self._call_loco("Move", 0.0, 0.0, 0.6, True)
            return
        if any(fuzzy_contains_phrase(text, phrase) for phrase in ("turn right", "rotate right", "yaw right")):
            self._call_loco("Move", 0.0, 0.0, -0.6, True)
            return
        if any(fuzzy_contains_phrase(text, phrase) for phrase in ("stop", "halt", "freeze")):
            self._call_loco("StopMove")
            if self.salute_controller is not None:
                self.salute_controller.stop_animations()
            return
        if any(fuzzy_contains_phrase(text, phrase) for phrase in ("stand up", "stand", "get up", "high stand")):
            self._call_loco("HighStand")
            return
        if any(fuzzy_contains_phrase(text, phrase) for phrase in ("sit down", "sit", "squat")):
            self._call_loco("Sit")
            return
        if any(fuzzy_contains_phrase(text, phrase) for phrase in ("curl", "bicep", "flex", "start exercise")):
            if self.salute_controller is not None:
                self.salute_controller.start_bicep_curl()
            else:
                self.log("[warn] bicep curl unavailable in audio-only mode")
            return
        if any(fuzzy_contains_phrase(text, phrase) for phrase in ("salute", "right salute")):
            if self.salute_controller is not None:
                self.salute_controller.salute()
            else:
                self.log("[warn] salute unavailable in audio-only mode")
            return

        self.log(f"[state] unknown command: {text}")


    def shutdown(self, skip_loco_stop: bool = False) -> None:
        self.running = False
        try:
            if self.loco is not None and not skip_loco_stop:
                self.loco.StopMove()
        except BaseException:
            pass
        try:
            if self.salute_controller is not None:
                self.salute_controller.shutdown()
        except BaseException:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Local wake-word and command system for Unitree G1")
    parser.add_argument("--interface", default="enp129s0", help="Network interface for the robot DDS link")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id")
    parser.add_argument("--input-device", default="auto", help='Input device index or name fragment; use "auto" for the default microphone')
    parser.add_argument("--arecord-device", default="auto", help="ALSA device string when using arecord fallback backend; use auto to try available devices")
    parser.add_argument("--list-audio-devices", action="store_true", help="Print available input audio devices and exit")
    parser.add_argument("--wake-word", default="cheese", help="The word used to wake the robot (default: cheese)")
    parser.add_argument("--wake-model", default="tiny", help="Whisper model for wake-word detection")
    parser.add_argument("--command-model", default="base", help="Whisper model for command transcription")
    parser.add_argument("--language", default="en", help="Whisper language code for transcription (e.g. sv, en, auto)")
    parser.add_argument("--mic-debug", action="store_true", help="Print per-chunk audio source, levels, and raw wake transcript before filtering")
    parser.add_argument("--mic-debug-save", action="store_true", help="Save each wake chunk to /tmp for listening tests (can create many files)")
    parser.add_argument("--switch-seconds", type=float, default=20.0, help="Force arecord source switch interval in seconds while waiting for wake word")
    parser.add_argument("--audio-only", action="store_true", help="Run mic wake pipeline without DDS/robot init (for microphone debugging)")
    args = parser.parse_args()

    if args.list_audio_devices:
        list_input_devices()
        return

    print("[state] starting")
    print(f"[state] DDS init on {args.interface}, domain {args.domain}")
    try:
        system = VoiceCommandSystem(
            interface=args.interface,
            domain_id=args.domain,
            wake_word=args.wake_word,
            wake_model_name=args.wake_model,
            command_model_name=args.command_model,
            input_device=args.input_device,
            arecord_device=args.arecord_device,
            language=args.language,
            mic_debug=args.mic_debug,
            mic_debug_save=args.mic_debug_save,
            switch_seconds=args.switch_seconds,
            audio_only=args.audio_only,
        )
    except Exception as error:
        print(f"[error] startup failed: {error}")
        print(
            "[hint] verify the robot NIC exists and is linked (NO-CARRIER/state DOWN will break DDS), "
            "then retry with --interface set to that NIC"
        )
        print("[hint] for microphone troubleshooting without robot networking, add --audio-only")
        return

    if system.salute_controller is not None:
        if not system.salute_controller.wait_for_state(timeout=10.0):
            print("[warn] lowstate did not arrive in time; salute may be skipped until state is ready")
        else:
            system.salute_controller._prime_message()

    interrupted = False
    try:
        system.wake_loop()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[state] stopping")
    finally:
        system.shutdown(skip_loco_stop=interrupted)
        print("[state] stopped")


if __name__ == "__main__":
    main()