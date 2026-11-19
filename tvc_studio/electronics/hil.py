"""Hardware-in-the-loop transport for a real Raspberry Pi Pico.

Frames are COBS-encoded with a CRC-16 trailer, so the protocol is self-synchronising on a
noisy USB-CDC link. The application sends sensor frames and expects actuator frames back;
timing statistics are recorded exactly as in SITL so the two paths can be compared.

Honest boundary: this drives real firmware over a serial port. Whether that firmware
computes what you think it computes is not something the simulator can assert, and the UI
says so.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import HardwareError
from ..core.interfaces import Diagnostic
from .sitl import HardwareLoopStats

FRAME_SENSOR = 0x01
FRAME_ACTUATOR = 0x02
FRAME_HEARTBEAT = 0x03
FRAME_LOG = 0x04

MAX_PAYLOAD = 512


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #
def cobs_encode(data: bytes) -> bytes:
    out = bytearray()
    index = 0
    length = len(data)
    while index < length:
        block_start = index
        while index < length and data[index] != 0:
            index += 1
        block = data[block_start:index]
        out.append(len(block) + 1)
        out.extend(block)
        if index < length:
            index += 1
            if index == length:
                out.append(1)
    return bytes(out)


def cobs_decode(data: bytes) -> bytes:
    out = bytearray()
    index = 0
    while index < len(data):
        code = data[index]
        if code == 0:
            raise ValueError("invalid COBS block")
        index += 1
        end = index + code - 1
        if end > len(data):
            raise ValueError("truncated COBS block")
        out.extend(data[index:end])
        index = end
        if code < 255 and index < len(data):
            out.append(0)
    return bytes(out)


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    crc = initial
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def frame(packet_type: int, payload: bytes) -> bytes:
    body = bytes([packet_type]) + payload
    crc = crc16_ccitt(body)
    return cobs_encode(body + struct.pack("<H", crc)) + b"\x00"


def unframe(stream: bytes) -> list[tuple[int, bytes]]:
    packets: list[tuple[int, bytes]] = []
    for chunk in stream.split(b"\x00"):
        if not chunk:
            continue
        try:
            decoded = cobs_decode(chunk)
        except ValueError:
            continue
        if len(decoded) < 3:
            continue
        body, crc_bytes = decoded[:-2], decoded[-2:]
        if crc16_ccitt(body) != struct.unpack("<H", crc_bytes)[0]:
            continue
        packets.append((body[0], body[1:]))
    return packets


# --------------------------------------------------------------------------- #
# Payloads
# --------------------------------------------------------------------------- #
def sensor_payload(timestamp: float, accel, gyro, phase: int = 0) -> bytes:
    return struct.pack("<I6fB", int(timestamp * 1e6) & 0xFFFFFFFF,
                       float(accel[0]), float(accel[1]), float(accel[2]),
                       float(gyro[0]), float(gyro[1]), float(gyro[2]), int(phase))


def parse_sensor_payload(payload: bytes) -> dict:
    if len(payload) != 29:
        raise ValueError("bad sensor payload length")
    ts, ax, ay, az, gx, gy, gz, phase = struct.unpack("<I6fB", payload)
    return {"time": ts / 1e6, "accel": np.array([ax, ay, az]),
            "gyro": np.array([gx, gy, gz]), "phase": phase}


def actuator_payload(pitch: float, yaw: float, flags: int = 0) -> bytes:
    return struct.pack("<2fB", float(pitch), float(yaw), int(flags))


def parse_actuator_payload(payload: bytes) -> dict:
    if len(payload) != 9:
        raise ValueError("bad actuator payload length")
    pitch, yaw, flags = struct.unpack("<2fB", payload)
    return {"pitch": pitch, "yaw": yaw, "flags": flags}


# --------------------------------------------------------------------------- #
# Link
# --------------------------------------------------------------------------- #
@dataclass
class HILConfiguration:
    port: str = "COM3"
    baudrate: int = 921600
    timeout: float = 0.05
    sensor_rate_hz: float = 500.0
    max_consecutive_misses: int = 50
    protocol_version: int = 1

    def validate(self) -> list[str]:
        problems = []
        if not self.port:
            problems.append("no serial port selected")
        if self.baudrate < 115200:
            problems.append("baud rate below 115200 will not sustain the sensor stream")
        return problems


class HILink:
    """Serial link to real firmware. Requires ``pyserial`` (optional extra)."""

    def __init__(self, config: HILConfiguration) -> None:
        self.config = config
        self.serial = None
        self.stats = HardwareLoopStats(1.0 / max(config.sensor_rate_hz, 1e-6))
        self.diagnostics: list[Diagnostic] = []
        self._buffer = b""
        self.connected = False

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        try:
            import serial  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise HardwareError(
                "pyserial is not installed; install the 'hardware' extra to use HIL: "
                "pip install -e .[hardware]"
            ) from exc
        try:
            self.serial = serial.Serial(self.config.port, self.config.baudrate,
                                        timeout=self.config.timeout)
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise HardwareError(f"could not open {self.config.port}: {exc}") from exc
        self.connected = True

    def disconnect(self) -> None:
        if self.serial is not None:
            try:
                self.serial.close()
            finally:
                self.serial = None
        self.connected = False

    # -- transfer ----------------------------------------------------------
    def send_sensor(self, timestamp: float, accel, gyro, phase: int = 0) -> None:
        if self.serial is None:
            raise HardwareError("HIL link is not connected")
        self.serial.write(frame(FRAME_SENSOR, sensor_payload(timestamp, accel, gyro, phase)))
        self.stats.record_sent()

    def read_actuator(self) -> dict | None:
        if self.serial is None:
            raise HardwareError("HIL link is not connected")
        try:
            chunk = self.serial.read(256)
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.diagnostics.append(Diagnostic("error", "hil-read", str(exc)))
            return None
        if not chunk:
            return None
        self._buffer += chunk
        packets = unframe(self._buffer)
        if packets:
            self._buffer = b""
        for packet_type, payload in packets:
            if packet_type is FRAME_ACTUATOR:
                try:
                    data = parse_actuator_payload(payload)
                except ValueError as exc:
                    self.diagnostics.append(Diagnostic("warning", "hil-payload", str(exc)))
                    continue
                self.stats.record_received(time.perf_counter())
                return data
            if packet_type is FRAME_LOG:
                self.diagnostics.append(Diagnostic("info", "hil-log",
                                                   payload.decode("utf-8", "replace")))
            elif packet_type is FRAME_HEARTBEAT:
                self.diagnostics.append(Diagnostic("info", "hil-heartbeat", "device alive"))
        return None

    def exchange(self, timestamp: float, accel, gyro, phase: int = 0) -> dict | None:
        self.send_sensor(timestamp, accel, gyro, phase)
        return self.read_actuator()

    def report(self) -> dict:
        return {
            "connected": self.connected,
            "port": self.config.port,
            "baud": self.config.baudrate,
            "stats": self.stats.summary(),
            "diagnostics": [str(d) for d in self.diagnostics[-20:]],
        }


class LoopbackHIL:
    """In-process stand-in for the Pico link, used by tests and by the UI demo mode.

    It exercises the same framing, CRC and timing statistics as the real link so the
    protocol can be validated without hardware. The UI labels it as loopback, never as a
    real board.
    """

    def __init__(self, control_law=None) -> None:
        self.control_law = control_law
        self.stats = HardwareLoopStats(0.002)
        self.frames_tx: list[bytes] = []
        self.frames_rx: list[bytes] = []
        self._pending: bytes = b""

    def send_sensor(self, timestamp: float, accel, gyro, phase: int = 0) -> None:
        self.frames_tx.append(frame(FRAME_SENSOR, sensor_payload(timestamp, accel, gyro, phase)))
        self.stats.record_sent()

    def exchange(self, timestamp: float, accel, gyro, phase: int = 0) -> dict | None:
        self.send_sensor(timestamp, accel, gyro, phase)
        if self.control_law is None:
            pitch = float(np.clip(accel[0] * 0.001, -0.1, 0.1))
            yaw = float(np.clip(accel[1] * 0.001, -0.1, 0.1))
        else:
            pitch, yaw = self.control_law(np.asarray(accel), np.asarray(gyro), phase)
        payload = frame(FRAME_ACTUATOR, actuator_payload(pitch, yaw))
        self.frames_rx.append(payload)
        self.stats.record_received(timestamp)
        packets = unframe(payload)
        if not packets:
            return None
        return parse_actuator_payload(packets[0][1])

    def report(self) -> dict:
        return {"connected": True, "mode": "loopback", "stats": self.stats.summary()}
