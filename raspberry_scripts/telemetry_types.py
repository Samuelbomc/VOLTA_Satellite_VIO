from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ImuPacket:
    recv_time: float
    timestamp_ms: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

    @staticmethod
    def from_tuple(recv_time: float, data: Tuple[int, float, float, float, float, float, float]) -> "ImuPacket":
        timestamp_ms, ax, ay, az, gx, gy, gz = data
        return ImuPacket(
            recv_time=recv_time,
            timestamp_ms=timestamp_ms,
            ax=ax,
            ay=ay,
            az=az,
            gx=gx,
            gy=gy,
            gz=gz,
        )


@dataclass(frozen=True)
class CameraFrame:
    frame_time: float
    data: bytes
