import math
import random

import carb
import omni
import omni.timeline
from pxr import Gf


class CubeSatRecoveryController:

    def __init__(
        self,
        cubesat_path="/World/CubeSat",
        drag_body_path="/World/Parachute_Dummy",
        ground_altitude_m=0.0,
    ):
        self.stage = omni.usd.get_context().get_stage()

        self.cubesat_prim = self.stage.GetPrimAtPath(cubesat_path)
        self.drag_prim = self.stage.GetPrimAtPath(drag_body_path)

        if not self.cubesat_prim.IsValid():
            raise RuntimeError("Invalid CubeSat prim")
        if not self.drag_prim.IsValid():
            raise RuntimeError("Invalid parachute prim")

        # --- Environment ---
        self.ground_altitude_m = ground_altitude_m
        self.main_deploy_alt = 450.0

        # --- Descent speeds (parachute effect) ---
        self.descent_drogue = -22.0
        self.descent_main = -5.0
        self.vertical_speed = self.descent_drogue
        self.descent_response = 1.8

        # --- Pendulum motion ---
        self.phase = 0.0
        self.swing_freq = 1.2
        self.swing_amp = 8.0
        self.swing_dir = Gf.Vec3f(1, 0, 0)
        self.dir_timer = 2.0

        # --- Wind / turbulence (kinematic approximation) ---
        self.wind = Gf.Vec3f(0, 0, 0)
        self.wind_target = Gf.Vec3f(0, 0, 0)
        self.wind_timer = 0.0
        self.wind_response = 0.9

        self.turbulence = Gf.Vec3f(0, 0, 0)
        self.turbulence_response = 4.0

        self.lateral_velocity = Gf.Vec3f(0, 0, 0)
        self.lateral_response = 2.5

        # --- Rotation (Z spin) ---
        self.spin_rate = 1
        self.spin_target = 0.0
        self.spin_timer = 0.0
        self.spin_damping = 0.985
        self.max_spin = 2.0  # rad/s

        self.current_stage = "drogue"

        # --- APIs ---
        self.vel_attr = self.cubesat_prim.GetAttribute("physics:velocity")
        self.ang_vel_attr = self.cubesat_prim.GetAttribute("physics:angularVelocity")

        self._sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
            self._on_step
        )

        carb.log_info("Kinematic recovery controller started")

    # ------------------------------------------------------------

    def stop(self):
        if self._sub:
            self._sub.unsubscribe()
            self._sub = None
        carb.log_info("Controller stopped")

    # ------------------------------------------------------------

    def _on_step(self, dt):

        dt = self._extract_dt(dt)
        if not dt:
            return

        pos = self._get_position(self.cubesat_prim)
        altitude = pos[2] - self.ground_altitude_m

        # --- Stop near ground ---
        if altitude < 3.0:
            self._set_velocity(Gf.Vec3f(0, 0, 0))
            self._set_angular_velocity(Gf.Vec3f(0, 0, 0))
            self.stop()
            return

        # --- Stage switch ---
        if self.current_stage == "drogue" and altitude < self.main_deploy_alt:
            self.current_stage = "main"
            carb.log_info("Main parachute deployed")

        target_descent_speed = (
            self.descent_main if self.current_stage == "main" else self.descent_drogue
        )

        # Smooth approach to stage descent speed (no hard velocity jumps)
        self.vertical_speed += (
            target_descent_speed - self.vertical_speed
        ) * min(1.0, self.descent_response * dt)

        # --------------------------------------------------------
        # RANDOM SWING DIRECTION
        # --------------------------------------------------------
        self.dir_timer -= dt
        if self.dir_timer <= 0:
            self.dir_timer = random.uniform(1.5, 4.0)

            angle = random.uniform(0, 2 * math.pi)
            self.swing_dir = Gf.Vec3f(math.cos(angle), math.sin(angle), 0)

            self.swing_amp = random.uniform(5.0, 12.0)
            self.swing_freq = random.uniform(0.8, 1.6)

        # --------------------------------------------------------
        # WIND FIELD (SLOWLY CHANGING)
        # --------------------------------------------------------
        self.wind_timer -= dt
        if self.wind_timer <= 0:
            self.wind_timer = random.uniform(2.0, 5.0)

            angle = random.uniform(0, 2 * math.pi)
            wind_strength = random.uniform(0.8, 2.8)
            if self.current_stage == "drogue":
                wind_strength *= 1.2

            self.wind_target = Gf.Vec3f(
                math.cos(angle) * wind_strength,
                math.sin(angle) * wind_strength,
                0,
            )

        self.wind = Gf.Vec3f(
            self.wind[0] + (self.wind_target[0] - self.wind[0]) * min(1.0, self.wind_response * dt),
            self.wind[1] + (self.wind_target[1] - self.wind[1]) * min(1.0, self.wind_response * dt),
            0,
        )

        # --------------------------------------------------------
        # PENDULUM MOTION
        # --------------------------------------------------------
        self.phase += dt * self.swing_freq
        altitude_scale = max(0.25, min(1.0, altitude / 600.0))
        stage_scale = 0.7 if self.current_stage == "main" else 1.0
        swing = math.sin(self.phase) * self.swing_amp * altitude_scale * stage_scale

        lateral = Gf.Vec3f(
            self.swing_dir[0] * swing,
            self.swing_dir[1] * swing,
            0,
        )

        # --- Filtered turbulence ---
        turbulence_target = Gf.Vec3f(
            random.uniform(-0.6, 0.6),
            random.uniform(-0.6, 0.6),
            0,
        )
        self.turbulence = Gf.Vec3f(
            self.turbulence[0]
            + (turbulence_target[0] - self.turbulence[0])
            * min(1.0, self.turbulence_response * dt),
            self.turbulence[1]
            + (turbulence_target[1] - self.turbulence[1])
            * min(1.0, self.turbulence_response * dt),
            0,
        )

        # --- Final velocity (smoothed lateral + controlled descent) ---
        lateral_target = lateral + self.wind + self.turbulence
        self.lateral_velocity = Gf.Vec3f(
            self.lateral_velocity[0]
            + (lateral_target[0] - self.lateral_velocity[0])
            * min(1.0, self.lateral_response * dt),
            self.lateral_velocity[1]
            + (lateral_target[1] - self.lateral_velocity[1])
            * min(1.0, self.lateral_response * dt),
            0,
        )

        velocity = self.lateral_velocity + Gf.Vec3f(0, 0, self.vertical_speed)
        self._set_velocity(velocity)

        # --------------------------------------------------------
        # ROTATION (Z AXIS SPIN)
        # --------------------------------------------------------
        self.spin_timer -= dt
        if self.spin_timer <= 0:
            self.spin_timer = random.uniform(1.0, 3.0)

            lateral_mag = math.sqrt(
                self.lateral_velocity[0] ** 2 + self.lateral_velocity[1] ** 2
            )

            base_spin = lateral_mag * 0.12
            random_spin = random.uniform(-1.2, 1.2)

            self.spin_target = base_spin + random_spin

        # Smooth transition
        self.spin_rate += (self.spin_target - self.spin_rate) * 0.8 * dt

        # Damping
        self.spin_rate *= self.spin_damping

        # Clamp
        self.spin_rate = max(-self.max_spin, min(self.max_spin, self.spin_rate))

        # Optional wobble (adds realism)
        wobble = min(0.25, 0.04 * math.sqrt(velocity[0] ** 2 + velocity[1] ** 2))
        wx = random.uniform(-wobble, wobble)
        wy = random.uniform(-wobble, wobble)

        self._set_angular_velocity(Gf.Vec3f(wx, wy, self.spin_rate))

    # ------------------------------------------------------------

    def _set_velocity(self, vel):
        if self.vel_attr and self.vel_attr.IsValid():
            self.vel_attr.Set(Gf.Vec3f(float(vel[0]), float(vel[1]), float(vel[2])))

    def _set_angular_velocity(self, w):
        if self.ang_vel_attr and self.ang_vel_attr.IsValid():
            self.ang_vel_attr.Set(Gf.Vec3f(float(w[0]), float(w[1]), float(w[2])))

    def _get_position(self, prim):
        xform = omni.usd.get_world_transform_matrix(prim)
        p = xform.ExtractTranslation()
        return Gf.Vec3f(p[0], p[1], p[2])

    def _extract_dt(self, value):
        if isinstance(value, (int, float)):
            return float(value)

        payload = getattr(value, "payload", None)
        if isinstance(payload, dict):
            return payload.get("dt", None)

        return None


# ------------------------------------------------------------
# Timeline autorun
# ------------------------------------------------------------

_controller = None
_timeline_sub = None


def start_controller():
    global _controller
    if _controller:
        _controller.stop()

    _controller = CubeSatRecoveryController()
    return _controller


def stop_controller():
    global _controller
    if _controller:
        _controller.stop()
        _controller = None


def _on_timeline_event(event):
    try:
        t = event.type

        if t == int(omni.timeline.TimelineEventType.PLAY):
            start_controller()

        elif t in (
            int(omni.timeline.TimelineEventType.STOP),
            int(omni.timeline.TimelineEventType.PAUSE),
        ):
            stop_controller()

    except Exception as e:
        carb.log_warn(str(e))


def enable_autorun():
    global _timeline_sub

    timeline = omni.timeline.get_timeline_interface()
    stream = timeline.get_timeline_event_stream()

    _timeline_sub = stream.create_subscription_to_pop(
        _on_timeline_event, name="RecoveryAutorun"
    )


enable_autorun()



