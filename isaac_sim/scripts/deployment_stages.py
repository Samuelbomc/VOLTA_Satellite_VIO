import math
import random

import carb
import omni
import omni.timeline
from pxr import Gf, PhysxSchema, UsdGeom, Usd


class RandomForceController:
    """Applies continuous random force and torque to a prim over a set duration, stopping if altitude <= 10m."""

    def __init__(
        self,
        target_prim_path="/World/CubeSat",
    ):
        self.stage = omni.usd.get_context().get_stage()
        self.target_prim = self.stage.GetPrimAtPath(target_prim_path)

        if not self.target_prim.IsValid():
            raise RuntimeError(f"Invalid prim path: {target_prim_path}")

        # ------------------------------------------------------------------
        # Force, Torque, and Timing Parameters
        # ------------------------------------------------------------------
        # Magnitudes are lowered because the force is now sustained over time.
        self.force_magnitude_n = 5.0  
        self.torque_magnitude_nm = 5.0
        
        self.duration_s = 2  # How long the force is continuously applied
        self.interval_s = 4  # How long to wait between pushes
        
        self.angular_damping = 0.4
        self.min_altitude = 10.0 # Minimum altitude to stop simulation
        
        self._timer = 0.0
        self._is_pushing = False

        # ------------------------------------------------------------------
        # PhysX API Setup
        # ------------------------------------------------------------------
        self._force_api = PhysxSchema.PhysxForceAPI.Apply(self.target_prim)
        self._force_attr = self._get_or_create_attr(self._force_api.GetForceAttr, self._force_api.CreateForceAttr)
        self._torque_attr = self._get_or_create_attr(self._force_api.GetTorqueAttr, self._force_api.CreateTorqueAttr)

        # Apply initial rotational damping to the prim
        self._set_angular_damping(self.target_prim, self.angular_damping)

        # Subscribe to physics step events
        self._sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
            self._on_physics_step
        )

        carb.log_info(
            f"RandomForceController started | prim={target_prim_path} | damping={self.angular_damping}"
        )

    def stop(self, clear_forces=True):
        if clear_forces:
            self._clear_forces()

        if self._sub is not None:
            unsubscribe = getattr(self._sub, "unsubscribe", None)
            if callable(unsubscribe):
                unsubscribe()
            self._sub = None
        carb.log_info("RandomForceController stopped.")

    @staticmethod
    def _get_or_create_attr(get_attr, create_attr):
        attr = get_attr()
        if not attr.IsValid():
            attr = create_attr()
        return attr

    def _set_angular_damping(self, prim, damping):
        """Applies angular damping to the Rigid Body using the PhysX API."""
        try:
            rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            damping_attr = rb_api.GetAngularDampingAttr()
            if not damping_attr.IsValid():
                damping_attr = rb_api.CreateAngularDampingAttr()
            damping_attr.Set(float(damping))
            return
        except Exception:
            pass        

        # Fallback method if direct API fails
        attr = prim.GetAttribute("physxRigidBody:angularDamping")
        if attr and attr.IsValid():
            attr.Set(float(damping))
        else:
            carb.log_warn(f"Could not set angular damping on prim: {prim.GetPath()}")

    def _on_physics_step(self, dt):
        dt = self._extract_dt(dt)
        if dt is None or dt <= 0.0:
            return

        if not self.target_prim.IsValid():
            carb.log_warn("Target prim is no longer valid; stopping controller.")
            self.stop()
            return

        # ------------------------------------------------------------------
        # Altitude Check
        # ------------------------------------------------------------------
        # Get the global transform to extract the Z position (altitude)
        pose = UsdGeom.Xformable(self.target_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = pose.ExtractTranslation()
        altitude = translation[2] # Z-axis is index 2

        if altitude <= self.min_altitude:
            carb.log_warn(f"Altitude reached {altitude:.2f} m (<= {self.min_altitude} m). Stopping simulation.")
            omni.timeline.get_timeline_interface().stop()
            # timeline stop event triggers our stop_controller automatically
            return

        # ------------------------------------------------------------------
        # Force/Torque Logic
        # ------------------------------------------------------------------
        self._timer += dt

        if not self._is_pushing:
            # WAITING PHASE: Check if it's time to start pushing
            if self._timer >= self.interval_s:
                self._timer = 0.0  # Reset timer for the duration phase
                self._is_pushing = True

                force = self._generate_force()
                torque = self._generate_torque()

                # Set the attributes once; PhysX will continuously apply them until zeroed out
                self._force_attr.Set(force)
                self._torque_attr.Set(torque)
                
                carb.log_info(
                    f"Push STARTED -> Force: ({force[0]:.1f}, {force[1]:.1f}, {force[2]:.1f}) N | "
                    f"Torque: ({torque[0]:.1f}, {torque[1]:.1f}, {torque[2]:.1f}) Nm "
                    f"for {self.duration_s}s"
                )
        else:
            # PUSHING PHASE: Check if it's time to stop pushing
            if self._timer >= self.duration_s:
                self._timer = 0.0  # Reset timer for the waiting phase
                self._is_pushing = False
                
                self._clear_forces()
                carb.log_info(f"Push ENDED. Waiting for {self.interval_s}s.")

    def _generate_force(self):
        angle = random.uniform(0.0, 2.0 * math.pi)
        return Gf.Vec3f(
            self.force_magnitude_n * math.cos(angle),
            self.force_magnitude_n * math.sin(angle),
            0.0,
        )

    def _generate_torque(self):
        axis = Gf.Vec3f(
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
        )
        magnitude = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
        if magnitude <= 0.001:
            return Gf.Vec3f(0.0, 0.0, self.torque_magnitude_nm)
        scale = self.torque_magnitude_nm / magnitude
        return Gf.Vec3f(axis[0] * scale, axis[1] * scale, axis[2] * scale)

    def _clear_forces(self):
        self._force_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        self._torque_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))

    def _extract_dt(self, value):
        if isinstance(value, (int, float)):
            return float(value)

        payload = getattr(value, "payload", None)
        if isinstance(payload, dict):
            for key in ("dt", "deltaTime", "delta_time"):
                dt_value = payload.get(key)
                if isinstance(dt_value, (int, float)):
                    return float(dt_value)

        if isinstance(value, dict):
            for key in ("dt", "deltaTime", "delta_time"):
                dt_value = value.get(key)
                if isinstance(dt_value, (int, float)):
                    return float(dt_value)

        return None


# ==============================================================================
# Isaac Sim Timeline Management
# ==============================================================================

_controller = None
_timeline_sub = None

def start_controller(target_prim_path="/World/CubeSat"):
    global _controller
    if _controller is not None:
        _controller.stop()
    _controller = RandomForceController(target_prim_path=target_prim_path)
    return _controller

def stop_controller():
    global _controller
    if _controller is not None:
        _controller.stop()
        _controller = None

def _on_timeline_event(event):
    try:
        event_type = getattr(event, "type", None)
        if event_type == int(omni.timeline.TimelineEventType.PLAY):
            carb.log_info("Playback started -> starting continuous random force/torque controller")
            start_controller()
        elif event_type in (
            int(omni.timeline.TimelineEventType.STOP),
            int(omni.timeline.TimelineEventType.PAUSE),
        ):
            carb.log_info("Playback paused/stopped -> stopping controller")
            stop_controller()
    except Exception as exc:
        carb.log_warn(f"Failed handling timeline event: {exc}")

def enable_playback_autorun():
    global _timeline_sub
    if _timeline_sub is not None:
        return

    timeline = omni.timeline.get_timeline_interface()
    stream = timeline.get_timeline_event_stream()
    _timeline_sub = stream.create_subscription_to_pop(
        _on_timeline_event, name="RandomForcePlaybackAutoRun"
    )
    carb.log_info("Continuous random force auto-run enabled.")

def disable_playback_autorun():
    global _timeline_sub
    if _timeline_sub is not None:
        unsubscribe = getattr(_timeline_sub, "unsubscribe", None)
        if callable(unsubscribe):
            unsubscribe()
        _timeline_sub = None
    stop_controller()
    carb.log_info("Continuous random force auto-run disabled.")

# Activate the script when pressing 'Play' in Isaac Sim
enable_playback_autorun()



















