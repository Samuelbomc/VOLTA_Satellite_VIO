import math
import random

import carb
import omni
from pxr import Gf, PhysxSchema, UsdGeom, Usd, UsdPhysics


class RandomForceController:
    """Applies random force/torque bursts with counter actions to cap acceleration and angular speed."""

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
        self.force_magnitude_n = 5
        self.torque_magnitude_nm = 0.005
        self.duration_s = 0.05
        self.interval_s = 5.0
        self.angular_damping = 0.9
        self.min_altitude = 20.0

        # Limits
        self.max_linear_accel_g = 2.0
        self.max_angular_speed = 1.5
        self.counter_torque_gain = 1.0

        self._timer = self.interval_s
        self._is_pushing = False
        self._current_force = Gf.Vec3f(0.0, 0.0, 0.0)
        self._current_torque = Gf.Vec3f(0.0, 0.0, 0.0)

        # ------------------------------------------------------------------
        # PhysX API Setup
        # ------------------------------------------------------------------
        self._force_api = PhysxSchema.PhysxForceAPI.Apply(self.target_prim)
        self._force_attr = self._get_or_create_attr(self._force_api.GetForceAttr, self._force_api.CreateForceAttr)
        self._torque_attr = self._get_or_create_attr(self._force_api.GetTorqueAttr, self._force_api.CreateTorqueAttr)

        # Apply initial rotational damping to the prim
        self._set_angular_damping(self.target_prim, self.angular_damping)

        carb.log_info(
            f"RandomForceController started | prim={target_prim_path} | damping={self.angular_damping}"
        )

    def stop(self, clear_forces=True):
        if clear_forces:
            self._clear_forces()

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

        attr = prim.GetAttribute("physxRigidBody:angularDamping")
        if attr and attr.IsValid():
            attr.Set(float(damping))
        else:
            carb.log_warn(f"Could not set angular damping on prim: {prim.GetPath()}")

    def update(self, dt):
        dt = float(dt)
        if dt <= 0.0:
            return

        if not self.target_prim.IsValid():
            carb.log_warn("Target prim is no longer valid; stopping controller.")
            self.stop()
            return

        pose = UsdGeom.Xformable(self.target_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = pose.ExtractTranslation()
        altitude = translation[2]

        if altitude <= self.min_altitude:
            carb.log_warn(f"Altitude reached {altitude:.2f} m (<= {self.min_altitude} m). Stopping simulation.")
            omni.timeline.get_timeline_interface().stop()
            return

        self._timer += dt

        if not self._is_pushing:
            if self._timer >= self.interval_s:
                self._timer = 0.0
                self._is_pushing = True

                self._current_force = self._generate_force()
                self._current_torque = self._generate_torque()

                carb.log_info(
                    f"Push STARTED -> Force: ({self._current_force[0]:.1f}, {self._current_force[1]:.1f}, {self._current_force[2]:.1f}) N | "
                    f"Torque: ({self._current_torque[0]:.3f}, {self._current_torque[1]:.3f}, {self._current_torque[2]:.3f}) Nm "
                    f"for {self.duration_s}s"
                )
        else:
            if self._timer >= self.duration_s:
                self._timer = 0.0
                self._is_pushing = False
                self._current_force = Gf.Vec3f(0.0, 0.0, 0.0)
                self._current_torque = Gf.Vec3f(0.0, 0.0, 0.0)

        counter_torque = self._counter_torque()
        total_force = self._limit_force(self._current_force)
        total_torque = Gf.Vec3f(
            self._current_torque[0] + counter_torque[0],
            self._current_torque[1] + counter_torque[1],
            self._current_torque[2] + counter_torque[2],
        )

        self._force_attr.Set(total_force)
        self._torque_attr.Set(total_torque)

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

    def _counter_torque(self):
        angular_velocity = self._get_angular_velocity()
        omega_mag = math.sqrt(
            angular_velocity[0] * angular_velocity[0]
            + angular_velocity[1] * angular_velocity[1]
            + angular_velocity[2] * angular_velocity[2]
        )
        if omega_mag <= self.max_angular_speed:
            return Gf.Vec3f(0.0, 0.0, 0.0)

        excess = omega_mag - self.max_angular_speed
        scale = -self.counter_torque_gain * excess / omega_mag
        return Gf.Vec3f(
            angular_velocity[0] * scale,
            angular_velocity[1] * scale,
            angular_velocity[2] * scale,
        )

    def _get_angular_velocity(self):
        rb_api = UsdPhysics.RigidBodyAPI(self.target_prim)
        attr = rb_api.GetAngularVelocityAttr()
        if attr and attr.IsValid():
            value = attr.Get()
            if value is not None:
                return value
        attr = self.target_prim.GetAttribute("physics:angularVelocity")
        if attr and attr.IsValid():
            value = attr.Get()
            if value is not None:
                return value
        return Gf.Vec3f(0.0, 0.0, 0.0)

    def _limit_force(self, force):
        max_force = self._get_mass() * (self.max_linear_accel_g * 9.80665)
        magnitude = math.sqrt(force[0] * force[0] + force[1] * force[1] + force[2] * force[2])
        if magnitude <= max_force:
            return force
        if magnitude <= 0.0:
            return Gf.Vec3f(0.0, 0.0, 0.0)
        scale = max_force / magnitude
        return Gf.Vec3f(force[0] * scale, force[1] * scale, force[2] * scale)

    def _get_mass(self):
        mass_attr = self.target_prim.GetAttribute("physics:mass")
        if mass_attr and mass_attr.IsValid():
            value = mass_attr.Get()
            if isinstance(value, (int, float)):
                return float(value)
        return 1.0

    def _clear_forces(self):
        self._force_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        self._torque_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
