import math
import math
import random

import carb
import omni
import omni.timeline
from pxr import Gf, PhysxSchema, Sdf, UsdPhysics


class CubeSatRecoveryController:
    """
    Two-stage parachute descent controller for a CubeSat recovery simulation.

    Model assumptions:
    - The payload (CubeSat) and the drag body (invisible sphere) are connected by a spring/distance joint.
    - Drag is applied to the drag body.
    - Stage switches from drogue to main when altitude crosses `main_deploy_alt_m`.
    """

    def __init__(
        self,
        cubesat_path="/World/CubeSat",
        drag_body_path="/World/Parachute_Dummy",
        joint_path="/World/CubeSat/Joint/Spring",
        ground_altitude_m=0.0,
    ):
        self.stage = omni.usd.get_context().get_stage()
        self.cubesat_prim = self.stage.GetPrimAtPath(cubesat_path)
        self.drag_prim = self.stage.GetPrimAtPath(drag_body_path)
        self.joint_prim = self.stage.GetPrimAtPath(joint_path)

        if not self.cubesat_prim.IsValid():
            raise RuntimeError(f"Invalid CubeSat prim path: {cubesat_path}")
        if not self.drag_prim.IsValid():
            raise RuntimeError(f"Invalid drag-body prim path: {drag_body_path}")
        if not self.joint_prim.IsValid():
            raise RuntimeError(f"Invalid joint prim path: {joint_path}")

        # ------------------------------------------------------------------
        # Mission / hardware parameters
        # ------------------------------------------------------------------
        self.apogee_m = 3161.91
        self.main_deploy_alt_m = 450.0
        self.vt_drogue_mps = 25.16
        self.vt_main_mps = 4.5
        self.cable_drogue_m = 19.51
        self.cable_main_m = 10.67
        self.mass_payload_kg = 4.0
        self.mass_recovery_kg = 2.8
        self.mass_total_kg = self.mass_payload_kg + self.mass_recovery_kg
        self.g = 9.81
        self.ground_altitude_m = ground_altitude_m
        self.rho0 = 1.225

        # ------------------------------------------------------------------
        # Atmosphere / wind model (Ornstein-Uhlenbeck process)
        # ------------------------------------------------------------------
        self.wind_xy = Gf.Vec3f(0.0, 0.0, 0.0)
        self.mean_wind_xy = Gf.Vec3f(1.5, -0.6, 0.0)  # m/s mean horizontal wind
        self.wind_tau = 2.5  # s
        self.wind_sigma = 1.2  # m/s stochastic component

        # ------------------------------------------------------------------
        # Stage transition state
        # ------------------------------------------------------------------
        self.main_blend_time_s = 1.8
        self.main_blend_elapsed_s = 0.0
        self.in_main_transition = False

        # Payload rotational damping
        self.payload_angular_damping = 0.45

        self.current_stage = "drogue"
        self.debug_enabled = True
        self._debug_elapsed_s = 0.0
        self._debug_log_interval_s = 1.0
        # ------------------------------------------------------------------
        # Controller tuning (realistic behavior with stability safeguards)
        # ------------------------------------------------------------------
        self.max_torque_nm = 18.0
        self.cable_torsional_damping_nm_per_radps = 2.8
        self.cable_spin_coupling_nm_per_mps = 0.10
        self.tangle_event_probability_per_s = 0.03
        self.tangle_decay_per_s = 0.45
        self.tangle_torque_nm = 4.0
        self.tangle_state = 0.0
        self.tangle_axis = Gf.Vec3f(0.0, 0.0, 1.0)
        self._invalid_prim_shutdown = False
        self.drag_linear_damping_drogue = 0.35
        self.drag_linear_damping_main = 1.1
        self._last_linear_damping_debug = None
        self.max_rel_speed_mps = 100.0
        self.max_drag_force_n = 220.0
        self.max_force_component_n = 170.0
        self.max_force_slew_rate_nps = 1500.0
        self._last_applied_force = Gf.Vec3f(0.0, 0.0, 0.0)
        self.max_operational_rel_speed_mps = 90.0
        self.stability_rel_speed_start_mps = 60.0
        self.high_speed_extra_linear_damping = 2.0
        self.enable_jitter = True
        self.wind_xy = Gf.Vec3f(0.0, 0.0, 0.0)
        self.mean_wind_xy = Gf.Vec3f(1.5, -0.6, 0.0)
        self.wind_sigma = 0.9

        # Effective CdA for each parachute stage (drag applied to drag body)
        self.cda_drogue = self._required_cda_for_terminal_speed(self.vt_drogue_mps, self.rho0)
        self.cda_main = self._required_cda_for_terminal_speed(self.vt_main_mps, self.rho0)

        # ------------------------------------------------------------------
        # PhysX APIs used by controller
        # ------------------------------------------------------------------
        # Force is applied to the parachute drag body.
        self.force_api = PhysxSchema.PhysxForceAPI.Apply(self.drag_prim)
        self.force_attr = self.force_api.GetForceAttr()
        if not self.force_attr.IsValid():
            self.force_attr = self.force_api.CreateForceAttr()

        # Torque is applied to CubeSat for cable-induced rotational effects.
        self.cubesat_force_api = PhysxSchema.PhysxForceAPI.Apply(self.cubesat_prim)
        self.torque_attr = self.cubesat_force_api.GetTorqueAttr()
        if not self.torque_attr.IsValid():
            self.torque_attr = self.cubesat_force_api.CreateTorqueAttr()

        self._set_initial_properties()
        self._sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
            self._on_physics_step
        )

        carb.log_info(
            f"CubeSatRecoveryController started | cubesat={cubesat_path}, drag={drag_body_path}, joint={joint_path}"
        )

    def stop(self, clear_forces=True):
        if clear_forces:
            if hasattr(self, "force_attr") and self.force_attr is not None:
                self._set_force_safe(Gf.Vec3f(0.0, 0.0, 0.0))
            if hasattr(self, "torque_attr") and self.torque_attr is not None:
                self._set_torque_safe(Gf.Vec3f(0.0, 0.0, 0.0))
        if self._sub is not None:
            unsubscribe = getattr(self._sub, "unsubscribe", None)
            if callable(unsubscribe):
                unsubscribe()
            self._sub = None
        carb.log_info("CubeSatRecoveryController stopped.")

    def _set_initial_properties(self):
        self._set_mass(self.cubesat_prim, self.mass_payload_kg)
        self._set_mass(self.drag_prim, self.mass_recovery_kg)
        self._set_angular_damping(self.cubesat_prim, self.payload_angular_damping)
        self._set_linear_damping(self.drag_prim, self.drag_linear_damping_drogue)
        print(f"[RecoveryDebug] init linear damping set for drag body: {self.drag_linear_damping_drogue}")
        self._set_joint_cable_length(self.cable_drogue_m)

    def _on_physics_step(self, dt):
        # Physics-step callback: compute forces/torques and apply safely each frame.
        dt = self._extract_dt(dt)
        if dt is None:
            return
        if dt <= 0.0:
            return

        dt = min(float(dt), 0.1)

        if (not self.cubesat_prim.IsValid()) or (not self.drag_prim.IsValid()):
            if not self._invalid_prim_shutdown:
                self._invalid_prim_shutdown = True
                carb.log_warn("Recovery controller prim became invalid; stopping controller.")
                self.stop()
            return

        try:
            cubesat_pos = self._get_world_position(self.cubesat_prim)
            drag_vel = self._get_linear_velocity(self.drag_prim)
        except Exception as exc:
            carb.log_warn(f"Recovery controller physics read failed: {exc}")
            self._set_force_safe(Gf.Vec3f(0.0, 0.0, 0.0))
            self._set_torque_safe(Gf.Vec3f(0.0, 0.0, 0.0))
            return

        if not self._state_is_valid(cubesat_pos, drag_vel):
            carb.log_warn("Invalid transform/state detected; disabling recovery controller for this run.")
            self.stop(clear_forces=False)
            return
        altitude_m = max(0.0, cubesat_pos[2] - self.ground_altitude_m)
        rho = self._air_density_isa(altitude_m)

        # Final safety cutoff near ground to avoid contact-phase force spikes.
        if altitude_m <= 10.0:
            print(f"[RecoveryDebug] landing threshold reached at alt={altitude_m:.2f}m -> controller stopped")
            self._set_force_safe(Gf.Vec3f(0.0, 0.0, 0.0))
            self._set_torque_safe(Gf.Vec3f(0.0, 0.0, 0.0))
            self.stop(clear_forces=False)
            return

        # Stage switch (drogue -> main).
        if self.current_stage == "drogue" and altitude_m <= self.main_deploy_alt_m:
            self.current_stage = "main"
            self.in_main_transition = True
            self.main_blend_elapsed_s = 0.0
            carb.log_info("Main parachute deployment triggered.")

        self._update_wind(dt)
        air_velocity = self._sanitize_vec3(self.wind_xy)
        relative_velocity = drag_vel - air_velocity
        rel_speed = math.sqrt(
            float(relative_velocity[0]) * float(relative_velocity[0])
            + float(relative_velocity[1]) * float(relative_velocity[1])
            + float(relative_velocity[2]) * float(relative_velocity[2])
        )
        stability_alpha = 0.0
        if rel_speed > self.stability_rel_speed_start_mps:
            denom = max(1.0, self.max_operational_rel_speed_mps - self.stability_rel_speed_start_mps)
            stability_alpha = min(1.0, (rel_speed - self.stability_rel_speed_start_mps) / denom)

        if stability_alpha > 0.0:
            print(
                f"[RecoveryDebug] stability mode alpha={stability_alpha:.2f} at alt={altitude_m:.1f}m, rel_speed={rel_speed:.2f}"
            )

        if self.in_main_transition:
            # Smooth transition of drag/cable/damping during deployment.
            self.main_blend_elapsed_s += dt
            alpha = min(1.0, self.main_blend_elapsed_s / self.main_blend_time_s)
            cda = self._lerp(self.cda_drogue, self.cda_main, alpha)
            target_cable = self._lerp(self.cable_drogue_m, self.cable_main_m, alpha)
            target_lin_damping = self._lerp(self.drag_linear_damping_drogue, self.drag_linear_damping_main, alpha)
            target_lin_damping += self.high_speed_extra_linear_damping * stability_alpha
            self._set_joint_cable_length(target_cable)
            self._set_linear_damping(self.drag_prim, target_lin_damping)
            if alpha >= 1.0:
                self.in_main_transition = False
        else:
            # Steady-state parameters after transition.
            cda = self.cda_main if self.current_stage == "main" else self.cda_drogue
            target_lin_damping = self.drag_linear_damping_main if self.current_stage == "main" else self.drag_linear_damping_drogue
            target_lin_damping += self.high_speed_extra_linear_damping * stability_alpha
            self._set_linear_damping(self.drag_prim, target_lin_damping)

        drag_force = self._quadratic_drag_force(relative_velocity, rho, cda)
        # Keep realistic lateral dynamics, but attenuate XY when stability mode ramps up.
        lateral_scale = max(0.35, 1.0 - 0.65 * stability_alpha)
        drag_force = Gf.Vec3f(
            float(drag_force[0]) * lateral_scale,
            float(drag_force[1]) * lateral_scale,
            float(drag_force[2]),
        )

        # Add weak pendulum excitation so the motion does not stay in a perfect 2D plane.
        jitter = self._random_lateral_force(self.mass_recovery_kg) if self.enable_jitter else Gf.Vec3f(0.0, 0.0, 0.0)
        raw_total_force = drag_force + jitter
        total_force = self._sanitize_force(raw_total_force)
        total_force = self._limit_force_slew(total_force, dt)
        self._set_force_safe(total_force)
        try:
            self._apply_cable_torsional_damping(dt, rel_speed, stability_alpha)
        except Exception as exc:
            carb.log_warn(f"Recovery controller torque update failed: {exc}")
            self._set_torque_safe(Gf.Vec3f(0.0, 0.0, 0.0))

        if self.debug_enabled:
            self._debug_elapsed_s += dt
            if self._debug_elapsed_s >= self._debug_log_interval_s:
                self._debug_elapsed_s = 0.0
                carb.log_info(
                    "[RecoveryDebug] "
                    f"stage={self.current_stage}, alt={altitude_m:.1f}m, "
                    f"rho={rho:.3f}, rel_speed={rel_speed:.2f}m/s, cda={cda:.2f}, "
                    f"force=({float(total_force[0]):.2f}, {float(total_force[1]):.2f}, {float(total_force[2]):.2f})"
                )
                print(
                    "[RecoveryDebug] "
                    f"stage={self.current_stage}, alt={altitude_m:.1f}m, "
                    f"lin_damp_target={target_lin_damping:.3f}, "
                    f"rel_speed={rel_speed:.2f}, "
                    f"raw_force_mag={self._vec_mag(raw_total_force):.2f}, "
                    f"force=({float(total_force[0]):.2f}, {float(total_force[1]):.2f}, {float(total_force[2]):.2f})"
                )

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

    def _set_mass(self, prim, mass_kg):
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_attr = mass_api.GetMassAttr()
        if not mass_attr.IsValid():
            mass_attr = mass_api.CreateMassAttr()
        mass_attr.Set(float(mass_kg))

    def _set_angular_damping(self, prim, damping):
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
            carb.log_warn("Unable to set angular damping on payload rigid body.")

    def _set_linear_damping(self, prim, damping):
        # Try multiple APIs/attributes because damping schemas differ across assets.
        target = float(max(0.0, damping))
        success = False
        readback = None

        try:
            rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            damping_attr = rb_api.GetLinearDampingAttr()
            if not damping_attr.IsValid():
                damping_attr = rb_api.CreateLinearDampingAttr()
            damping_attr.Set(target)
            readback = damping_attr.Get()
            success = True
        except Exception:
            pass

        if not success:
            try:
                rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                damping_attr = rb_api.GetLinearDampingAttr()
                if not damping_attr.IsValid():
                    damping_attr = rb_api.CreateLinearDampingAttr()
                damping_attr.Set(target)
                readback = damping_attr.Get()
                success = True
            except Exception:
                pass

        if not success:
            attr = prim.GetAttribute("physics:linearDamping")
            if not attr or not attr.IsValid():
                attr = prim.CreateAttribute("physics:linearDamping", Sdf.ValueTypeNames.Float)
            if attr and attr.IsValid():
                attr.Set(target)
                readback = attr.Get()
                success = True

        if self._last_linear_damping_debug is None or abs(self._last_linear_damping_debug - target) > 1e-4:
            self._last_linear_damping_debug = target
            print(
                f"[RecoveryDebug] set linear damping target={target:.4f}, "
                f"success={success}, readback={readback}"
            )

        if not success:
            print("[RecoveryDebug] failed to set linear damping on drag body")
            carb.log_warn("Unable to set linear damping on drag rigid body.")


    def _set_joint_cable_length(self, length_m):
        candidate_attrs = [
            "physics:distance",
            "physics:minDistance",
            "physics:maxDistance",
            "physxDistanceJoint:distance",
            "physxDistanceJoint:restLength",
        ]
        for attr_name in candidate_attrs:
            attr = self.joint_prim.GetAttribute(attr_name)
            if attr and attr.IsValid():
                attr.Set(float(length_m))

    def _get_world_position(self, prim):
        xform = omni.usd.get_world_transform_matrix(prim)
        p = xform.ExtractTranslation()
        return Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))

    def _get_linear_velocity(self, prim):
        vel_attr = prim.GetAttribute("physics:velocity")
        if vel_attr and vel_attr.IsValid():
            v = vel_attr.Get()
            if v is not None:
                vec = Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))
                return self._sanitize_vec3(vec)
        return Gf.Vec3f(0.0, 0.0, 0.0)

    def _get_angular_velocity(self, prim):
        vel_attr = prim.GetAttribute("physics:angularVelocity")
        if vel_attr and vel_attr.IsValid():
            v = vel_attr.Get()
            if v is not None:
                return self._sanitize_vec3(Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])))
        return Gf.Vec3f(0.0, 0.0, 0.0)

    def _air_density_isa(self, altitude_m):
        # Simple ISA troposphere approximation (valid for your altitude range)
        t0 = 288.15
        p0 = 101325.0
        l = 0.0065
        r = 287.058
        g = 9.80665

        h = max(0.0, min(11000.0, altitude_m))
        t = t0 - l * h
        p = p0 * (t / t0) ** (g / (r * l))
        rho = p / (r * t)
        return rho

    def _required_cda_for_terminal_speed(self, vt_mps, rho):
        # mg = 0.5 * rho * CdA * vt^2
        vt = max(0.5, float(vt_mps))
        return (2.0 * self.mass_total_kg * self.g) / (max(0.05, rho) * vt * vt)

    def _quadratic_drag_force(self, v_rel, rho, cda):
        vx, vy, vz = float(v_rel[0]), float(v_rel[1]), float(v_rel[2])
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
            return Gf.Vec3f(0.0, 0.0, 0.0)

        rho = max(0.0, float(rho)) if math.isfinite(rho) else 0.0
        cda = max(0.0, float(cda)) if math.isfinite(cda) else 0.0

        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        if speed < 1e-4:
            return Gf.Vec3f(0.0, 0.0, 0.0)
        coeff = -0.5 * rho * cda * speed
        return Gf.Vec3f(coeff * vx, coeff * vy, coeff * vz)

    def _update_wind(self, dt):
        # Ornstein-Uhlenbeck for correlated gusts in XY
        for i in (0, 1):
            x = float(self.wind_xy[i])
            mu = float(self.mean_wind_xy[i])
            dx = (mu - x) * (dt / self.wind_tau) + self.wind_sigma * math.sqrt(dt) * random.gauss(0.0, 1.0)
            if i == 0:
                self.wind_xy[0] = x + dx
            else:
                self.wind_xy[1] = x + dx
        self.wind_xy[2] = 0.0

    def _sanitize_vec3(self, vec):
        x = self._safe_float(vec[0])
        y = self._safe_float(vec[1])
        z = self._safe_float(vec[2])
        return Gf.Vec3f(x, y, z)

    def _safe_float(self, value, default=0.0):
        try:
            f = float(value)
            return f if math.isfinite(f) else float(default)
        except Exception:
            return float(default)

    def _state_is_valid(self, cubesat_pos, drag_vel):
        px = self._safe_float(cubesat_pos[0])
        py = self._safe_float(cubesat_pos[1])
        pz = self._safe_float(cubesat_pos[2])
        vx = self._safe_float(drag_vel[0])
        vy = self._safe_float(drag_vel[1])
        vz = self._safe_float(drag_vel[2])

        if not all(math.isfinite(v) for v in (px, py, pz, vx, vy, vz)):
            return False

        max_pos = 1.0e6
        max_vel = 500.0
        if abs(px) > max_pos or abs(py) > max_pos or abs(pz) > max_pos:
            return False
        if abs(vx) > max_vel or abs(vy) > max_vel or abs(vz) > max_vel:
            return False

        return True

    def _sanitize_force(self, force):
        force = self._sanitize_vec3(force)
        mag = self._vec_mag(force)

        if not math.isfinite(mag) or mag <= 0.0:
            return Gf.Vec3f(0.0, 0.0, 0.0)

        if mag > self.max_drag_force_n:
            scale = self.max_drag_force_n / mag
            force = Gf.Vec3f(float(force[0]) * scale, float(force[1]) * scale, float(force[2]) * scale)

        fx = max(-self.max_force_component_n, min(self.max_force_component_n, float(force[0])))
        fy = max(-self.max_force_component_n, min(self.max_force_component_n, float(force[1])))
        fz = max(-self.max_force_component_n, min(self.max_force_component_n, float(force[2])))
        force = Gf.Vec3f(fx, fy, fz)

        return force

    def _limit_force_slew(self, force, dt):
        dt = max(1e-4, self._safe_float(dt, 0.01))
        max_delta = self.max_force_slew_rate_nps * dt

        last = self._last_applied_force
        delta = Gf.Vec3f(
            float(force[0]) - float(last[0]),
            float(force[1]) - float(last[1]),
            float(force[2]) - float(last[2]),
        )
        delta_mag = self._vec_mag(delta)

        if delta_mag > max_delta and delta_mag > 1e-6:
            s = max_delta / delta_mag
            force = Gf.Vec3f(
                float(last[0]) + float(delta[0]) * s,
                float(last[1]) + float(delta[1]) * s,
                float(last[2]) + float(delta[2]) * s,
            )

        self._last_applied_force = force
        return force

    def _vec_mag(self, vec):
        return math.sqrt(
            float(vec[0]) * float(vec[0])
            + float(vec[1]) * float(vec[1])
            + float(vec[2]) * float(vec[2])
        )

    def _set_force_safe(self, force):
        force = self._sanitize_force(force)
        fx = self._safe_float(force[0])
        fy = self._safe_float(force[1])
        fz = self._safe_float(force[2])
        if not (math.isfinite(fx) and math.isfinite(fy) and math.isfinite(fz)):
            fx, fy, fz = 0.0, 0.0, 0.0
        self.force_attr.Set(Gf.Vec3f(fx, fy, fz))

    def _set_torque_safe(self, torque):
        torque = self._sanitize_vec3(torque)
        mag = math.sqrt(
            float(torque[0]) * float(torque[0])
            + float(torque[1]) * float(torque[1])
            + float(torque[2]) * float(torque[2])
        )
        if not math.isfinite(mag) or mag <= 0.0:
            self.torque_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
            return
        if mag > self.max_torque_nm:
            scale = self.max_torque_nm / mag
            torque = Gf.Vec3f(float(torque[0]) * scale, float(torque[1]) * scale, float(torque[2]) * scale)
        self.torque_attr.Set(Gf.Vec3f(self._safe_float(torque[0]), self._safe_float(torque[1]), self._safe_float(torque[2])))

    def _apply_cable_torsional_damping(self, dt, rel_speed, stability_alpha=0.0):
        w = self._get_angular_velocity(self.cubesat_prim)
        xform = omni.usd.get_world_transform_matrix(self.cubesat_prim)
        z_axis = self._sanitize_vec3(xform.ExtractRotationMatrix().GetColumn(2))

        z_mag = math.sqrt(float(z_axis[0]) * float(z_axis[0]) + float(z_axis[1]) * float(z_axis[1]) + float(z_axis[2]) * float(z_axis[2]))
        if z_mag < 1e-6 or not math.isfinite(z_mag):
            self._set_torque_safe(Gf.Vec3f(0.0, 0.0, 0.0))
            return

        z_hat = Gf.Vec3f(float(z_axis[0]) / z_mag, float(z_axis[1]) / z_mag, float(z_axis[2]) / z_mag)
        spin_rate = float(w[0]) * float(z_hat[0]) + float(w[1]) * float(z_hat[1]) + float(w[2]) * float(z_hat[2])
        damping_coeff = self.cable_torsional_damping_nm_per_radps * (1.0 + 1.5 * stability_alpha)
        damping_torque_mag = -damping_coeff * self._safe_float(spin_rate)
        damping_torque = Gf.Vec3f(
            float(z_hat[0]) * damping_torque_mag,
            float(z_hat[1]) * damping_torque_mag,
            float(z_hat[2]) * damping_torque_mag,
        )

        # Aerodynamic spin coupling: faster airflow can induce cable-twist torque.
        excitation_scale = max(0.0, 1.0 - stability_alpha)
        coupling_mag = self.cable_spin_coupling_nm_per_mps * self._safe_float(rel_speed) * excitation_scale
        coupling_sign = 1.0 if random.random() > 0.5 else -1.0
        coupling_torque = Gf.Vec3f(
            float(z_hat[0]) * coupling_mag * coupling_sign,
            float(z_hat[1]) * coupling_mag * coupling_sign,
            float(z_hat[2]) * coupling_mag * coupling_sign,
        )

        # Intermittent tangle events: temporary off-axis torque that can start or increase rotation.
        dt = max(1e-4, self._safe_float(dt, 1.0 / 60.0))
        if random.random() < self.tangle_event_probability_per_s * dt * excitation_scale:
            self.tangle_state = min(1.0, self.tangle_state + random.uniform(0.35, 0.85))
            ax = random.uniform(-1.0, 1.0)
            ay = random.uniform(-1.0, 1.0)
            az = random.uniform(-0.3, 0.3)
            amag = math.sqrt(ax * ax + ay * ay + az * az)
            if amag > 1e-6:
                self.tangle_axis = Gf.Vec3f(ax / amag, ay / amag, az / amag)
                print(f"[RecoveryDebug] tangle event: level={self.tangle_state:.2f}, axis={self.tangle_axis}")

        self.tangle_state = max(0.0, self.tangle_state - self.tangle_decay_per_s * dt)
        tangle_mag = self.tangle_torque_nm * self.tangle_state * excitation_scale
        tangle_torque = Gf.Vec3f(
            float(self.tangle_axis[0]) * tangle_mag,
            float(self.tangle_axis[1]) * tangle_mag,
            float(self.tangle_axis[2]) * tangle_mag,
        )

        total_torque = damping_torque + coupling_torque + tangle_torque
        self._set_torque_safe(total_torque)

    def _random_lateral_force(self, mass_kg):
        a_xy = 0.12  # m/s^2 low-amplitude turbulent excitation
        fx = mass_kg * a_xy * random.uniform(-1.0, 1.0)
        fy = mass_kg * a_xy * random.uniform(-1.0, 1.0)
        return Gf.Vec3f(float(fx), float(fy), 0.0)

    @staticmethod
    def _lerp(a, b, alpha):
        return a + (b - a) * alpha


_controller = None
_timeline_sub = None


def start_controller(
    cubesat_path="/World/CubeSat",
    drag_body_path="/World/Parachute_Dummy",
    joint_path="/World/CubeSat/Joint/Spring",
    ground_altitude_m=0.0,
):
    global _controller
    if _controller is not None:
        _controller.stop()
    _controller = CubeSatRecoveryController(
        cubesat_path=cubesat_path,
        drag_body_path=drag_body_path,
        joint_path=joint_path,
        ground_altitude_m=ground_altitude_m,
    )
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
            carb.log_info("Playback started -> starting recovery controller")
            start_controller()
        elif event_type in (
            int(omni.timeline.TimelineEventType.STOP),
            int(omni.timeline.TimelineEventType.PAUSE),
        ):
            carb.log_info("Playback paused/stopped -> stopping recovery controller")
            stop_controller()
    except Exception as exc:
        carb.log_warn(f"Timeline event handling failed: {exc}")


def enable_playback_autorun():
    global _timeline_sub
    if _timeline_sub is not None:
        return

    timeline = omni.timeline.get_timeline_interface()
    stream = timeline.get_timeline_event_stream()
    _timeline_sub = stream.create_subscription_to_pop(
        _on_timeline_event, name="CubeSatRecoveryPlaybackAutoRun"
    )
    carb.log_info("CubeSat recovery playback auto-run enabled.")


def disable_playback_autorun():
    global _timeline_sub
    if _timeline_sub is not None:
        unsubscribe = getattr(_timeline_sub, "unsubscribe", None)
        if callable(unsubscribe):
            unsubscribe()
        _timeline_sub = None
    stop_controller()
    carb.log_info("CubeSat recovery playback auto-run disabled.")


enable_playback_autorun()



