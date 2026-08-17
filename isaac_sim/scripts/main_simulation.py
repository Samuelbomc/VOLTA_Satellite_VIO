import math
import logging
import os
import sys
import random

from isaacsim import SimulationApp

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.append(_SCRIPT_DIR)



def configure_simulation_timing(physics_hz=100.0, render_hz=100.0, sync_render_to_physics=True):
    from simulation_timing import SimulationTimingConfig

    timing_config = SimulationTimingConfig(
        physics_hz=physics_hz,
        render_hz=render_hz,
        sync_render_to_physics=sync_render_to_physics,
    )
    timing_config.apply()
    return timing_config


def run(
    physics_hz=100.0,
    render_hz=100.0,
    headless=False,
    usd_path=None,
    isaac_sim_path=r"D:\isaac_sim",
    enable_ros2_bridge=True,
    warmup_frames=10,
    lock_render_to_physics=True,
    joint_radius_m=0.5,
    joint_period_s=2.0,
):
    # Environment setup for Isaac Sim runtime.
    os.environ.setdefault("EXP_PATH", _SCRIPT_DIR)
    os.environ.setdefault("CARB_APP_PATH", isaac_sim_path)
    os.environ.setdefault("ISAAC_PATH", isaac_sim_path)
    simulation_app = SimulationApp({"headless": headless})

    import omni
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.utils import extensions
    from pxr import Gf, Sdf, UsdPhysics

    if enable_ros2_bridge:
        # Enable ROS 2 bridge when available.
        extensions.enable_extension("isaacsim.ros2.bridge")

    simulation_app.update()

    if usd_path:
        # Load the stage and wait for it to finish loading.
        usd_context = omni.usd.get_context()
        usd_context.open_stage(usd_path)
        simulation_app.update()
        is_loading = getattr(usd_context, "is_stage_loading", None)
        if callable(is_loading):
            while is_loading():
                simulation_app.update()
        else:
            for _ in range(60):
                simulation_app.update()

    # Fixed-step timing.
    physics_dt = 1.0 / float(physics_hz)
    if lock_render_to_physics:
        render_hz = physics_hz
    render_dt = 1.0 / float(render_hz)

    configure_simulation_timing(
        physics_hz=physics_hz,
        render_hz=render_hz,
        sync_render_to_physics=True,
    )

    stage = omni.usd.get_context().get_stage()

    # DistanceJoint path for circular motion.
    joint_attrs = []
    if stage is not None:
        joint_prim = stage.GetPrimAtPath("/World/DistanceJoint")
        if joint_prim and joint_prim.IsValid() and joint_prim.IsA(UsdPhysics.DistanceJoint):
            joint = UsdPhysics.DistanceJoint(joint_prim)
            local_pos_attr = joint.GetLocalPos0Attr()
            if local_pos_attr and local_pos_attr.IsValid():
                joint_attrs.append(local_pos_attr)

    base_joint_z = 0.0
    if joint_attrs:
        current_value = joint_attrs[0].Get()
        if current_value is not None:
            base_joint_z = current_value[2]

    sim_context = SimulationContext(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    sim_context.initialize_physics()

    # Warmup frames to stabilize render and physics.
    for _ in range(max(0, int(warmup_frames))):
        simulation_app.update()
        sim_context.step(render=not headless)

    sim_context.play()
    sim_time = 0.0
    scale_min = 0.99
    scale_max = 1.01
    scale_change_period_s = 0.5
    scale_smoothing_rate = 4.0
    current_scale = 1.0
    target_scale = random.uniform(scale_min, scale_max)
    next_scale_change_time = 0.0

    while simulation_app.is_running():
        if joint_attrs:
            # Rotation simulation.
            phase = (sim_time / max(joint_period_s, 1e-6)) * 2.0 * math.pi
            if sim_time >= next_scale_change_time:
                target_scale = random.uniform(scale_min, scale_max)
                next_scale_change_time = sim_time + scale_change_period_s
            smoothing = min(1.0, scale_smoothing_rate * physics_dt)
            current_scale += (target_scale - current_scale) * smoothing
            x = (joint_radius_m * current_scale) * math.cos(phase)
            y = (joint_radius_m * current_scale) * math.sin(phase)
            pos = Gf.Vec3f(x, y, base_joint_z)
            for attr in joint_attrs:
                attr.Set(pos)
        sim_context.step(render=not headless)
        sim_time += physics_dt

    simulation_app.close()

if __name__ == "__main__":
    run(usd_path=r"D:\dev\vio\isaac_sim\VIO.usd", headless=False, isaac_sim_path=r"D:\isaac_sim")


