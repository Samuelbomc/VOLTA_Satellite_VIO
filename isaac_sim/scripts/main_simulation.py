import os
import os
import sys
import time

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
    target_prim_path="/World/CubeSat",
    headless=False,
    usd_path=None,
    isaac_sim_path=r"D:\isaac_sim",
    enable_ros2_bridge=True,
    warmup_frames=5,
    lock_render_to_physics=True,
    step_delay_s=0.2,
):
    os.environ.setdefault("EXP_PATH", _SCRIPT_DIR)
    os.environ.setdefault("CARB_APP_PATH", isaac_sim_path)
    os.environ.setdefault("ISAAC_PATH", isaac_sim_path)
    simulation_app = SimulationApp({"headless": headless})

    import omni
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.utils import extensions
    from random_force_controller import RandomForceController

    if enable_ros2_bridge:
        extensions.enable_extension("isaacsim.ros2.bridge")

    simulation_app.update()

    if usd_path:
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

    physics_dt = 1.0 / float(physics_hz)
    if lock_render_to_physics:
        render_hz = physics_hz
    render_dt = 1.0 / float(render_hz)

    configure_simulation_timing(
        physics_hz=physics_hz,
        render_hz=render_hz,
        sync_render_to_physics=True,
    )

    controller = RandomForceController(target_prim_path=target_prim_path)
    sim_context = SimulationContext(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
    sim_context.initialize_physics()

    for _ in range(max(0, int(warmup_frames))):
        simulation_app.update()
        sim_context.step(render=not headless)

    sim_context.play()

    while simulation_app.is_running():
        step_start = time.perf_counter()
        controller.update(physics_dt)
        sim_context.step(render=not headless)
        if step_delay_s > 0:
            elapsed = time.perf_counter() - step_start
            remaining = step_delay_s - elapsed
            if remaining > 0:
                time.sleep(remaining)

    controller.stop()
    simulation_app.close()

if __name__ == "__main__":
    run(usd_path=r"D:\dev\vio\isaac_sim\VIO.usd", headless=False, isaac_sim_path=r"D:\isaac_sim")


