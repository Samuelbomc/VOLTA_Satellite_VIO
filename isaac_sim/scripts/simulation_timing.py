import carb
import omni
from pxr import Sdf, UsdPhysics


def _find_physics_scene_prim(stage):
    if stage is None:
        return None
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return prim
    return None


class SimulationTimingConfig:
    def __init__(self, physics_hz=100.0, render_hz=100.0, rate_limit_enabled=False, skip_render_while_simulating=True, sync_render_to_physics=True):
        self.physics_hz = float(physics_hz)
        self.render_hz = float(render_hz)
        self.rate_limit_enabled = bool(rate_limit_enabled)
        self.skip_render_while_simulating = bool(skip_render_while_simulating)
        self.sync_render_to_physics = bool(sync_render_to_physics)

    def apply(self):
        render_hz = self.physics_hz if self.sync_render_to_physics else self.render_hz
        skip_render = False if self.sync_render_to_physics else self.skip_render_while_simulating

        settings = carb.settings.get_settings()
        settings.set("/app/runLoops/main/rateLimitEnabled", self.rate_limit_enabled)
        settings.set("/app/runLoops/main/rateLimitFrequency", 0 if not self.rate_limit_enabled else render_hz)
        settings.set("/app/renderer/skipWhileSimulating", skip_render)

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            stage.SetTimeCodesPerSecond(self.physics_hz)

        scene_prim = _find_physics_scene_prim(stage)
        if scene_prim is not None:
            scene = UsdPhysics.Scene(scene_prim)
            get_steps_attr = getattr(scene, "GetTimeStepsPerSecondAttr", None)
            create_steps_attr = getattr(scene, "CreateTimeStepsPerSecondAttr", None)
            if callable(get_steps_attr):
                steps_attr = get_steps_attr()
                if not steps_attr.IsValid() and callable(create_steps_attr):
                    steps_attr = create_steps_attr()
            else:
                steps_attr = scene_prim.GetAttribute("timeStepsPerSecond")
                if not steps_attr.IsValid():
                    steps_attr = scene_prim.CreateAttribute("timeStepsPerSecond", Sdf.ValueTypeNames.Float)
            if steps_attr and steps_attr.IsValid():
                steps_attr.Set(self.physics_hz)

        timeline = omni.timeline.get_timeline_interface()
        set_target_fps = getattr(timeline, "set_target_fps", None)
        if callable(set_target_fps):
            set_target_fps(render_hz)
