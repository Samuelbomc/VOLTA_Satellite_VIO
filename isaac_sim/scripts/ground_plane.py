from pxr import Usd, Sdf, UsdShade, Vt

# Get plane prim
stage = omni.usd.get_context().get_stage()
plane_prim = stage.GetPrimAtPath("/World/Plane")

# Create a material and assign image
material_path = Sdf.Path("/World/Looks/PlaneMaterial")
image_path = "/isaac_sim/scene/ground.png"
material = UsdShade.Material.Define(stage, material_path)

# Create a shader
shader = UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
color_array = Vt.Vec3fArray([1.0, 1.0, 1.0])
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Vector3fArray).Set(color_array)

# Connect the shader to the material's surface output
material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

# Add st reader
st_Reader = UsdShade.Shader.Define(stage, material_path.AppendPath('st_Reader'))
st_Reader.CreateIdAttr('UsdPrimvarReader_float2')
st_Reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

# Add diffuse texture
diffuseTextureSampler = UsdShade.Shader.Define(stage,material_path.AppendPath("diffuseTexture"))
diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(image_path)
diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_Reader.ConnectableAPI(), 'result')
diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')

# Bind the material the plane
UsdShade.MaterialBindingAPI(plane_prim).Bind(material)