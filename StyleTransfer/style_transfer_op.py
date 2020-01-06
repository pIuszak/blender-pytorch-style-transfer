import bpy

class StyleTransfer_OT_Operator(bpy.types.Operator):
    bl_idname = "view3d.cursor_center"
    bl_label = "Simple operator"
    bl_description = "Center 3d cursor"

    def execute(self, context):
        bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, location=(0, 0, 0))
        return {'FINISHED'}

class StyleTransfer_OT_TextField(bpy.types.Operator):
    bl_idname = "view3d.textfield"
    bl_label = "Simple asd"
    bl_description = "Center 3d asd"

    def execute(self, context):
        bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, location=(0, 0, 0))
        return {'FINISHED'}