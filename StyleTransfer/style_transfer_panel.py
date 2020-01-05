import bpy 

class StyleTransfer_PT_Panel(bpy.types.Panel):
    bl_idname = "view3d.cursor_center"
    bl_label = "Simple operator"
    bl_category = "Center 3d cursor"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.operator('view3d.cursor_center', text='Center 3D Cursor')

