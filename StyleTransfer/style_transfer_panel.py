import bpy 

class StyleTransfer_PT_Panel(bpy.types.Panel):
    bl_idname = "view3d.cursor_center"
    bl_label = "Simple operator"
    bl_category = "Style Transfer by pluszak"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator('view3d.cursor_center', text='Center 3D xd 13')
        row.operator('view3d.cursor_center', text='Center 3D xd 14')
        row.operator('view3d.cursor_center', text='Center 3D xd 15')
        column = layout.column()

        column.operator('view3d.cursor_center', text='Center 3D xd 16')
        column.operator('view3d.cursor_center', text='Center 3D xd 17')
        column.operator('view3d.textfield', text='Center 3D xd 18')
        column.operator()

        wm = context.window_manager
        row = self.layout.row()
        row.prop(wm.rb_filter, "rb_filter_enum", expand=True)
        row = self.layout.row()
        row.operator("rb.renderbutton", text='Render!')
        row = self.layout.row()
        row.operator('view3d.cursor_center', text='Center 3D xd13')

