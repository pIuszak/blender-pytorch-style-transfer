import bpy 

class StyleTransfer_PT_Panel(bpy.types.Panel):
    bl_idname = "view3d.cursor_center"
    bl_label = "Style Transfer by pluszak"
    bl_space_type = "PROPERTIES"
    bl_region_type = 'WINDOW'
    bl_context = "material"
    COMPAT_ENGINES = {'CYCLES'}

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        column = layout.column()
        column.prop(context.window_manager, "pmc_texture_path")
        column.prop(context.window_manager, "pmc_texture_path")
        column.operator('view3d.cursor_center', text='Start')
        #column.operator()
'''        wm = context.window_manager
        row = self.layout.row()
        row.prop(wm.rb_filter, "rb_filter_enum", expand=True)
        row = self.layout.row()
        row.operator("rb.renderbutton", text='Render!')
        row = self.layout.row()
        row.operator('view3d.cursor_center', text='Center 3D xd13')'''


