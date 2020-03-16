import bpy
from bpy.types import Operator, Panel
'''
'''

from bpy_extras.io_utils import ImportHelper
from . style_transfer_op import StyleTransfer_OT_Operator

class StyleTransfer_PT_Panel(bpy.types.Panel, ImportHelper):
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

        layout = self.layout

        scene = context.scene

        col = layout.column()
        first_string = col.prop(context.scene, 'first_path')
        fsecond_string = col.prop(context.scene, 'second_path')
        ss = col.prop(context.scene, 'steps')

        row = layout.row()
        #GenerateOperator.string1 = bpy.context.scene.first_path
        #GenerateOperator.string2 = bpy.context.scene.second_path
        #row.operator('xd.generate')
        #row.operator('lol.generate')

        StyleTransfer_OT_Operator.content = bpy.context.scene.first_path
        StyleTransfer_OT_Operator.style =  bpy.context.scene.second_path
        StyleTransfer_OT_Operator.steps = bpy.context.scene.steps

        row.operator('view3d.cursor_center', text='Start Style Transfer')



        #column.operator()
'''        wm = context.window_manager
        row = self.layout.row()
        row.prop(wm.rb_filter, "rb_filter_enum", expand=True)
        row = self.layout.row()
        row.operator("rb.renderbutton", text='Render!')
        row = self.layout.row()
        row.operator('view3d.cursor_center', text='Center 3D xd13')'''


