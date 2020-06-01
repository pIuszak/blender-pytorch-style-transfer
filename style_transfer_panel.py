import bpy
from bpy.types import Operator, Panel

from bpy_extras.io_utils import ImportHelper
from . style_transfer_op import StyleTransfer_OT_Operator
import bpy

from bpy.props import StringProperty, BoolProperty
from bpy.types import Operator, Panel
from bpy_extras.io_utils import ImportHelper


class StyleTransfer_PT_Panel(bpy.types.Panel, ImportHelper):
    bl_idname = "view3d.cursor_center"
    bl_label = "Style Transfer by pluszak"
    bl_space_type = "PROPERTIES"
    bl_region_type = 'WINDOW'
    bl_context = "material"
    COMPAT_ENGINES = {'CYCLES'}

    bpy.types.Scene.first_path = StringProperty(
    name = "Content",
    default = "",
    description = "Define file path",
    subtype = 'FILE_PATH'
    )
    bpy.types.Scene.second_path = StringProperty(
    name = "Style",
    default = "",
    description = "Define file path",
    subtype = 'FILE_PATH'
    )
    bpy.types.Scene.resolution = StringProperty(
    name = "Resolution",
    default = "1024",
    description = "Numbers of X and Y in final image",
    )
    bpy.types.Scene.steps = StringProperty(
    name = "Steps",
    default = "2048",
    description = "Numbers of Steps",
    )


    def draw(self, context):
        layout = self.layout
        row = layout.row()
        column = layout.column()

        layout = self.layout

        scene = context.scene

        col = layout.column()
        first_string = col.prop(context.scene, 'first_path')
        fsecond_string = col.prop(context.scene, 'second_path')
        rs = col.prop(context.scene, 'resolution')
        ss = col.prop(context.scene, 'steps')


        row = layout.row()


        StyleTransfer_OT_Operator.content = bpy.context.scene.first_path
        StyleTransfer_OT_Operator.style =  bpy.context.scene.second_path
        StyleTransfer_OT_Operator.resolution = bpy.context.scene.resolution
        StyleTransfer_OT_Operator.steps = bpy.context.scene.steps


        row.operator('view3d.cursor_center', text='Start Style Transfer')



