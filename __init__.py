bl_info = {
    "name": "StyleTransfer",
    "author": "Paweł Luszuk",
    "description": "Implement the Neural-Style algorithm developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge. Neural-Style, or Neural-Transfer, allows you to take an image and reproduce it with a new artistic style. ",
    "blender": (2, 82, 0),
    "version": (0, 9, 0),
    "location": "View3D",
    "warning": "",
    "category": "Generic"
}

import bpy

from .style_transfer_op import StyleTransfer_OT_Operator
from .style_transfer_op import StyleTransfer_OT_TextField
from .style_transfer_panel import StyleTransfer_PT_Panel

classes = (StyleTransfer_OT_Operator, StyleTransfer_PT_Panel, StyleTransfer_OT_TextField)

register, unregister = bpy.utils.register_classes_factory(classes)
