# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "StyleTransfer",
    "author": "pluszak",
    "description": "xdxdxdxd",
    "blender": (2, 80, 0),
    "version": (0, 0, 1),
    "location": "View3D",
    "warning": "",
    "category": "Generic"
}

import bpy

from .style_transfer_op import StyleTransfer_OT_Operator
from .style_transfer_op import StyleTransfer_OT_TextField
from .style_transfer_panel import StyleTransfer_PT_Panel
from .style_transfer_panel_new import RenderBurstCamerasPanel
from .style_transfer_panel_new import RenderBurst

classes = (StyleTransfer_OT_Operator, StyleTransfer_PT_Panel, RenderBurstCamerasPanel,StyleTransfer_OT_TextField)

register, unregister = bpy.utils.register_classes_factory(classes)
