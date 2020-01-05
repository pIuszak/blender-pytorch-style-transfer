# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

import os
import json

import bpy

# from . import addon_updater_ops
# from . import poliigon_ops_props

# poliigon_url = "https://www.poliigon.com/search/recent/narrow/list/"
poliigon_url = "https://www.poliigon.com/search?query="
FILE_TICK = 'FILE_TICK' if bpy.app.version <= (2, 80) else 'CHECKBOX_HLT'


# -----------------------------------------------------------------------------
# UI CODE
# -----------------------------------------------------------------------------


def layout_split(layout, factor=0.0, align=False):
    """Intermediate method for pre and post blender 2.8 split UI function"""
    if not hasattr(bpy.app, "version") or bpy.app.version < (2, 80):
        return layout.split(percentage=factor, align=align)
    return layout.split(factor=factor, align=align)


class PMC_UL_foldersets(bpy.types.UIList):
    """UI list of folder sets"""

    # Called for each drawn item.
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index, flt_flag):
        # 'DEFAULT' and 'COMPACT' layout types should usually use the same draw code.
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            col = layout.column()

            # make checkbox more condense, but disable for 2.8 as
            # the UI interaction is misaligned for x-scaled checkboxes
            if bpy.app.version < (2, 80):
                col.scale_x = 0.7

            status = json.loads(item.status)
            col.prop(item, "checked", text="")

            col = layout.column()
            col.scale_x = 0.3
            col.label(text=item.name)

            if status != {}:
                col = layout.column()
                col.label(text="", icon="ERROR")
            if item.loaded:
                col = layout.column()
                col.label(text="", icon=FILE_TICK)

        # 'GRID' layout type should be as compact as possible (typically a single icon!)
        elif self.layout_type in {'GRID'}:
            col = layout
            if item.loaded == True:
                col.label(text="", icon=FILE_TICK)
            else:
                col.prop(item, "checked", text="")


class PMC_PT_main_panel(bpy.types.Panel):
    bl_idname = "PMC_PT_panel"
    bl_label = "Poliigon Material Converter"
    bl_space_type = "PROPERTIES"
    bl_region_type = 'WINDOW'
    bl_context = "material"
    COMPAT_ENGINES = {'CYCLES'}

    def draw(self, context):
        # addon_updater_ops.check_for_update_background()
        layout = self.layout
        pmcp = context.window_manager.pmc_props
        pmcs = context.scene.pmc_sceneprops

        if not hasattr(bpy.types, 'ShaderNodeBsdfPrincipled'):
            box = layout.box()
            col = box.column()
            col.scale_y = 0.8
            col.label(text="")
            col.label(text="This blender version is missing required", icon="ERROR")
            col.label(text="BSDF Principled shader node,", icon="BLANK1")
            col.label(text="use blender v2.79+ to use addon", icon="BLANK1")
            col.label(text="")
            return

        wrns = [itm for itm in pmcp.folderset_list.values()
                if itm.status != "{}"]
        ind = pmcp.folderset_list_index
        itm = None
        if pmcp.folderset_list:
            itm = pmcp.folderset_list[ind]

        col = layout.column(align=True)
        row = col.row(align=True)
        row.prop(context.window_manager, "pmc_texture_path")
        row.operator("pmc.refresh_folder", text="", icon="FILE_REFRESH")

        # config_data = poliigon_ops_props.config_data
        # if "texture_path" in config_data \
        #		and config_data["texture_path"]!="//" \
        #		and config_data["texture_path"]==context.window_manager.pmc_texture_path:
        #	row.operator("pmc.unset_default_path", text="", icon="X")
        # else:
        #	row.operator("pmc.set_default_path", text="", icon="DISK_DRIVE")

        row = col.row(align=False)
        row.prop(pmcp, "deselect_all")
        row = col.row(align=True)

        if not pmcp.folderset_list.values():
            box = row.box()
            boxcol = box.column()
            boxcol.scale_y = 0.7
            boxcol.label(text="")
            boxcol.label(text="No materials found,", icon="FILE_PARENT")
            boxcol.label(text="set path above or refresh")
            boxcol.label(text="")
        else:
            row.template_list("PMC_UL_foldersets", "",
                              pmcp, "folderset_list",
                              pmcp, "folderset_list_index")
        row = col.row(align=False)
        split = layout_split(row, factor=0.5)
        thumbnail_size = 6 if bpy.app.version >= (2, 80) else 5  # to keep flush
        splitrow = split.row()
        splitrow.scale_y = 0.5
        splitrow.template_icon_view(
            pmcp,
            'thumbnails',
            show_labels=False,  # Setting false as memory error shows odd characters
            scale=thumbnail_size,
        )
        splitrow.scale_y = 0.5
        splitcol = split.column(align=True)
        splitcol.label(text="Preview type")
        splitcol.prop(pmcp, "preview_type", text="")
        splitrow = splitcol.row(align=True)
        if itm is not None:
            ops = splitrow.operator("wm.url_open", text="View online")
            ops.url = poliigon_url + itm.urlend
        else:
            splitrow.operator("wm.url_open", text="View online")
            splitrow.enabled = False

        row = col.row(align=True)
        row.label(text="")
        row = col.row(align=True)

        n = len([itm for itm in pmcp.folderset_list.values() if itm.checked])
        row.enabled = n > 0
        row.scale_y = 1.25
        p = row.operator("pmc.load_materials_from_uilist",
                         text="Load {} Material{}".format(n, "" if n == 1 else "s"))

        row = col.row(align=True)
        p = row.operator("pmc.apply_material", text="Apply/Replace Selected Material")
        if pmcp.folderset_list:
            p.index = ind

        row = col.row(align=True)
        icon = "TRIA_DOWN" if pmcp.show_advanced else "TRIA_RIGHT"
        row.prop(pmcp, "show_advanced", icon=icon)
        if pmcp.show_advanced:
            row = col.row(align=True)
            box = row.box()
            box.scale_y = 1
            bcol = box.column(align=True)
            brow = bcol.row(align=True)
            brow.enabled = len([itm for itm in pmcp.folderset_list.values()
                                if itm.loaded]) > 0
            brow.operator("pmc.remove_unused_materials")
            bcol.prop(pmcs, "use_ao")
            bcol.prop(pmcs, "use_disp")
            bcol.prop(pmcs, "use_sixteenbit")
            bcol.prop(pmcs, "conform_uv")
        # box.prop(pmcs, "use_micro_displacements") # disabled due to micro displacement not working right with normal maps

        # Display selected material's path
        row = col.row()
        if itm != None:
            itmdir = os.path.dirname(itm.setpath)
            row.label(
                text="//" + \
                     itmdir.split(context.window_manager.pmc_texture_path)[-1] + \
                     os.path.sep)
        else:
            row.label(text="")

        # display warnings or other messages
        row = col.row()
        if itm != None and itm.status != "{}":
            status = json.loads(itm.status)
            for stat in list(status):
                box = col.box()
                boxcol = box.column()
                boxcol.scale_y = 0.7
                boxcol.label(text=stat, icon="ERROR")
                for stat_itm in status[stat]:
                    boxcol.label(text=stat_itm)
                # option to open link if relevant
                if stat == "Missing critical passes":
                    p = box.operator("wm.url_open", text="Download material from Poliigon")
                    p.url = poliigon_url + itm.urlend
                elif stat == "Specular workflow found":
                    p = box.operator("wm.url_open", text="Download material from Poliigon")
                    p.url = poliigon_url + itm.urlend
        elif len(wrns) > 0:
            box = col.box()
            box.label(text="Highlight warning row for info")

    # call built-in function with draw code/checks
    # addon_updater_ops.update_notice_box_ui(self, context)


def PMC_material_draw_append(self, context):
    """Appending to material panel, to be able to control texture scale"""
    layout = self.layout
    col = layout.column(align=True)
    mat = context.material

    if not mat:
        return
    if mat and mat.pmc_matprops.workflow != '':

        # get the mapping nodes, if found
        main_map = None
        overlay_map = None
        for n in mat.node_tree.nodes:
            if 'main_map' in n:  # ie n['main_map'] exists
                main_map = n

        if main_map:
            row = col.row(align=True)
            row.prop(main_map, "scale", text="Main scale")

    # if mat.pmc_matprops.use_micro_displacements:
    # 	# see if everything enabled, ie experimental,
    # 	# normal muted
    # 	#
    # 	layout.prop(mat.pmc_matprops,"use_micro_displacements")
    # 	# Need to setup handler
    # else:
    # 	layout.prop(mat.pmc_matprops,"use_micro_displacements")


# -----------------------------------------------------------------------------
# REGISTER / UNREGISTER
# -----------------------------------------------------------------------------


def register():
    """Register classes."""

    # panels changed name between blender 2.7x and 2.8x
    if hasattr(bpy.types, "Cycles_PT_context_material"):
        bpy.types.Cycles_PT_context_material.append(PMC_material_draw_append)
    elif hasattr(bpy.types, "CYCLES_PT_context_material"):
        bpy.types.CYCLES_PT_context_material.append(
            PMC_material_draw_append)
        if hasattr(bpy.types, 'EEVEE_MATERIAL_PT_context_material'):
            bpy.types.EEVEE_MATERIAL_PT_context_material.append(
                PMC_material_draw_append)



    bpy.utils.register_class(PMC_UL_foldersets)
    bpy.utils.register_class(PMC_PT_main_panel)


def unregister():
    """Unregister classes."""
    bpy.utils.unregister_class(PMC_UL_foldersets)
    bpy.utils.unregister_class(PMC_PT_main_panel)

    if hasattr(bpy.types, "Cycles_PT_context_material"):
        bpy.types.Cycles_PT_context_material.remove(PMC_material_draw_append)
    elif hasattr(bpy.types, "CYCLES_PT_context_material"):
        bpy.types.CYCLES_PT_context_material.remove(
            PMC_material_draw_append)
        bpy.types.EEVEE_MATERIAL_PT_context_material.remove(
            PMC_material_draw_append)
