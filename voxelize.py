import bpy

# set source and create target
sourceName = bpy.context.object.name
source = bpy.data.objects[sourceName]

bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked": False, "mode": 'TRANSLATION'})
bpy.context.object.name = sourceName + "_Voxelized"
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
bpy.ops.object.convert(target='MESH')

source.hide_viewport = True
source.hide_render = True

targetName = bpy.context.object.name
target = bpy.data.objects[targetName]

# turn the target into blocks


bpy.ops.object.modifier_add(type='REMESH')
bpy.context.object.modifiers["Remesh"].mode = 'BLOCKS'


bpy.context.object.modifiers["Remesh"].octree_depth = 7

bpy.context.object.modifiers["Remesh"].use_remove_disconnected = False
bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Remesh")

# tranfer UV'to target

bpy.ops.object.modifier_add(type='DATA_TRANSFER')
bpy.context.object.modifiers["DataTransfer"].use_loop_data = True
bpy.context.object.modifiers["DataTransfer"].data_types_loops = {'UV'}
bpy.context.object.modifiers["DataTransfer"].loop_mapping = 'POLYINTERP_NEAREST'
bpy.context.object.modifiers["DataTransfer"].object = source
bpy.ops.object.datalayout_transfer(modifier="DataTransfer")
bpy.ops.object.modifier_apply(apply_as='DATA', modifier="DataTransfer")

# reduce faces to single color

bpy.ops.object.editmode_toggle()
bpy.ops.mesh.select_mode(type='FACE')
bpy.context.area.ui_type = 'UV'
bpy.context.scene.tool_settings.use_uv_select_sync = False
bpy.context.space_data.uv_editor.sticky_select_mode = 'DISABLED'
bpy.context.scene.tool_settings.uv_select_mode = 'FACE'
bpy.context.space_data.pivot_point = 'INDIVIDUAL_ORIGINS'
bpy.ops.mesh.select_all(action='DESELECT')


count = 0
while count < 100:
    bpy.ops.mesh.select_random(percent=count + 1, seed=count)
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.transform.resize(value=(0.001,0.001,0.001))
    bpy.ops.mesh.hide(unselected=False)
    count += 1

# return to previous context
bpy.context.area.ui_type = 'VIEW_3D'
bpy.ops.mesh.reveal()
bpy.ops.object.editmode_toggle()
