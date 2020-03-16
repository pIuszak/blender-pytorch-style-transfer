import bpy
import time

class ProgressWidget(object):
    update_every = 0.2 # seconds

    widget_visible = False
    last_updated = 0

    @staticmethod
    def update_widget(context, force=False):
        sec_since_update = time.time() - ProgressWidget.last_updated

        if not force and sec_since_update < ProgressWidget.update_every:
            return

        # Update the top header
        for area in bpy.context.screen.areas:
            if area.type == 'INFO':
                area.tag_redraw()

        ProgressWidget.last_updated = time.time()

    @staticmethod
    def draw(self, context):
        if ProgressWidget.get_progress(context) < 100:

            # Shows the custom progress property as a "bar"
            self.layout.prop(context.scene, "ProgressWidget_progress", text="Progress", slider=True)

        else:
            ProgressWidget.hide()

    @staticmethod
    def create_progress_property():
        bpy.types.Scene.ProgressWidget_progress = bpy.props.IntProperty(default=0, min=0, max=100, step=1, subtype='PERCENTAGE')

    @staticmethod
    def set_progress(context, value):
        if ProgressWidget.widget_visible:
            context.scene.ProgressWidget_progress = value

    @staticmethod
    def get_progress(context):
        if ProgressWidget.widget_visible:
            return context.scene.ProgressWidget_progress
        else:
            return 0

    @staticmethod
    def show(context):
        if not ProgressWidget.widget_visible:
            ProgressWidget.create_progress_property()

            bpy.app.handlers.scene_update_pre.append(ProgressWidget.update_widget)
            bpy.types.INFO_HT_header.append(ProgressWidget.draw)

            ProgressWidget.widget_visible = True

            ProgressWidget.set_progress(context, 0)

    @staticmethod
    def hide():
        bpy.types.INFO_HT_header.remove(ProgressWidget.draw)
        bpy.app.handlers.scene_update_pre.remove(ProgressWidget.update_widget)

        ProgressWidget.widget_visible = False


# Creates a widget simulator in the text editor side panel
# The code below is only used for widget development

class CUSTOM_PT_testPanel(bpy.types.Panel):
    """Adds a custom panel to the TEXT_EDITOR"""
    bl_idname = 'TEXT_PT_testPanel'
    bl_space_type = "TEXT_EDITOR"
    bl_region_type = "UI"
    bl_label = "Progress Simulator"

    def draw(self, context):
        self.layout.operator("custom.show_progress_widget")
        self.layout.operator("custom.make_progress")


class CUSTOM_OT_show_progress_widget(bpy.types.Operator):
    bl_idname = "custom.show_progress_widget"
    bl_label="Show Widget in Header"

    def execute(self, context):

        ProgressWidget.show(context) # Call this to start showing progress

        return {'FINISHED'}

class CUSTOM_OT_make_progress(bpy.types.Operator):
    bl_idname = "custom.make_progress"
    bl_label="Make progress"

    def execute(self, context):
        new_progress_value = (ProgressWidget.get_progress(context) + 25) % 125

        # This line will update the progress widget
        # call it from your long-running modal operator (0-100)
        # When progress is 100 or more, the widget disappears
        ProgressWidget.set_progress(context, new_progress_value)

        return {'FINISHED'}

if __name__ == "__main__":
    bpy.utils.register_class(CUSTOM_PT_testPanel)
    bpy.utils.register_class(CUSTOM_OT_show_progress_widget)
    bpy.utils.register_class(CUSTOM_OT_make_progress)