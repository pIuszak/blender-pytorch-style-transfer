import bpy
#!pip install torch torchvision
# !pip install Pillow==4.0.0
# %matplotlib inline

# info of addon -------------------

# import matplotlib.pyplot as plt
import numpy as np
import torch
import PIL
# from matplotlib import transforms
from torch import optim
from torchvision import models, transforms


class StyleTransfer_OT_TextField(bpy.types.Operator):
    bl_idname = "view3d.textfield"
    bl_label = "Simple asd"
    bl_description = "Center 3d asd"

    def execute(self, context):
        bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, location=(0, 0, 0))
        return {'FINISHED'}



class StyleTransfer_OT_Operator(bpy.types.Operator):
    bl_idname = "view3d.cursor_center"
    bl_label = "Simple operator"
    bl_description = "Center 3d cursor"



    def im_convert(tensor):
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
        image = image.clip(0, 1)

        return image

    def execute(self, context):
        print("0 execute started")
        #form example to remove
        #bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, location=(0, 0, 0))

        vgg = models.vgg19(pretrained=True).features

        for param in vgg.parameters():
            param.requires_grad_(False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg.to(device)

        print("1 resources imported")

        def load_image(img_path, max_size=400, shape=None):
            image = PIL.Image.open(img_path).convert('RGB')
            if max(image.size) > max_size:
                size = max_size
            else:
                size = max(image.size)

            if shape is not None:  
                size = shape

            in_transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))])

            image = in_transform(image).unsqueeze(0)

            return image
        # load content from file
        # todo: loading more
        content = load_image('C:/Projects/blender-pytorch-style-transfer/images/images/mfi.jpg').to(device)
        style = load_image('images/cubi.jpg').to(device)

        print("2 image loaded")
        print("Works So far !!!")
        return {'FINISHED'}

