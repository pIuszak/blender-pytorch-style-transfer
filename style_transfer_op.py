import bpy
# !pip install torch torchvision
# !pip install Pillow==4.0.0
# %matplotlib inline

# info of addon -------------------

# import matplotlib.pyplot as plt
import bpy_extras
import numpy as np
import platform


def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)


import sys

import bpy

wm = bpy.context.window_manager

# PROGRESS ------------------------
# https://www.youtube.com/watch?v=mRiTfLpRlRU
# # progress from [0 - 1000]


from pathlib import Path

str_path = "my_path"

import os
from bpy.types import Operator, Image

stream = os.popen('echo Returned output')
output = stream.read()
print(output)
import subprocess

# subprocess.call([data["om_points"], ">", diz['d']+"/points.xml"])


command = 'xd'
# command = "\"" + os.path.join(sys.exec_prefix, "bin\python.exe") + "\"" + " -m pip install matplotlib --user"
# stream = os.popen(command)
# output = stream.read()
# print(output)

# install_and_import('matplotlib')


if platform.system() == "Windows":

    print(" ================== basename ", sys.exec_prefix)
    command = "\"" + os.path.join(sys.exec_prefix,
                                  "bin\python.exe") + "\"" + " -m pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html --user"

elif platform.system() == "Linux":

    print(" ================== basename ", sys.exec_prefix)
    command = "\"" + "pip install torch torchvision -t " + os.path.join(sys.exec_prefix, "binpython3") + "\""

elif platform.system() == "Darwin":

    print(" ================== basename ", sys.exec_prefix)
    command = "\"" + "pip install torch torchvision -t " + os.path.join(sys.exec_prefix, "binpython3") + "\""

stream = os.popen(command)


# try:
#     import torch
# except ImportError:
#     subprocess.call([sys.executable, "-m", "pip", "install", 'torch==1.3.1+cpu', "-f",
#                      "https://download.pytorch.org/whl/torch_stable.html"])
# finally:
#     import torch


class StyleTransfer_OT_TextField(bpy.types.Operator):
    bl_idname = "view3d.textfield"
    bl_label = "Simple asd"
    bl_description = "Center 3d asd"

    def execute(self, context):
        # bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, location=(0, 0, 0))
        bpy.ops.import_test.some_data('INVOKE_DEFAULT')
        return {'FINISHED'}


class StyleTransfer_OT_Operator(bpy.types.Operator):
    bl_idname = "view3d.cursor_center"
    bl_label = "Simple operator"
    bl_description = "Center 3d cursor"

    content = ""
    style = ""
    steps = ""

    def im_convert(self, tensor):
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
        image = image.clip(0, 1)

        return image

    def execute(self, context):

        output = stream.read()
        print(output)

        import torch
        import PIL
        # from matplotlib import transforms
        from torch import optim
        from torchvision import models, transforms

        from time import time
        from itertools import chain
        from PIL import Image as Img

        start = time()
        print("0 execute started")
        # form example to remove
        # bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, location=(0, 0, 0))

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
        # content = load_image('C:/Projects/blender-pytorch-style-transfer/images/images/mfi.jpg').to(device)
        # style = load_image('C:\Projects\blender-pytorch-style-transfer\images\style/cubi.jpg').to(device)

        # self.content
        # print(self.content.strip() + "xd")
        print("style image loaded ")
        # self.style.replace('\\', '/')
        # print(self.style.strip())

        # content = bpy_extras.image_utils.load_image('images/content/mfi.jpg').to(device)
        # style = bpy_extras.image_utils.load_image('images/style/cubi.jpg').to

        # this load works in blender but is not compatibile with .to(device)
        # content = bpy.data.images.load("C:/Projects/blender-pytorch-style-transfer/images/content/mfi.jpg", check_existing=True).to(device)
        # print("content image loaded ")
        # style = bpy.data.images.load("C:/Projects/blender-pytorch-style-transfer/images/style/cubi.jpg", check_existing=True).to(device)

        self.content = bpy.path.abspath(self.content)
        self.content.replace('\\', '/')

        self.style = bpy.path.abspath(self.style)
        self.style.replace('\\', '/')
        print("XD " + self.content)
        print("XD " + self.style)

        c = load_image(self.content).to(device)
        # print("content image loaded ")
        s = load_image(self.style).to(device)

        # c = load_image(self.content.strip()).to(device)
        print("content image loaded ")
        # = load_image(self.content.strip()).to(device)
        print("style image loaded ")

        # content = bpy.data.images.load('images/content/mfi.jpg', False).to(device)
        # style = bpy.data.images.load('images/style/cubi.jpg', False).t

        # plt to preview
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(self.im_convert(content))
        ax1.axis("off")
        ax2.imshow(self.im_convert(style))
        ax2.axis("off")
        '''

        def get_features(image, model):

            layers = {'0': 'conv1_1',
                      '5': 'conv2_1',
                      '10': 'conv3_1',
                      '19': 'conv4_1',
                      '21': 'conv4_2',  # Content Extraction
                      '28': 'conv5_1'}

            features = {}

            for name, layer in model._modules.items():
                image = layer(image)
                if name in layers:
                    features[layers[name]] = image

            return features

        print("get_features defined ")
        content_features = get_features(c, vgg)
        print("content_features defined ")
        style_features = get_features(s, vgg)
        print("style_features defined ")

        def gram_matrix(tensor):
            _, d, h, w = tensor.size()
            tensor = tensor.view(d, h * w)
            gram = torch.mm(tensor, tensor.t())
            return gram

        print("gram_matrix defined ")
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
        print("style_grams defined ")
        style_weights = {'conv1_1': 1.,
                         'conv2_1': 0.75,
                         'conv3_1': 0.2,
                         'conv4_1': 0.2,
                         'conv5_1': 0.2}
        print("style_weights defined ")
        content_weight = 1  # alpha
        style_weight = 1e6  # beta

        target = c.clone().requires_grad_(True).to(device)
        print("target defined ")
        show_every = 1
        optimizer = optim.Adam([target], lr=0.003)

        steps = int(self.steps)
        # height, width, channels = im_convert(target).shape
        # image_array = np.empty(shape=(300, height, width, channels))
        capture_frame = int(steps) / 300
        counter = 0
        print("for loop started ")

        wm.progress_begin(0, steps)

        for ii in range(1, steps + 1):
            wm.progress_update(ii)
            # print("step ")
            target_features = get_features(target, vgg)
            # print("target_features defined ")
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
            # print("content_loss defined ")
            style_loss = 0

            for layer in style_weights:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                style_gram = style_grams[layer]
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                _, d, h, w = target_feature.shape
                style_loss += layer_style_loss / (d * h * w)

            total_loss = content_weight * content_loss + style_weight * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if ii % show_every == 0:
                print('Total loss: ', total_loss.item())
                print('Iteration: ', ii)
                # plt.imshow(im_convert(target))
                # plt.axis("off")
                # plt.show()

            if ii % capture_frame == 0:
                counter = counter + 1

        # start = time()
        # Image information. Change these to your liking.
        NAME = 'Procedural Image'
        WIDTH = 64
        HEIGHT = 64
        USE_ALPHA = True
        newImage = bpy.data.images.new(NAME, WIDTH, HEIGHT, alpha=USE_ALPHA)

        print("-----")
        wm.progress_end()
        # import scipy.misc
        # newImage = scipy.misc.toimage((self.im_convert(target)), cmin=0.0, cmax=...)

        # print(type(self.im_convert(target)))
        # newImage = Img.fromarray((self.im_convert(target)))
        im = Img.fromarray((self.im_convert(target) * 255).astype(np.uint8))

        # todo : this is workaround, to parse PIL Image to blender bpy.data
        # import subprocess
        # subprocess.check_call(["attrib", "-w", "output.jpg"])
        im.save(self.content)
        newImage = bpy.data.images.load(self.content)
        newImage.update()
        # newImage = Img.fromarray(np.uint8(cm.gist_earth(self.im_convert(target)) * 255))

        # newImage.update()

        # print('TIME TAKEN: %f seconds' % (time() - start))  # Outputs to the system console.

        # Make all UV/Image Editor views show the new image.
        for area in bpy.context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                for space in area.spaces:
                    if space.type == 'IMAGE_EDITOR':
                        space.image = newImage
        # image_array[counter] = self.im_convert(target)

        # All done.
        # experimantal video transfe
        '''
        import cv2

        frame_height, frame_width, _ = im_convert(target).shape
        vid = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

        for i in range(0, 300):
            img = image_array[i]
            img = img * 255
            img = np.array(img, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vid.write(img)

        vid.release()
        '''

        print('TIME TAKEN: %f seconds' % (time() - start))  # Outputs to the system console.
        return {'FINISHED'}


class GenerateOperator(Operator):
    bl_idname = "xd.generate"
    bl_label = "GENEREJT"
    string1 = ""
    string2 = ""

    def execute(self, context):
        # set_paths(self, context, self.string1, ".drl")
        global content
        global style

        content = self.string1
        style = self.string2

        # global style = self.string2
        return {'FINISHED'}
