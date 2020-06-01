import bpy_extras

import platform
import sys
import bpy
import numpy as np

wm = bpy.context.window_manager
str_path = "my_path"

import os
from bpy.types import Operator, Image


def exec(cmd):
    stream = os.popen(cmd)
    output = stream.read()
    print(output)

def config_windows():
    print("Windows Config")
    # command("\"" + os.path.join(sys.exec_prefix,
    #                             "bin\python.exe") + "\"" + " -m ensurepip")
    command = "\"" + os.path.join(sys.exec_prefix,
                                  "bin\python.exe") + "\"" + " -m pip install " + "\"" + bpy.utils.user_resource(
        'SCRIPTS',
        "addons") + "\\" + "blender-pytorch-style-transfer\\windows" + "\\" + "torch-1.5.0-cp37-cp37m-win_amd64.whl" + "\" --user"
    command.replace('\\', '/')
    exec(command)

    command = "\"" + os.path.join(sys.exec_prefix,
                                  "bin\python.exe") + "\"" + " -m pip install " + "\"" + bpy.utils.user_resource(
        'SCRIPTS',
        "addons") + "\\" + "blender-pytorch-style-transfer\\windows" + "\\" + "torchvision-0.6.0-cp37-cp37m-win_amd64.whl" + "\" --user"
    command.replace('\\', '/')
    exec(command)

    command = "\"" + os.path.join(sys.exec_prefix,
                                  "bin\python.exe") + "\"" + " -m pip install Pillow --user"
    command.replace('\\', '/')
    exec(command)

def config_linux():
    print("Linux Config")
    command = "\"" + os.path.join(sys.exec_prefix,
                                "bin/python3.7m") + "\"" + " -m ensurepip --user"
    exec(command)
    command = "\"" + os.path.join(sys.exec_prefix,
                                  "bin/python3.7m") + "\"" + " -m pip install " + "\"" + bpy.utils.user_resource(
        'SCRIPTS',
        "addons") + "/" + "blender-pytorch-style-transfer/linux" + "/" + "torch-1.5.0-cp37-cp37m-linux_x86_64.whl" + "\" --user"
    command.replace('\\', '/')
    exec(command)

    command = "\"" + os.path.join(sys.exec_prefix,
                                  "bin/python3.7m") + "\"" + " -m pip install " + "\"" + bpy.utils.user_resource(
        'SCRIPTS',
        "addons") + "/" + "blender-pytorch-style-transfer/linux" + "/" + "torchvision-0.6.0-cp37-cp37m-linux_x86_64.whl" + "\" --user"
    command.replace('\\', '/')
    exec(command)

    command = "\"" + os.path.join(sys.exec_prefix,
                                  "bin/python3.7m") + "\"" + " -m pip install Pillow --user"
    command.replace('\\', '/')
    exec(command)

print(str(platform.system()))

if platform.system() == "Windows":
    config_windows()

if platform.system() == "Linux":
    config_linux()


class StyleTransfer_OT_TextField(bpy.types.Operator):
    bl_idname = "view3d.textfield"
    bl_label = "Style Transfer"
    bl_description = "implement the Neural-Style algorithm developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge. Neural-Style, or Neural-Transfer, allows you to take an image and reproduce it with a new artistic style. "

    def execute(self, context):
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

        import torch
        import PIL


        # from matplotlib import transforms
        from torch import optim
        from torchvision import models, transforms

        from time import time
        from itertools import chain
        from PIL import Image as Img
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        vgg = models.vgg19(pretrained=True).features

        start = time()
        steps = int(self.steps)
        wm.progress_begin(0, steps)

        for param in vgg.parameters():
            param.requires_grad_(False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg.to(device)

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
        self.content = bpy.path.abspath(self.content)
        self.content.replace('\\', '/')

        self.style = bpy.path.abspath(self.style)
        self.style.replace('\\', '/')

        c = load_image(self.content).to(device)
        s = load_image(self.style).to(device)

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

        content_features = get_features(c, vgg)
        style_features = get_features(s, vgg)

        def gram_matrix(tensor):
            _, d, h, w = tensor.size()
            tensor = tensor.view(d, h * w)
            gram = torch.mm(tensor, tensor.t())
            return gram

        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
        style_weights = {'conv1_1': 1.,
                         'conv2_1': 0.75,
                         'conv3_1': 0.2,
                         'conv4_1': 0.2,
                         'conv5_1': 0.2}

        content_weight = 1  # alpha
        style_weight = 1e6  # beta

        target = c.clone().requires_grad_(True).to(device)
        show_every = 1
        # set up optimizer
        optimizer = optim.Adam([target], lr=0.003)

        capture_frame = int(steps) / 300
        counter = 0

        # iterations
        for j in range(1, steps + 1):
            wm.progress_update(j)
            target_features = get_features(target, vgg)
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
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

        # Image information. Change these to your liking.
        NAME = 'Procedural Image'
        WIDTH = 64
        HEIGHT = 64
        USE_ALPHA = True
        newImage = bpy.data.images.new(NAME, WIDTH, HEIGHT, alpha=USE_ALPHA)

        wm.progress_end()

        im = Img.fromarray((self.im_convert(target) * 255).astype(np.uint8))

        # parse PIL Image to blender bpy.data
        im.save(self.content)
        newImage = bpy.data.images.load(self.content)
        newImage.update()

        # Make all UV/Image Editor views show the new image.
        for area in bpy.context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                for space in area.spaces:
                    if space.type == 'IMAGE_EDITOR':
                        space.image = newImage

        print('TIME TAKEN: %f seconds' % (time() - start))  # Outputs to the system console.
        return {'FINISHED'}


class GenerateOperator(Operator):
    bl_idname = "op.generate"
    bl_label = "GENERATE"
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
