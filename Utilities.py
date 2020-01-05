#Try this to batch load a list of images in a folder:

import bpy
from os.path import join, isfile
from os import listdir

imageObjects = []
imagePath = '/users/you/yourImagePath'
imgFiles  = [
    join( imagePath, fn )                    # Create full paths to images
    for fn in listdir( imagePath )           # For each item in the image folder
    if isfile( join( imagePath, fn ) )       # If the item is indeed a file
    and fn.lower().endswith(('.png','.jpg')) # Which ends with an image suffix (can add more types here if needed)
]

# Load entire list of images
for imgFile in imgFiles:
    # Add to image object list to use later
    imageObjects.append( bpy.data.images.load( imgFile ) )

# Load a specific index from the list (5th in the list in this example)
imgObj = bpy.data.images.load( imgFiles[4] )