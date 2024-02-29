import cv2
import os

# Define the path to the directory containing the images
image_dir = "C:/Personal_Data/VT SEM2/Human Robot Interaction/Final Github/New Folder/images/"
from PIL import Image

image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Sort the image filenames alphabetically
image_files.sort()

# Create a list of image objects from the files
images = [Image.open(os.path.join(image_dir, f)) for f in image_files]

# Save the list of images as an animated GIF
images[0].save('output.gif', save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)